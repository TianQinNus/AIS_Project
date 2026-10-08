"""Local, reviewed document retrieval. No vehicle values are stored in this index."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KINDS = {"maintenance", "project", "dataset"}
STOPWORDS = set("a an the and or is are of to in for on with this that vehicle vehicles truck trucks".split())


def tokens(text):
    """English identifiers plus Chinese characters/bigrams for lexical matching."""
    words = [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run)
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words


def load_catalog(root):
    root = Path(root).resolve()
    entries = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog.json must contain a non-empty list of reviewed documents")
    seen = set()
    for entry in entries:
        path = (root / entry["source"]).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Document paths must stay inside the knowledge directory")
        if entry["source"] in seen:
            raise ValueError("Duplicate document in catalog")
        seen.add(entry["source"])
        if entry.get("kind") not in KINDS or entry.get("reviewed") is not True:
            raise ValueError("Every indexed document needs a valid kind and reviewed=true")
        if not entry.get("title") or not entry.get("applicability") or not entry.get("version"):
            raise ValueError("Every document needs title, applicability and version")
        if not path.is_file():
            raise FileNotFoundError(path)
    return entries


def fingerprint(root, entries):
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode())
    for entry in entries:
        digest.update(entry["source"].encode())
        digest.update((Path(root) / entry["source"]).read_bytes())
    return digest.hexdigest()


def document_units(path):
    """Keep page/paragraph/section locations instead of flattening the whole corpus."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        for page, obj in enumerate(PdfReader(path).pages, start=1):
            text = obj.extract_text() or ""
            if not text.strip():
                raise ValueError(f"{path.name}, page {page}: no extractable text; OCR or curate it first")
            yield {"page": page}, text
    elif suffix == ".docx":
        from docx import Document
        # Paragraph and table order follows the document body.
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        doc = Document(path)
        for i, element in enumerate(doc.element.body, start=1):
            if element.tag == qn("w:p"):
                text = Paragraph(element, doc).text
            elif element.tag == qn("w:tbl"):
                text = "\n".join(" | ".join(c.text for c in row.cells) for row in Table(element, doc).rows)
            else:
                continue
            if text.strip():
                yield {"body_block": i}, text
    elif suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".md":
            heading, start, lines = "Document", 1, []
            for i, line in enumerate(text.splitlines(), start=1):
                if re.match(r"^#{1,6}\s", line):
                    if lines:
                        yield {"section": heading, "line": start}, "\n".join(lines)
                    heading, start, lines = line.lstrip("# "), i, [line]
                else:
                    lines.append(line)
            if lines:
                yield {"section": heading, "line": start}, "\n".join(lines)
        else:
            yield {"line": 1}, text
    else:
        raise ValueError(f"Unsupported source format: {suffix}")


def load_encoder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def normalized_encode(encoder, texts):
    vectors = np.asarray(encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned zero-length vectors")
    return vectors / norms


def build_index(root, index_path, model_name=DEFAULT_MODEL, *, encoder=None):
    root, index_path = Path(root), Path(index_path)
    catalog = load_catalog(root)
    before = fingerprint(root, catalog)
    encoder = encoder if encoder is not None else load_encoder(model_name)
    # This model has a short context window; character-based 1000-char chunks truncate silently.
    budget = int(encoder.max_seq_length) - encoder.tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 16:
        raise ValueError("Embedding token budget is too small")
    overlap = min(24, budget // 4)
    chunks = []
    for entry in catalog:
        count = 0
        for location, text in document_units(root / entry["source"]):
            if not text.strip():
                continue
            ids = encoder.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            for offset in range(0, len(ids), budget - overlap):
                content = encoder.tokenizer.decode(ids[offset:offset + budget], skip_special_tokens=True).strip()
                if not content:
                    continue
                identity = json.dumps([entry, location, offset, content], sort_keys=True, ensure_ascii=False)
                chunks.append({**entry, **location, "token_start": offset,
                               "citation_id": "K-" + hashlib.sha256(identity.encode()).hexdigest()[:12],
                               "text": content})
                count += 1
                if offset + budget >= len(ids):
                    break
        if count == 0:
            raise ValueError(f"Empty document: {entry['source']}")
    vectors = normalized_encode(encoder, [chunk["text"] for chunk in chunks])
    if fingerprint(root, load_catalog(root)) != before:
        raise ValueError("Knowledge changed during indexing; rebuild")
    metadata = {"schema_version": 1, "model": model_name, "fingerprint": before,
                "created_at": datetime.now(timezone.utc).isoformat(), "chunk_token_budget": budget,
                "catalog": catalog}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # One atomic archive avoids mixing text from one rebuild with vectors from another.
    with tempfile.NamedTemporaryFile(dir=index_path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, vectors=vectors,
                                chunks=json.dumps(chunks, ensure_ascii=False),
                                metadata=json.dumps(metadata, ensure_ascii=False))
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, index_path)
    return {"documents": len(catalog), "chunks": len(chunks), "index": str(index_path)}


class KnowledgeRetriever:
    def __init__(self, root, index_path, *, encoder=None, min_similarity=0.32):
        self.root = Path(root)
        with np.load(index_path, allow_pickle=False) as data:
            self.chunks = json.loads(str(data["chunks"].item()))
            self.metadata = json.loads(str(data["metadata"].item()))
            self.vectors = np.asarray(data["vectors"], dtype=np.float32)
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported index schema; rebuild")
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.chunks) or not np.isfinite(self.vectors).all():
            raise ValueError("Corrupt knowledge index; rebuild")
        self._check_current()
        self.encoder = encoder if encoder is not None else load_encoder(self.metadata["model"])
        self.min_similarity = float(min_similarity)
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
        self.terms = [Counter(tokens(chunk["text"])) for chunk in self.chunks]
        self.lengths = np.array([sum(terms.values()) for terms in self.terms], dtype=float)
        self.avg_length = max(1.0, float(self.lengths.mean()))
        self.df = Counter(term for terms in self.terms for term in terms)

    def _check_current(self):
        entries = load_catalog(self.root)
        if fingerprint(self.root, entries) != self.metadata["fingerprint"]:
            raise ValueError("Knowledge changed since indexing; run build_knowledge.py again")

    def _bm25(self, query):
        scores = np.zeros(len(self.chunks), dtype=float)
        for term in set(tokens(query)):
            frequency = self.df.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(self.chunks) - frequency + 0.5) / (frequency + 0.5))
            for i, terms in enumerate(self.terms):
                tf = terms.get(term, 0)
                if tf:
                    scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avg_length))
        return scores

    def search(self, query, *, context="", top_k=5, kind=None):
        query = query.strip()
        if not query or len(query) > 6000:
            raise ValueError("Query must have 1-6000 characters")
        if not 1 <= top_k <= 10 or (kind is not None and kind not in KINDS):
            raise ValueError("Invalid retrieval options")
        self._check_current()
        encoded = normalized_encode(self.encoder, [query] + ([context] if context else []))
        q = encoded[0] if not context else 0.7 * encoded[0] + 0.3 * encoded[1]
        norm = np.linalg.norm(q)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Query embedding is invalid")
        q /= norm
        if q.shape[0] != self.vectors.shape[1]:
            raise ValueError("Embedding dimensions do not match the index")
        cosine = self.vectors @ q
        lexical = self._bm25(query + " " + context)
        eligible = [i for i, chunk in enumerate(self.chunks)
                    if (kind is None or chunk["kind"] == kind) and cosine[i] >= self.min_similarity]
        # Reciprocal rank fusion: scores have different scales, so combine ranks.
        fusion = Counter()
        for ranking in (sorted(eligible, key=lambda i: (-cosine[i], i)),
                        sorted([i for i in eligible if lexical[i] > 0], key=lambda i: (-lexical[i], i))):
            for rank, i in enumerate(ranking, start=1):
                fusion[i] += 1.0 / (60 + rank)
        hits = []
        per_location = Counter()
        for i in sorted(eligible, key=lambda i: (-fusion[i], -cosine[i], i)):
            chunk = self.chunks[i]
            location = (chunk["source"], chunk.get("page"), chunk.get("section"), chunk.get("body_block"))
            if per_location[location] >= 2:
                continue
            hits.append({**chunk, "similarity": round(float(cosine[i]), 4),
                         "retrieval_score": round(float(fusion[i]), 6)})
            per_location[location] += 1
            if len(hits) == top_k:
                break
        return {"success": True, "query": query, "hits": hits,
                "maintenance_evidence_found": any(hit["kind"] == "maintenance" for hit in hits),
                "note": "Similarity ranks relevance, not factual correctness or mechanical applicability."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--root", type=Path, default=BASE_DIR / "knowledge")
    parser.add_argument("--index", type=Path, default=BASE_DIR / "knowledge_index" / "index.npz")
    parser.add_argument("--kind", choices=sorted(KINDS))
    args = parser.parse_args()
    retriever = KnowledgeRetriever(args.root, args.index,
                                   min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.32")))
    print(json.dumps(retriever.search(args.query, kind=args.kind), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
