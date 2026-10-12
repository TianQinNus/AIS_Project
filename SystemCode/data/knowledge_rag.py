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
from sqlalchemy import text as sql_text
from database import create_postgres_engine
from pg_knowledge_store import json_value, publish
from urllib.parse import urlparse
from datetime import date

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
KINDS = {"maintenance", "policy", "project", "dataset"}
TRUSTED_HOSTS = {"government": {"www.gov.uk", "www.hse.gov.uk", "onemotoring.lta.gov.sg"},
                 "research": {"arxiv.org", "www.nature.com", "pmc.ncbi.nlm.nih.gov"},
                 "manufacturer": {"www.scania.com"}}
STOPWORDS = set("a an the and or is are of to in for on with this that vehicle vehicles truck trucks".split())


def expand_query(text):
    # Local bilingual search vocabulary only; never adds knowledge or policy facts.
    aliases = {"货车": "goods vehicle", "轮胎": "tyres", "制动": "brakes", "漏气": "air leaks",
               "气压": "air pressure", "警告灯": "warning lights", "盲区": "blind spot",
               "检验": "inspection", "定期": "periodic", "缺陷": "defects", "车龄": "vehicle age"}
    return text + " " + " ".join(value for key,value in aliases.items() if key in text)


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
        if entry.get("status") != "active":
            raise ValueError("Only active sources may enter the database")
        for key in ("sha256", "publisher", "authority", "evidence_type", "jurisdiction", "verified_on", "review_due"):
            if not entry.get(key):
                raise ValueError(f"Missing provenance field: {key}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Source hash changed; verify and refresh the catalog before rebuilding")
        if date.fromisoformat(entry["review_due"]) < date.today():
            raise ValueError("Source review overdue; refresh and review before rebuilding")
        date.fromisoformat(entry["verified_on"])
        if entry["kind"] == "project":
            if entry["authority"] != "project" or entry["evidence_type"] != "project_policy":
                raise ValueError("Project sources must be identified as project policy")
        else:
            url = urlparse(entry.get("source_url", ""))
            if url.scheme != "https" or url.hostname not in TRUSTED_HOSTS.get(entry["authority"], set()):
                raise ValueError("External sources require a trusted primary HTTPS publisher")
            original = (root / entry.get("original_source", "")).resolve()
            if not original.is_relative_to(root) or not original.is_file():
                raise ValueError("Original source must be a file inside the knowledge directory")
            if hashlib.sha256(original.read_bytes()).hexdigest() != entry.get("original_sha256"):
                raise ValueError("Original source hash mismatch")
            if entry["kind"] == "policy" and entry["authority"] != "government":
                raise ValueError("Policy evidence requires a government source")
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
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for unit in data["units"]:
            yield {key: value for key, value in unit.items() if key != "text"}, unit["text"]
    elif suffix == ".pdf":
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


def load_encoder(model_name, *, local_only=True):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, local_files_only=local_only, device="cpu")


def normalized_encode(encoder, texts):
    vectors = np.asarray(encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned zero-length vectors")
    return vectors / norms


def build_index(root, model_name=DEFAULT_MODEL, *, engine=None, encoder=None, allow_model_download=False):
    root = Path(root)
    engine = engine if engine is not None else create_postgres_engine()
    catalog = load_catalog(root)
    before = fingerprint(root, catalog)
    encoder = encoder if encoder is not None else load_encoder(model_name, local_only=not allow_model_download)
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
            offsets = None
            if getattr(encoder.tokenizer, "is_fast", False):
                offsets = encoder.tokenizer(text, add_special_tokens=False, truncation=False,
                                            return_offsets_mapping=True)["offset_mapping"]
            offset = 0
            while offset < len(ids):
                if offsets:
                    end = min(offset + budget, len(ids)) - 1
                    content = text[offsets[offset][0]:offsets[end][1]].strip()
                else:
                    content = encoder.tokenizer.decode(ids[offset:offset + budget], skip_special_tokens=True).strip()
                if offsets:
                    while len(encoder.tokenizer.encode(content, add_special_tokens=False, truncation=False)) > budget:
                        end -= 1
                        if end < offset:
                            raise ValueError("Unable to preserve source text within embedding token budget")
                        content = text[offsets[offset][0]:offsets[end][1]].strip()
                if not content:
                    offset += max(1, budget - overlap)
                    continue
                identity = json.dumps([entry, location, offset, content], sort_keys=True, ensure_ascii=False)
                chunks.append({**entry, **location, "token_start": offset,
                               "token_end": end + 1 if offsets else min(offset + budget, len(ids)),
                               "citation_id": "K-" + hashlib.sha256(identity.encode()).hexdigest()[:12],
                               "text": content})
                count += 1
                token_end = end + 1 if offsets else min(offset + budget, len(ids))
                if token_end >= len(ids):
                    break
                offset = max(offset + 1, token_end - overlap)
        if count == 0:
            raise ValueError(f"Empty document: {entry['source']}")
    vectors = normalized_encode(encoder, [chunk["text"] for chunk in chunks])
    if fingerprint(root, load_catalog(root)) != before:
        raise ValueError("Knowledge changed during indexing; rebuild")
    metadata = {"schema_version": 3, "model": model_name, "fingerprint": before,
                "created_at": datetime.now(timezone.utc).isoformat(), "chunk_token_budget": budget,
                "dimension": int(vectors.shape[1]), "chunk_count": len(chunks), "catalog": catalog}
    archives = {}
    for entry in catalog:
        extracted = (root / entry["source"]).read_bytes()
        original = (root / entry.get("original_source", entry["source"])).read_bytes()
        if (hashlib.sha256(extracted).hexdigest() != entry["sha256"]
                or hashlib.sha256(original).hexdigest() != entry.get("original_sha256", entry["sha256"])):
            raise ValueError("Source changed before database publication; rebuild")
        archives[entry["source"]] = (extracted, original)
    version = publish(engine, metadata, chunks, vectors, archives)
    return {"documents": len(catalog), "chunks": len(chunks), "backend": "postgresql_pgvector",
            "version": version, "model": model_name, "dimension": int(vectors.shape[1])}


class KnowledgeRetriever:
    def __init__(self, *, engine=None, encoder=None, min_similarity=0.32):
        self.engine = engine if engine is not None else create_postgres_engine()
        self._load_database()
        self.encoder = encoder if encoder is not None else load_encoder(self.metadata["model"])
        self.min_similarity = float(min_similarity)
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
        self._prepare_terms()

    def _load_database(self):
        with self.engine.connect().execution_options(isolation_level="REPEATABLE READ") as db:
            db.execute(sql_text("SET TRANSACTION READ ONLY"))
            self.version = db.execute(sql_text("SELECT active_version FROM rag.kb_state WHERE id=1")).scalar()
            if not self.version:
                raise ValueError("No active PostgreSQL knowledge version; build or migrate first")
            params = {"version": self.version}
            self.metadata = json_value(db.execute(sql_text("SELECT metadata FROM rag.kb_versions WHERE version=:version"), params).scalar_one())
            if self.metadata.get("schema_version") != 3 or self.metadata["dimension"] != 384:
                raise ValueError("Unsupported pgvector knowledge schema or embedding dimension")
            documents, archives = {}, {}
            for source,meta,extracted,original in db.execute(sql_text("SELECT source,metadata,extracted,original FROM rag.kb_documents WHERE version=:version"),params):
                entry = json_value(meta)
                if hashlib.sha256(extracted).hexdigest() != entry["sha256"] or hashlib.sha256(original).hexdigest() != entry.get("original_sha256",entry["sha256"]):
                    raise ValueError("Database source hash mismatch")
                documents[source] = entry
                archives[source] = extracted
            catalog = self.metadata["catalog"]
            if documents != {entry["source"]:entry for entry in catalog}:
                raise ValueError("Database catalog provenance mismatch")
            digest = hashlib.sha256(json.dumps(catalog,sort_keys=True,ensure_ascii=False).encode())
            for entry in catalog:
                digest.update(entry["source"].encode())
                digest.update(archives[entry["source"]])
            if digest.hexdigest() != self.metadata["fingerprint"]:
                raise ValueError("Database catalog integrity failure")
            self.chunks, vectors = [], []
            rows = db.execute(sql_text("""SELECT ordinal,source,payload,vector_blob,vector_sha256,text_sha256,embedding::text
                FROM rag.kb_chunks WHERE version=:version ORDER BY ordinal"""), params)
            for ordinal,source,payload,blob,vector_hash,text_hash,embedding in rows:
                chunk = json_value(payload)
                if ordinal != len(self.chunks) or source not in documents or any(chunk.get(k)!=v for k,v in documents[source].items()):
                    raise ValueError("Database chunk provenance mismatch")
                if hashlib.sha256(blob).hexdigest()!=vector_hash or hashlib.sha256(chunk["text"].encode()).hexdigest()!=text_hash:
                    raise ValueError("Database chunk integrity failure")
                vector = np.frombuffer(blob,dtype="<f4")
                stored = np.asarray(json.loads(embedding),dtype=np.float32)
                if not np.array_equal(vector,stored):
                    raise ValueError("pgvector embedding integrity failure")
                vectors.append(vector)
                self.chunks.append(chunk)
        self.vectors = np.asarray(vectors,dtype=np.float32)  # audit only, never query dot products
        if (self.vectors.shape != (self.metadata["chunk_count"],384) or not len(self.chunks)
                or not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors,axis=1),1,atol=1e-4)):
            raise ValueError("Invalid database vectors or chunk count")

    def _check_current(self):
        with self.engine.connect() as db:
            version = db.execute(sql_text("SELECT active_version FROM rag.kb_state WHERE id=1")).scalar_one()
        if version != self.version:
            model = self.metadata["model"]
            self._load_database()
            if model != self.metadata["model"]:
                self.encoder = load_encoder(self.metadata["model"])
            self._prepare_terms()

    def _similarities(self, query_vector):
        # pgvector performs exact cosine distance inside PostgreSQL.
        literal = "["+",".join(format(float(v),".17g") for v in query_vector)+"]"
        cosine = np.full(len(self.chunks), -np.inf, dtype=float)
        with self.engine.connect() as db:
            rows = db.execute(sql_text("""SELECT ordinal,1-(embedding OPERATOR(public.<=>) CAST(:query AS public.vector)) AS similarity
                FROM rag.kb_chunks WHERE version=:version ORDER BY embedding OPERATOR(public.<=>) CAST(:query AS public.vector),ordinal"""),
                {"query":literal,"version":self.version})
            for ordinal,similarity in rows:
                if ordinal < len(cosine) and similarity is not None:
                    cosine[ordinal] = float(similarity)
        return cosine

    def _prepare_terms(self):
        self.terms = [Counter(tokens(chunk["text"])) for chunk in self.chunks]
        self.lengths = np.array([sum(terms.values()) for terms in self.terms], dtype=float)
        self.avg_length = max(1.0, float(self.lengths.mean()))
        self.df = Counter(term for terms in self.terms for term in terms)

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

    def search(self, query, *, context="", top_k=5, kind=None, jurisdiction="SG"):
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
        cosine = self._similarities(q)
        lexical = self._bm25(expand_query(query + " " + context))
        eligible = [i for i, chunk in enumerate(self.chunks)
                    if (kind is None or chunk["kind"] == kind) and cosine[i] >= self.min_similarity
                    and chunk.get("status") == "active"
                    and date.fromisoformat(chunk["review_due"]) >= date.today()
                    and (chunk["kind"] != "policy" or chunk["jurisdiction"] == jurisdiction)]
        # Reciprocal rank fusion: scores have different scales, so combine ranks.
        fusion = Counter()
        for ranking in (sorted(eligible, key=lambda i: (-cosine[i], i)),
                        sorted([i for i in eligible if lexical[i] > 0], key=lambda i: (-lexical[i], i))):
            for rank, i in enumerate(ranking, start=1):
                fusion[i] += 1.0 / (60 + rank)
        # For explicit frequency questions, prefer the original table row for the
        # named vehicle type. This prevents semantically similar taxi/car rows
        # from displacing the requested goods-vehicle row.
        subject = None
        lower = query.lower()
        if any(term in lower for term in ("多久", "频率", "周期", "frequency", "how often", "periodic")):
            for phrases, label in [(("货车", "goods vehicle", "truck"), "Goods Vehicle"),
                                   (("拖车", "trailer"), "Trailer"),
                                   (("摩托车", "motorcycle"), "Motorcycle/Scooter"),
                                   (("私召车", "private hire"), "Chauffeured Private Hire Car ^"),
                                   (("出租车", "taxi"), "Taxi 1")]:
                if any(term in lower for term in phrases):
                    subject = label
                    break
        if subject:
            for i in eligible:
                if (self.chunks[i]["kind"] == "policy"
                        and self.chunks[i].get("table_subject") == subject):
                    fusion[i] += 0.04
        hits = []
        per_location = Counter()
        for i in sorted(eligible, key=lambda i: (-fusion[i], -cosine[i], i)):
            chunk = self.chunks[i]
            location = (chunk["source"], chunk.get("page"), chunk.get("section"), chunk.get("block"), chunk.get("body_block"))
            if per_location[location] >= 2:
                continue
            hits.append({**chunk, "similarity": round(float(cosine[i]), 4),
                         "retrieval_score": round(float(fusion[i]), 6)})
            per_location[location] += 1
            if len(hits) == top_k:
                break
        return {"success": True, "query": query, "hits": hits,
                "database_version": self.version, "retrieval_backend": "postgresql_pgvector",
                "policy_jurisdiction": jurisdiction,
                "excluded_overdue_documents": sorted({c["source"] for c in self.chunks if date.fromisoformat(c["review_due"]) < date.today()}),
                "maintenance_evidence_found": any(hit["kind"] == "maintenance" for hit in hits),
                "note": "Similarity ranks relevance, not factual correctness or mechanical applicability."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--kind", choices=sorted(KINDS))
    args = parser.parse_args()
    retriever = KnowledgeRetriever(
                                   min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.32")))
    print(json.dumps(retriever.search(args.query, kind=args.kind), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
