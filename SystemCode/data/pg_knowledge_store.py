"""Transactional pgvector persistence; no business-table writes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from sqlalchemy import text
from database import create_postgres_engine


def json_value(value):
    return json.loads(value) if isinstance(value, str) else value


def initialize(engine):
    with engine.begin() as connection:
        # Refuse an extension in an unexpected schema rather than relocate it.
        connection.execute(text("SELECT pg_advisory_xact_lock(47110384)"))
        ddl = (Path(__file__).parent / "sql/001_rag_pgvector.sql").read_text()
        for statement in ddl.split(";"):
            if statement.strip():
                connection.execute(text(statement))


def publish(engine, metadata, chunks, vectors, archives):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all():
        raise ValueError("The current pgvector schema requires finite 384-dimensional embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Embeddings must be normalized")
    if len({c["citation_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate knowledge citation")
    metadata = dict(metadata, schema_version=3, dimension=384, chunk_count=len(chunks))
    version = hashlib.sha256((json.dumps({k:v for k,v in metadata.items() if k != "created_at"}, sort_keys=True, ensure_ascii=False)
                              + json.dumps(chunks, sort_keys=True, ensure_ascii=False)).encode()
                             + vectors.astype("<f4").tobytes()).hexdigest()
    if not chunks:
        raise ValueError("Cannot publish an empty knowledge version")
    documents = {entry["source"]:entry for entry in metadata["catalog"]}
    digest = hashlib.sha256(json.dumps(metadata["catalog"],sort_keys=True,ensure_ascii=False).encode())
    for entry in metadata["catalog"]:
        digest.update(entry["source"].encode())
        digest.update(archives[entry["source"]][0])
    if digest.hexdigest() != metadata["fingerprint"]:
        raise ValueError("Knowledge catalog fingerprint mismatch")
    for chunk in chunks:
        if chunk["source"] not in documents or any(chunk.get(k)!=v for k,v in documents[chunk["source"]].items()):
            raise ValueError("Knowledge chunk provenance mismatch")
    initialize(engine)
    with engine.begin() as db:
        db.execute(text("SELECT pg_advisory_xact_lock(47110384)"))
        exists = db.execute(text("SELECT 1 FROM rag.kb_versions WHERE version=:version"), {"version":version}).scalar()
        if not exists:
            db.execute(text("INSERT INTO rag.kb_versions VALUES (:version,CAST(:metadata AS jsonb))"),
                       {"version":version, "metadata":json.dumps(metadata,ensure_ascii=False)})
            for entry in metadata["catalog"]:
                extracted, original = archives[entry["source"]]
                if (hashlib.sha256(extracted).hexdigest() != entry["sha256"]
                        or hashlib.sha256(original).hexdigest() != entry.get("original_sha256", entry["sha256"])):
                    raise ValueError("Source archive integrity failure")
                db.execute(text("INSERT INTO rag.kb_documents VALUES (:version,:source,CAST(:metadata AS jsonb),:extracted,:original)"),
                           dict(version=version,source=entry["source"],metadata=json.dumps(entry,ensure_ascii=False),extracted=extracted,original=original))
            rows = []
            for i,(chunk,vector) in enumerate(zip(chunks,vectors)):
                blob = vector.astype("<f4").tobytes()
                rows.append(dict(version=version,ordinal=i,source=chunk["source"],payload=json.dumps(chunk,ensure_ascii=False),
                                 embedding="["+",".join(format(float(v),".17g") for v in vector)+"]",blob=blob,
                                 vector_hash=hashlib.sha256(blob).hexdigest(),text_hash=hashlib.sha256(chunk["text"].encode()).hexdigest()))
            db.execute(text("""INSERT INTO rag.kb_chunks VALUES
                (:version,:ordinal,:source,CAST(:payload AS jsonb),CAST(:embedding AS public.vector),:blob,:vector_hash,:text_hash)"""), rows)
        db.execute(text("""INSERT INTO rag.kb_state VALUES (1,:version)
            ON CONFLICT(id) DO UPDATE SET active_version=EXCLUDED.active_version"""), {"version":version})
    return version
