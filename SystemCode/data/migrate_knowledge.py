"""One-time SQLite seed -> PostgreSQL migration. Runtime never reads SQLite."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import gzip
import base64
import numpy as np
from database import create_postgres_engine
from pg_knowledge_store import publish


def migrate_sqlite(path, *, engine=None):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
        db.execute("BEGIN")
        version = db.execute("SELECT active_version FROM kb_state WHERE id=1").fetchone()[0]
        metadata = json.loads(db.execute("SELECT metadata FROM kb_versions WHERE version=?",(version,)).fetchone()[0])
        archives, documents = {}, {}
        for source,meta,extracted,original in db.execute("SELECT source,metadata,extracted,original FROM kb_documents WHERE version=?",(version,)):
            documents[source] = json.loads(meta)
            archives[source] = (extracted,original)
        chunks,vectors = [],[]
        for ordinal,source,payload,blob,vhash,thash in db.execute("SELECT ordinal,source,payload,vector,vector_sha256,text_sha256 FROM kb_chunks WHERE version=? ORDER BY ordinal",(version,)):
            chunk = json.loads(payload)
            if ordinal != len(chunks) or source not in documents or any(chunk.get(k)!=v for k,v in documents[source].items()):
                raise ValueError("Legacy chunk provenance mismatch")
            if hashlib.sha256(blob).hexdigest()!=vhash or hashlib.sha256(chunk["text"].encode()).hexdigest()!=thash:
                raise ValueError("Legacy chunk integrity failure")
            chunks.append(chunk)
            vectors.append(np.frombuffer(blob,dtype="<f4"))
    catalog=metadata["catalog"]
    if documents != {e["source"]:e for e in catalog}:
        raise ValueError("Legacy catalog mismatch")
    digest=hashlib.sha256(json.dumps(catalog,sort_keys=True,ensure_ascii=False).encode())
    for entry in catalog:
        digest.update(entry["source"].encode())
        digest.update(archives[entry["source"]][0])
    if digest.hexdigest()!=metadata["fingerprint"] or len(chunks)!=metadata["chunk_count"]:
        raise ValueError("Legacy snapshot integrity failure")
    from knowledge_rag import BASE_DIR, load_catalog, fingerprint
    if fingerprint(BASE_DIR/"knowledge",load_catalog(BASE_DIR/"knowledge")) != metadata["fingerprint"]:
        raise ValueError("Legacy SQLite seed does not match current reviewed sources; use build_knowledge.py or the v3 seed")
    target=publish(engine if engine is not None else create_postgres_engine(),metadata,chunks,vectors,archives)
    return {"backend":"postgresql_pgvector","documents":len(catalog),"chunks":len(chunks),"dimension":384,"version":target}

def migrate_seed(path, *, engine=None):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        seed = json.load(stream)
    if seed.get("format") != "fleetcare-pgvector-v3":
        raise ValueError("Unsupported knowledge seed format")
    archives = {source:(base64.b64decode(data["extracted"],validate=True),base64.b64decode(data["original"],validate=True))
                for source,data in seed["archives"].items()}
    metadata,chunks = seed["metadata"],seed["chunks"]
    from knowledge_rag import BASE_DIR,load_catalog,fingerprint
    if fingerprint(BASE_DIR/"knowledge",load_catalog(BASE_DIR/"knowledge")) != metadata["fingerprint"]:
        raise ValueError("Seed does not match current reviewed sources; run build_knowledge.py")
    version = publish(engine if engine is not None else create_postgres_engine(),metadata,chunks,seed["vectors"],archives)
    return {"backend":"postgresql_pgvector","documents":len(metadata["catalog"]),"chunks":len(chunks),"dimension":384,"version":version}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--sqlite",type=Path,help="Legacy SQLite migration only; rebuild if source catalog changed")
    group.add_argument("--seed",type=Path,default=Path(__file__).parent/"knowledge_seed/pgvector_seed.json.gz")
    args=parser.parse_args()
    result=migrate_sqlite(args.sqlite) if args.sqlite else migrate_seed(args.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2))
