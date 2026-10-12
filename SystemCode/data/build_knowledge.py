"""Publish reviewed source embeddings to PostgreSQL + pgvector."""
import argparse
import json
import os
from pathlib import Path
from knowledge_rag import BASE_DIR, DEFAULT_MODEL, build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=BASE_DIR/"knowledge")
    parser.add_argument("--model",default=os.getenv("RAG_EMBEDDING_MODEL",DEFAULT_MODEL))
    parser.add_argument("--allow-model-download",action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_index(args.root,args.model,allow_model_download=args.allow_model_download),ensure_ascii=False,indent=2))
