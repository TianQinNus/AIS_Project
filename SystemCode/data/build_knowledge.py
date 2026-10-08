"""Offline ingestion. Run again after adding, editing or removing reviewed documents."""
import argparse
import json
import os
from pathlib import Path

from knowledge_rag import BASE_DIR, DEFAULT_MODEL, build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BASE_DIR / "knowledge")
    parser.add_argument("--index", type=Path, default=BASE_DIR / "knowledge_index" / "index.npz")
    parser.add_argument("--model", default=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()
    print(json.dumps(build_index(args.root, args.index, args.model), ensure_ascii=False, indent=2))
