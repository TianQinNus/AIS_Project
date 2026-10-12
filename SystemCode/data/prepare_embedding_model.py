"""Prepare the cached embedding model without touching any database."""
import argparse
import json
import os
from knowledge_rag import DEFAULT_MODEL,load_encoder,normalized_encode

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default=DEFAULT_MODEL)
    parser.add_argument("--allow-download",action="store_true",help="Allow network access only during preparation")
    args=parser.parse_args()
    encoder=load_encoder(args.model,local_only=not args.allow_download)
    vectors=normalized_encode(encoder,["FleetCare maintenance inspection"])
    if vectors.shape[1]!=384:
        raise ValueError("Current pgvector seed requires a 384-dimensional embedding model")
    print(json.dumps({"model":args.model,"dimension":384,"cache_ready":True}))
