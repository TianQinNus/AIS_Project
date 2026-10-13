
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sqlalchemy import text

from app.db.session import engine


EXPECTED_FORMAT = "fleetcare-pgvector-v3"
EXPECTED_DIMENSION = 384
EXPECTED_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_seed(path: Path) -> dict:
    """Read and validate the compressed RAG seed."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Seed file not found: {path}"
        )

    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("format") != EXPECTED_FORMAT:
        raise ValueError("Unsupported seed format")

    chunks = data.get("chunks", [])
    vectors = data.get("vectors", [])

    if not chunks or len(chunks) != len(vectors):
        raise ValueError(
            "Chunk and vector counts do not match"
        )

    model = data.get("metadata", {}).get("model")

    if model != EXPECTED_MODEL:
        raise ValueError(
            f"Unexpected embedding model: {model}"
        )

    citation_ids = [
        chunk["citation_id"]
        for chunk in chunks
    ]

    if len(set(citation_ids)) != len(citation_ids):
        raise ValueError("Duplicate citation IDs")

    for index, vector in enumerate(vectors):
        if len(vector) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Invalid dimension at vector {index}"
            )

        values = np.asarray(vector, dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid numeric values at vector {index}"
            )

    return data


def vector_to_string(vector: list[float]) -> str:
    """Convert an embedding into pgvector text format."""

    return "[" + ",".join(
        str(float(value))
        for value in vector
    ) + "]"


def import_seed(path: Path) -> None:
    """Import documents and chunks in one transaction."""

    data = load_seed(path)

    chunks = data["chunks"]
    vectors = data["vectors"]

    document_ids = {}

    # All changes commit together or roll back on failure.
    with engine.begin() as conn:

        for chunk, vector in zip(
            chunks, vectors, strict=True
        ):

            source_key = chunk["source"]

            # 1. Create or update source document.
            if source_key not in document_ids:

                document_id = conn.execute(
                    text("""
                        INSERT INTO public.knowledge_document (
                            source_key,
                            title,
                            kind,
                            publisher,
                            source_url,
                            jurisdiction
                        )
                        VALUES (
                            :source_key,
                            :title,
                            :kind,
                            :publisher,
                            :source_url,
                            :jurisdiction
                        )
                        ON CONFLICT (source_key)
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            kind = EXCLUDED.kind,
                            publisher = EXCLUDED.publisher,
                            source_url = EXCLUDED.source_url,
                            jurisdiction = EXCLUDED.jurisdiction
                        RETURNING document_id
                    """),
                    {
                        "source_key": source_key,
                        "title": chunk["title"],
                        "kind": chunk["kind"],
                        "publisher": chunk.get("publisher"),
                        "source_url": chunk.get("source_url"),
                        "jurisdiction": chunk.get("jurisdiction"),
                    },
                ).scalar_one()

                document_ids[source_key] = document_id

            # 2. Create or update the knowledge chunk.
            conn.execute(
                text("""
                    INSERT INTO public.knowledge_chunk (
                        document_id,
                        citation_id,
                        content,
                        embedding,
                        kind,
                        source_metadata
                    )
                    VALUES (
                        :document_id,
                        :citation_id,
                        :content,
                        CAST(:embedding AS vector),
                        :kind,
                        CAST(:source_metadata AS jsonb)
                    )
                    ON CONFLICT (citation_id)
                    DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        kind = EXCLUDED.kind,
                        source_metadata = EXCLUDED.source_metadata
                """),
                {
                    "document_id": document_ids[source_key],
                    "citation_id": chunk["citation_id"],
                    "content": chunk["text"],
                    "embedding": vector_to_string(vector),
                    "kind": chunk["kind"],
                    "source_metadata": json.dumps(
                        chunk,
                        ensure_ascii=False,
                    ),
                },
            )

    print("Import completed successfully.")
    print(f"Documents imported: {len(document_ids)}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Embedding dimension: {EXPECTED_DIMENSION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import FleetCare RAG seed into pgvector"
    )

    parser.add_argument(
        "--file",
        default="data/pgvector_seed.json.gz",
        help="Path to compressed seed JSON",
    )

    args = parser.parse_args()

    import_seed(Path(args.file))
