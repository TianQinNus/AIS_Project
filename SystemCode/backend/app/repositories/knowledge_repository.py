
import math
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session


EMBEDDING_DIMENSION = 384

ALLOWED_KINDS = {
    "maintenance",
    "policy",
    "project",
    "dataset",
}

ALLOWED_JURISDICTIONS = {
    "SG",
    "GB",
    "GLOBAL",
}


class KnowledgeRepository:
    """
    Read-only repository for FleetCare RAG knowledge.

    Database tables:
        public.knowledge_document
        public.knowledge_chunk
    """

    @staticmethod
    def _validate_vector(
        vector: list[float],
    ) -> None:

        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Query vector must contain "
                f"{EMBEDDING_DIMENSION} dimensions"
            )

        if not all(
            math.isfinite(float(value))
            for value in vector
        ):
            raise ValueError(
                "Query vector contains invalid values"
            )

        if not any(float(value) != 0 for value in vector):
            raise ValueError(
                "Query vector cannot be zero"
            )

    @staticmethod
    def _vector_literal(
        vector: list[float],
    ) -> str:

        return "[" + ",".join(
            str(float(value))
            for value in vector
        ) + "]"

    @classmethod
    def search(
        cls,
        db: Session,
        vector: list[float],
        *,
        kind: str | None = None,
        jurisdiction: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.30,
    ) -> list[dict]:

        cls._validate_vector(vector)

        if kind is not None and kind not in ALLOWED_KINDS:
            raise ValueError("Invalid knowledge kind")

        if (
            jurisdiction is not None
            and jurisdiction not in ALLOWED_JURISDICTIONS
        ):
            raise ValueError("Invalid jurisdiction")

        if not 1 <= limit <= 20:
            raise ValueError(
                "limit must be between 1 and 20"
            )

        if not -1 <= min_similarity <= 1:
            raise ValueError(
                "min_similarity must be between -1 and 1"
            )

        # The CASE expression safely handles missing or
        # malformed review dates without blindly casting
        # every JSON value to DATE.
        statement = text("""
            WITH eligible AS (
                SELECT
                    c.citation_id,
                    c.content,
                    c.kind,
                    c.source_metadata,
                    d.title,
                    d.publisher,
                    d.source_url,
                    d.jurisdiction,

                    CASE
                        WHEN (
                            c.source_metadata->>'review_due'
                        ) ~ '^\\d{4}-\\d{2}-\\d{2}$'
                        THEN
                            c.source_metadata->>'review_due'
                        ELSE NULL
                    END AS review_due_text

                FROM public.knowledge_chunk c

                JOIN public.knowledge_document d
                    ON d.document_id = c.document_id

                WHERE
                    (
                        CAST(:kind AS text) IS NULL
                        OR c.kind = :kind
                    )
                    AND (
                        CAST(:jurisdiction AS text) IS NULL
                        OR d.jurisdiction = :jurisdiction
                        OR d.jurisdiction = 'GLOBAL'
                    )
            ),

            reviewed AS (
                SELECT *
                FROM eligible
                WHERE review_due_text IS NOT NULL
                  AND review_due_text >= :today_text
            ),

            scored AS (
                SELECT
                    *,
                    1 - (
                        embedding_placeholder
                    ) AS similarity
                FROM reviewed
            )

            SELECT * FROM scored
        """)

        # Distance must be calculated against the chunk
        # embedding. Use a direct query rather than the
        # explanatory CTE above.
        statement = text("""
            SELECT
                c.citation_id,
                c.content,
                c.kind,
                c.source_metadata,

                d.title,
                d.publisher,
                d.source_url,
                d.jurisdiction,

                1 - (
                    c.embedding <=> CAST(:vector AS vector)
                ) AS similarity

            FROM public.knowledge_chunk c

            JOIN public.knowledge_document d
                ON d.document_id = c.document_id

            WHERE
                (
                    CAST(:kind AS text) IS NULL
                    OR c.kind = :kind
                )

                AND (
                    CAST(:jurisdiction AS text) IS NULL
                    OR d.jurisdiction = :jurisdiction
                    OR d.jurisdiction = 'GLOBAL'
                )

                AND (
                    c.source_metadata->>'review_due'
                ) ~ '^\\d{4}-\\d{2}-\\d{2}$'

                AND (
                    c.source_metadata->>'review_due'
                ) >= :today_text

                AND (
                    1 - (
                        c.embedding <=> CAST(:vector AS vector)
                    )
                ) >= :min_similarity

            ORDER BY
                c.embedding <=> CAST(:vector AS vector),
                c.citation_id

            LIMIT :limit
        """)

        params = {
            "vector": cls._vector_literal(vector),
            "kind": kind,
            "jurisdiction": jurisdiction,
            "today_text": date.today().isoformat(),
            "min_similarity": min_similarity,
            "limit": limit,
        }

        rows = db.execute(
            statement,
            params,
        ).mappings().all()

        return [
            {
                "citation_id": row["citation_id"],
                "content": row["content"],
                "kind": row["kind"],
                "title": row["title"],
                "publisher": row["publisher"],
                "source_url": row["source_url"],
                "jurisdiction": row["jurisdiction"],
                "similarity": float(row["similarity"]),
                "metadata": row["source_metadata"] or {},
            }
            for row in rows
        ]
