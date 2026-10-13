
import logging

from sqlalchemy.orm import Session

from app.repositories.knowledge_repository import (
    KnowledgeRepository,
)
from app.services.assistant.retrieval.embedding_provider import (
    embed_query,
)


logger = logging.getLogger(__name__)


class KnowledgeSearchTool:

    name = "search_maintenance_knowledge"

    description = (
        "Search FleetCare's documented maintenance guidance, "
        "Singapore vehicle inspection policy, and project "
        "risk-model rules. Use this tool when answering "
        "maintenance, inspection, safety, policy, Risk6, "
        "TR rules, or SHAP interpretation questions. "
        "For vehicle-specific risk or maintenance decisions, "
        "also call get_vehicle_analysis. "
        "Use citation IDs from retrieved sources."
    )

    schema = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Specific knowledge search question. "
                        "Chinese and English are supported."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": [
                        "maintenance",
                        "policy",
                        "project",
                    ],
                    "description": (
                        "maintenance: vehicle checks and safety; "
                        "policy: Singapore inspection regulations; "
                        "project: FleetCare Risk6, SHAP and rules."
                    ),
                },
            },
            "required": [
                "query",
                "kind",
            ],
            "additionalProperties": False,
        },
    }

    @staticmethod
    def execute(
        db: Session,
        *,
        query: str,
        kind: str,
    ) -> dict:

        if not isinstance(query, str):
            return {
                "success": False,
                "error": "Query must be a string",
            }

        query = query.strip()

        if not query or len(query) > 2000:
            return {
                "success": False,
                "error": "Invalid query length",
            }

        if kind not in {
            "maintenance",
            "policy",
            "project",
        }:
            return {
                "success": False,
                "error": "Invalid knowledge category",
            }

        # Singapore policy questions must not
        # retrieve UK-specific legal requirements.
        jurisdiction = (
            "SG" if kind == "policy" else None
        )

        try:
            # Step 1: Query embedding
            vector = embed_query(query)

            # Step 2: pgvector similarity search
            hits = KnowledgeRepository.search(
                db=db,
                vector=vector,
                kind=kind,
                jurisdiction=jurisdiction,
                limit=5,
                min_similarity=0.30,
            )

        except Exception:
            logger.exception(
                "FleetCare knowledge retrieval failed"
            )

            return {
                "success": False,
                "query": query,
                "kind": kind,
                "hits": [],
                "warning": (
                    "Knowledge retrieval is currently unavailable."
                ),
            }

        # Step 3: Normalize evidence for Qwen
        evidence = []

        for hit in hits:
            metadata = hit["metadata"]

            evidence.append({
                "citation_id": hit["citation_id"],
                "text": hit["content"],
                "title": hit["title"],
                "publisher": hit["publisher"],
                "source_url": hit["source_url"],
                "jurisdiction": hit["jurisdiction"],
                "kind": hit["kind"],
                "similarity": round(
                    hit["similarity"],
                    4,
                ),
                "verified_on": metadata.get(
                    "verified_on"
                ),
                "review_due": metadata.get(
                    "review_due"
                ),
                "applicability": metadata.get(
                    "applicability"
                ),
            })

        return {
            "success": True,
            "query": query,
            "kind": kind,
            "retrieval_backend": "pgvector",
            "embedding_dimension": 384,
            "hits": evidence,
            "warning": (
                None
                if evidence
                else (
                    "No sufficiently relevant, "
                    "reviewed knowledge evidence was found."
                )
            ),
        }
