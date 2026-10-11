"""Local HTTP adapter for FleetCare RAG. Run with uvicorn rag_http_api:app."""
import logging
import os
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from fleetcare_agent import ask_fleetcare_detailed
from knowledge_rag import BASE_DIR, DEFAULT_DB, KnowledgeRetriever

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="FleetCare RAG", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in os.getenv(
        "FLEETCARE_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",") if s.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
# The existing lazy singleton caches an encoder and a database version.
# Serialize its access within a worker; request-specific evidence remains isolated.
_lock = Lock()
_retriever = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, Field(min_length=1, max_length=6000, strict=True)]

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class AskRequest(Question):
    vehicle_id: StrictInt | None = None


class SearchRequest(Question):
    kind: Literal["maintenance", "policy", "project", "dataset"] | None = None


@app.get("/health")
def health():
    """Liveness only; does not certify API keys, MySQL or knowledge readiness."""
    return {"status": "alive"}


@app.post("/api/rag/ask")
def ask(body: AskRequest):
    try:
        with _lock:
            return ask_fleetcare_detailed(body.question, vehicle_id=body.vehicle_id)
    except Exception as exc:
        LOGGER.warning("RAG answer failed (%s)", type(exc).__name__)
        raise HTTPException(502, "RAG answer unavailable; check server configuration.") from None


@app.post("/api/rag/search")
def search(body: SearchRequest):
    global _retriever
    try:
        with _lock:
            if _retriever is None:
                database = os.getenv("FLEETCARE_KNOWLEDGE_DB", os.getenv("FLEETCARE_KNOWLEDGE_INDEX", str(DEFAULT_DB)))
                _retriever = KnowledgeRetriever(
                    BASE_DIR / "knowledge", database,
                    min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.32")),
                )
            return _retriever.search(body.question, kind=body.kind)
    except Exception as exc:
        LOGGER.warning("Local knowledge search failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Local knowledge unavailable; check database and cached model.") from None
