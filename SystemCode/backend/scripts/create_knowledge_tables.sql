-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Store document information
CREATE TABLE IF NOT EXISTS public.knowledge_document (
    document_id BIGSERIAL PRIMARY KEY,
    source_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    kind VARCHAR(30) NOT NULL,
    publisher TEXT,
    source_url TEXT,
    jurisdiction VARCHAR(30),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Store knowledge chunks and embeddings
CREATE TABLE IF NOT EXISTS public.knowledge_chunk (
    chunk_id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL
        REFERENCES public.knowledge_document(document_id)
        ON DELETE CASCADE,

    citation_id VARCHAR(64) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    kind VARCHAR(30) NOT NULL,

    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_kind
ON public.knowledge_chunk(kind);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_document
ON public.knowledge_chunk(document_id);