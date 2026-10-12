-- Apply once to the SAME database as public.analysis / public.prediction_shap.
-- pgvector must already be installed on the PostgreSQL server.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE TABLE IF NOT EXISTS rag.kb_versions (
 version text PRIMARY KEY, metadata jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS rag.kb_documents (
 version text REFERENCES rag.kb_versions(version), source text,
 metadata jsonb NOT NULL, extracted bytea NOT NULL, original bytea NOT NULL,
 PRIMARY KEY(version,source)
);
CREATE TABLE IF NOT EXISTS rag.kb_chunks (
 version text, ordinal integer, source text, payload jsonb NOT NULL,
 embedding public.vector(384) NOT NULL,
 vector_blob bytea NOT NULL, vector_sha256 text NOT NULL, text_sha256 text NOT NULL,
 PRIMARY KEY(version,ordinal),
 FOREIGN KEY(version,source) REFERENCES rag.kb_documents(version,source)
);
CREATE TABLE IF NOT EXISTS rag.kb_state (
 id integer PRIMARY KEY CHECK(id=1),
 active_version text NOT NULL REFERENCES rag.kb_versions(version)
);
CREATE INDEX IF NOT EXISTS kb_chunks_version_source ON rag.kb_chunks(version,source);
-- No ANN index initially: exact cosine search is appropriate for the current small knowledge base.
-- This migration does not ALTER/DROP existing project business tables.
