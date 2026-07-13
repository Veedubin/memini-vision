"""SQL query constants for the ``memories_image`` table.

Mirrors the naming convention from memini-ai's ``postgres/queries.py``
(``INSERT_*``, ``SEARCH_*``, ``SELECT_*``). All queries use asyncpg's
``$1, $2`` parameter notation.
"""

from __future__ import annotations

# =============================================================================
# Schema
# =============================================================================

#: Idempotent table creation. Mirrors the ``memories_1024`` shape from
#: memini-ai's ``schema.py:117-174``: UUID PK, memory_id FK CASCADE,
#: embedding vector(768), embedding_model, created_at, trust_score.
#: The ``source_tool`` and ``tags`` columns are extra (spec Section 4.3).
CREATE_MEMORIES_IMAGE_TABLE = """
CREATE TABLE IF NOT EXISTS memories_image (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    sha256 VARCHAR(64) UNIQUE NOT NULL,
    mime_type VARCHAR(50) NOT NULL DEFAULT 'image/png',
    width INT,
    height INT,
    file_size_bytes BIGINT,
    embedding vector(768) NOT NULL,
    caption TEXT,
    source_tool VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding_model VARCHAR(100) DEFAULT 'clip-ViT-B-32'
)
"""

CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN = """
CREATE INDEX IF NOT EXISTS idx_memories_image_embedding ON memories_image
USING diskann (embedding vector_cosine_ops)
"""

CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_memories_image_embedding ON memories_image
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
"""

CREATE_MEMORIES_IMAGE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_memories_image_memory_id ON memories_image(memory_id);
CREATE INDEX IF NOT EXISTS idx_memories_image_sha256 ON memories_image(sha256);
CREATE INDEX IF NOT EXISTS idx_memories_image_created_at ON memories_image(created_at DESC);
"""

# =============================================================================
# CRUD
# =============================================================================

INSERT_IMAGE_MEMORY = """
INSERT INTO memories_image (memory_id, file_path, sha256, mime_type, width, height,
                             file_size_bytes, embedding, caption, source_tool, tags,
                             embedding_model)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
RETURNING id, created_at
"""

GET_IMAGE_BY_ID = """
SELECT id, memory_id, file_path, sha256, mime_type, width, height,
       file_size_bytes, caption, source_tool, tags, created_at,
       embedding_model
FROM memories_image
WHERE id = $1
"""

GET_IMAGE_BY_MEMORY_ID = """
SELECT id, memory_id, file_path, sha256, mime_type, width, height,
       file_size_bytes, caption, source_tool, tags, created_at,
       embedding_model
FROM memories_image
WHERE memory_id = $1
"""

GET_IMAGE_BY_SHA256 = """
SELECT id, memory_id, file_path, sha256, mime_type, width, height,
       file_size_bytes, caption, source_tool, tags, created_at,
       embedding_model
FROM memories_image
WHERE sha256 = $1
"""

DELETE_IMAGE_BY_ID = """
DELETE FROM memories_image WHERE id = $1 RETURNING id
"""

# =============================================================================
# Cross-modal search
# =============================================================================

SEARCH_IMAGE_BY_VECTOR = """
SELECT mi.id AS image_id, mi.memory_id, mi.file_path, mi.sha256, mi.caption,
       mi.embedding <=> $1::vector AS distance
FROM memories_image mi
ORDER BY mi.embedding <=> $1::vector
LIMIT $2
"""
