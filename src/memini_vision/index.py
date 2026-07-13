"""asyncpg pool + CRUD for the ``memories_image`` table.

Mirrors the connection-pool pattern from memini-ai's
``postgres/database.py``: lazy pool creation on first use, pgvector
codec registered via an ``init`` callback, idempotent schema via
:meth:`ensure_schema` (``CREATE TABLE IF NOT EXISTS`` — safe to call
from both videre-mcp and memini-ai; the second call is a no-op).

The schema is owned by this library — unlike the spec's Section 2.4 note
that migration ownership lives in memini-ai, the spec's Section 4.6
says the table is created at memini-ai startup. To satisfy both
consumers without coupling, ``ensure_schema()`` is idempotent and safe
to call from either process. Whichever process starts first creates
the table; the other's call is a no-op.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import asyncpg

from memini_vision.config import EMBEDDING_DIM
from memini_vision.queries import (
    CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN,
    CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW,
    CREATE_MEMORIES_IMAGE_INDEXES,
    CREATE_MEMORIES_IMAGE_TABLE,
    DELETE_IMAGE_BY_ID,
    GET_IMAGE_BY_ID,
    GET_IMAGE_BY_MEMORY_ID,
    GET_IMAGE_BY_SHA256,
    INSERT_IMAGE_MEMORY,
)
from memini_vision.types import ImageRecord

if TYPE_CHECKING:
    pass


class ImageIndex:
    """asyncpg-backed CRUD for the ``memories_image`` table.

    The pool is created lazily on first DB operation (mirrors memini-ai's
    ``_get_pool`` pattern). The pgvector type codec is registered on
    every connection via an ``init`` callback so pool connections can
    bind Python lists to the ``vector`` column type.
    """

    def __init__(self, db_url: str) -> None:
        """Initialize the index with a PostgreSQL connection string.

        Args:
            db_url: ``postgresql://user:pass@host:port/db``. May be empty —
                the caller should resolve via ``VisionConfig.resolved_db_url``
                before constructing this class.
        """
        self._db_url = db_url
        self._pool: asyncpg.Pool | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create the connection pool and ensure the schema exists.

        Idempotent — safe to call multiple times. After the first call,
        subsequent calls are no-ops.

        Raises:
            RuntimeError: If pool creation or schema initialization fails.
        """
        if self._initialized:
            return
        if not self._db_url:
            raise RuntimeError(
                "ImageIndex requires a db_url. Set MEMINI_IMAGE_DB_URL or MEMINI_DB_URL."
            )

        async def _init_conn(conn: asyncpg.Connection) -> None:
            from pgvector.asyncpg import register_vector

            await register_vector(conn)

        try:
            self._pool = await asyncpg.create_pool(
                self._db_url,
                min_size=1,
                max_size=5,
                init=_init_conn,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create PostgreSQL pool: {e}") from e

        await self.ensure_schema()
        self._initialized = True

    async def ensure_schema(self) -> None:
        """Create the ``memories_image`` table + indexes if absent.

        Idempotent: ``CREATE ... IF NOT EXISTS``. Uses DiskANN index when
        pgvectorscale is available, else HNSW. Safe to call from both
        videre-mcp and memini-ai processes — the second call is a no-op.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_MEMORIES_IMAGE_TABLE)
            use_diskann = await self._has_vectorscale(conn)
            idx_sql = (
                CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN
                if use_diskann
                else CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW
            )
            await conn.execute(idx_sql)
            await conn.execute(CREATE_MEMORIES_IMAGE_INDEXES)

    @staticmethod
    async def _has_vectorscale(conn: asyncpg.Connection) -> bool:
        """Check if the pgvectorscale extension is available."""
        row = await conn.fetchrow(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'vectorscale'"
        )
        return row is not None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or lazily create the connection pool."""
        if self._pool is None:
            await self.initialize()
        if self._pool is None:
            raise RuntimeError("Failed to create database pool")
        return self._pool

    async def insert(
        self,
        *,
        memory_id: str,
        file_path: str,
        sha256: str,
        mime_type: str,
        embedding: list[float],
        width: int | None = None,
        height: int | None = None,
        file_size_bytes: int | None = None,
        caption: str | None = None,
        source_tool: str | None = None,
        tags: list[str] | None = None,
        embedding_model: str = "clip-ViT-B-32",
    ) -> tuple[str, str | None]:
        """Insert an image memory row.

        The embedding must be :data:`EMBEDDING_DIM` (768) elements long.
        ViT-B/32 vectors should be zero-padded by :class:`ClipEmbedder`
        before being passed here.

        Args:
            memory_id: UUID of the parent ``memories`` row (FK).
            file_path: Absolute filesystem path to the stored image.
            sha256: SHA-256 hex digest of the image bytes.
            mime_type: MIME type (e.g. ``image/png``).
            embedding: 768-dim CLIP vector.
            width/height: Image dimensions (optional).
            file_size_bytes: File size in bytes (optional).
            caption: Optional human-readable caption.
            source_tool: Name of the tool that created this row (optional).
            tags: Optional list of tags (stored as JSONB).
            embedding_model: CLIP model ID (default ``clip-ViT-B-32``).

        Returns:
            ``(image_id, created_at)`` — the UUID of the new row and its
            creation timestamp string (or None).
        """
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"embedding must be {EMBEDDING_DIM}-dim, got {len(embedding)}")
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                INSERT_IMAGE_MEMORY,
                memory_id,
                file_path,
                sha256,
                mime_type,
                width,
                height,
                file_size_bytes,
                embedding,
                caption,
                source_tool,
                json.dumps(tags or []),
                embedding_model,
            )
        return str(row["id"]), str(row["created_at"]) if row["created_at"] else None

    async def get_by_id(self, image_id: str) -> ImageRecord | None:
        """Get an image record by its UUID."""
        return await self._get_one(GET_IMAGE_BY_ID, image_id)

    async def get_by_memory_id(self, memory_id: str) -> ImageRecord | None:
        """Get an image record by its parent memory UUID."""
        return await self._get_one(GET_IMAGE_BY_MEMORY_ID, memory_id)

    async def get_by_sha256(self, sha256: str) -> ImageRecord | None:
        """Get an image record by its SHA-256 digest."""
        return await self._get_one(GET_IMAGE_BY_SHA256, sha256)

    async def delete(self, image_id: str) -> bool:
        """Delete an image record by UUID. Returns True if a row was deleted."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(DELETE_IMAGE_BY_ID, image_id)
        return result is not None

    async def _get_one(self, sql: str, key: str) -> ImageRecord | None:
        """Fetch a single row and convert to :class:`ImageRecord`."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, key)
        if row is None:
            return None
        return self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: asyncpg.Record) -> ImageRecord:
        """Convert an asyncpg row to an :class:`ImageRecord`."""
        tags_raw = row.get("tags")
        tags: list[str] = []
        if tags_raw is not None:
            if isinstance(tags_raw, str):
                try:
                    tags = list(json.loads(tags_raw))
                except json.JSONDecodeError:
                    tags = []
            elif isinstance(tags_raw, list):
                tags = list(tags_raw)
        return ImageRecord(
            id=str(row["id"]),
            memory_id=str(row["memory_id"]),
            file_path=row["file_path"],
            sha256=row["sha256"],
            mime_type=row["mime_type"],
            embedding=None,
            width=row.get("width"),
            height=row.get("height"),
            file_size_bytes=row.get("file_size_bytes"),
            caption=row.get("caption"),
            embedding_model=row.get("embedding_model") or "clip-ViT-B-32",
            tags=tags,
            created_at=str(row["created_at"]) if row.get("created_at") else None,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
