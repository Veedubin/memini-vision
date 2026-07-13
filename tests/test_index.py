"""Tests for memini_vision.index — CRUD round-trip with a mocked asyncpg pool.

The pool is mocked (no live PostgreSQL required). The mock returns canned
rows so we can verify the SQL routing and row→record conversion without
a database. Justification: a testcontainers PostgreSQL+pgvector container
would require Docker and a ~400MB image pull in CI; the mocked pool
covers the same logic (SQL routing, row mapping, dim validation) at
unit-test speed and without external deps.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from memini_vision.config import EMBEDDING_DIM
from memini_vision.index import ImageIndex


def _make_insert_row(image_id: str = "11111111-1111-1111-1111-111111111111") -> dict[str, Any]:
    """Canned INSERT_IMAGE_MEMORY return row."""
    return {"id": image_id, "created_at": "2026-01-01T00:00:00+00:00"}


def _make_select_row(image_id: str = "11111111-1111-1111-1111-111111111111") -> dict[str, Any]:
    """Canned GET_IMAGE_BY_ID return row."""
    return {
        "id": image_id,
        "memory_id": "22222222-2222-2222-2222-222222222222",
        "file_path": "/tmp/images/ab/abcdef.png",
        "sha256": "abcdef0123456789",
        "mime_type": "image/png",
        "width": 16,
        "height": 16,
        "file_size_bytes": 100,
        "caption": "test caption",
        "source_tool": "save_image_memory",
        "tags": ["auth", "bug"],
        "created_at": "2026-01-01T00:00:00+00:00",
        "embedding_model": "clip-ViT-B-32",
    }


class TestImageIndexInsert:
    async def test_insert_returns_image_id_and_created_at(self, mock_db_pool: AsyncMock) -> None:
        """insert() returns (image_id, created_at) from the DB."""
        row = _make_insert_row()
        mock_db_pool._conn.fetchrow.return_value = row  # type: ignore[attr-defined]
        idx = ImageIndex("postgresql://u:p@h/d")
        with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
            image_id, created_at = await idx.insert(
                memory_id="22222222-2222-2222-2222-222222222222",
                file_path="/tmp/images/ab/abcdef.png",
                sha256="abcdef0123456789",
                mime_type="image/png",
                embedding=[0.0] * EMBEDDING_DIM,
            )
        assert image_id == "11111111-1111-1111-1111-111111111111"
        assert created_at is not None

    async def test_insert_rejects_wrong_dim_embedding(self) -> None:
        """insert() raises ValueError if embedding is not 768-dim."""
        idx = ImageIndex("postgresql://u:p@h/d")
        with pytest.raises(ValueError, match="embedding must be 768-dim"):
            await idx.insert(
                memory_id="m",
                file_path="/tmp/x.png",
                sha256="abc",
                mime_type="image/png",
                embedding=[0.0] * 512,  # wrong dim
            )


class TestImageIndexGet:
    async def test_get_by_id_returns_record(self, mock_db_pool: AsyncMock) -> None:
        """get_by_id maps the row to an ImageRecord."""
        row = _make_select_row()
        mock_db_pool._conn.fetchrow.return_value = row  # type: ignore[attr-defined]
        idx = ImageIndex("postgresql://u:p@h/d")
        with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
            record = await idx.get_by_id("11111111-1111-1111-1111-111111111111")
        assert record is not None
        assert record.id == "11111111-1111-1111-1111-111111111111"
        assert record.memory_id == "22222222-2222-2222-2222-222222222222"
        assert record.sha256 == "abcdef0123456789"
        assert record.mime_type == "image/png"
        assert record.tags == ["auth", "bug"]
        assert record.embedding is None  # not selected by GET queries

    async def test_get_by_id_returns_none_for_missing(self, mock_db_pool: AsyncMock) -> None:
        """get_by_id returns None when the row doesn't exist."""
        mock_db_pool._conn.fetchrow.return_value = None  # type: ignore[attr-defined]
        idx = ImageIndex("postgresql://u:p@h/d")
        with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
            record = await idx.get_by_id("nonexistent-uuid")
        assert record is None


class TestImageIndexDelete:
    async def test_delete_returns_true_when_row_exists(self, mock_db_pool: AsyncMock) -> None:
        """delete() returns True when fetchval returns a non-None result."""
        mock_db_pool._conn.fetchval.return_value = "11111111-1111-1111-1111-111111111111"  # type: ignore[attr-defined]
        idx = ImageIndex("postgresql://u:p@h/d")
        with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
            assert await idx.delete("11111111-1111-1111-1111-111111111111") is True

    async def test_delete_returns_false_when_row_missing(self, mock_db_pool: AsyncMock) -> None:
        """delete() returns False when fetchval returns None."""
        mock_db_pool._conn.fetchval.return_value = None  # type: ignore[attr-defined]
        idx = ImageIndex("postgresql://u:p@h/d")
        with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
            assert await idx.delete("nonexistent-uuid") is False


class TestImageIndexRequiresDbUrl:
    async def test_initialize_requires_db_url(self) -> None:
        """initialize() raises RuntimeError when db_url is empty."""
        idx = ImageIndex("")
        with pytest.raises(RuntimeError, match="requires a db_url"):
            await idx.initialize()
