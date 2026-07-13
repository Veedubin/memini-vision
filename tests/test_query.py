"""Tests for memini_vision.query — cross-modal search returns ranked results."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from memini_vision.embedder import ClipEmbedder
from memini_vision.index import ImageIndex
from memini_vision.query import ImageQuery


class TestImageQuerySearchByText:
    async def test_search_by_text_returns_ranked_results(
        self, mock_clip_model: Any, mock_db_pool: AsyncMock
    ) -> None:
        """search_by_text encodes the query and returns ranked SearchResults."""
        # Configure the mock DB to return 2 rows
        mock_db_pool._conn.fetch.return_value = [  # type: ignore[attr-defined]
            {
                "image_id": "11111111-1111-1111-1111-111111111111",
                "memory_id": "22222222-2222-2222-2222-222222222222",
                "file_path": "/tmp/images/ab/abcdef.png",
                "sha256": "abcdef0123456789",
                "caption": "terminal traceback",
                "distance": 0.12,
            },
            {
                "image_id": "33333333-3333-3333-3333-333333333333",
                "memory_id": "44444444-4444-4444-4444-444444444444",
                "file_path": "/tmp/images/cd/cdef0123.png",
                "sha256": "cdef0123456789",
                "caption": None,
                "distance": 0.45,
            },
        ]
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_clip_model):
            idx = ImageIndex("postgresql://u:p@h/d")
            with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
                q = ImageQuery(emb, idx)
                results = await q.search_by_text("terminal traceback", limit=5)
        assert len(results) == 2
        assert results[0].memory_id == "22222222-2222-2222-2222-222222222222"
        assert results[0].distance == 0.12
        assert results[0].caption == "terminal traceback"
        assert results[1].distance == 0.45
        # The mock CLIP text encoder returns 512-dim zeros → padded to 768
        mock_db_pool._conn.fetch.assert_called_once()  # type: ignore[attr-defined]
        call_args = mock_db_pool._conn.fetch.call_args  # type: ignore[attr-defined]
        vec_arg = call_args.args[1]
        assert len(vec_arg) == 768  # EMBEDDING_DIM

    async def test_search_by_text_empty_results(
        self, mock_clip_model: Any, mock_db_pool: AsyncMock
    ) -> None:
        """search_by_text returns [] when the DB has no matches."""
        mock_db_pool._conn.fetch.return_value = []  # type: ignore[attr-defined]
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_clip_model):
            idx = ImageIndex("postgresql://u:p@h/d")
            with patch.object(idx, "_get_pool", AsyncMock(return_value=mock_db_pool)):
                q = ImageQuery(emb, idx)
                results = await q.search_by_text("nothing matches this", limit=5)
        assert results == []
