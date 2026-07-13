"""Test fixtures for memini-vision.

CLIP tests mock the sentence-transformers ``SentenceTransformer`` model
(no 150MB download in CI). DB tests mock the asyncpg pool (no live
PostgreSQL required). Storage tests use ``tmp_path`` (no real FS
pollution).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image


# Isolate env vars so tests don't pick up the developer's real .env.
# Mirrors the memini-ai test_config.py ``_isolate_env`` pattern.
@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all MEMINI_IMAGE_* and MEMINI_DB_URL env vars before each test."""
    for key in list(os.environ):
        if key.startswith("MEMINI_IMAGE_") or key == "MEMINI_DB_URL":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """A temporary directory for image storage tests."""
    d = tmp_path / "images"
    d.mkdir()
    return d


@pytest.fixture
def sample_png_bytes() -> bytes:
    """A 16x16 red PNG image as bytes (in-memory)."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_png_path(tmp_path: Path, sample_png_bytes: bytes) -> Path:
    """A 16x16 red PNG written to a temp file."""
    p = tmp_path / "sample.png"
    p.write_bytes(sample_png_bytes)
    return p


@pytest.fixture
def mock_clip_model() -> MagicMock:
    """A mock SentenceTransformer that returns a fixed-shape numpy array.

    Configured for ViT-B/32 (512-dim) by default. Tests that need L/14
    (768-dim) can override ``get_embedding_dimension``.
    """
    model = MagicMock()
    model.get_embedding_dimension.return_value = 512

    def _encode(inputs: Any, **kwargs: Any) -> np.ndarray:
        # If input is a string → text tower (1 sample); if list → batch;
        # if PIL image → image tower (1 sample).
        if isinstance(inputs, str):
            return np.zeros(512, dtype=np.float32)
        if isinstance(inputs, Image.Image):
            return np.ones(512, dtype=np.float32) * 0.5
        if isinstance(inputs, list):
            return np.zeros((len(inputs), 512), dtype=np.float32)
        return np.zeros(512, dtype=np.float32)

    model.encode.side_effect = _encode
    return model


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """A mock asyncpg pool with async context-manager acquire().

    The ``acquire()`` method returns an async context manager yielding a
    mock connection with async ``fetch``/``fetchrow``/``fetchval``/``execute``
    methods. Tests set return values on these methods as needed.
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="OK")

    pool = AsyncMock()

    # asyncpg pool.acquire() is an async context manager
    class _Ctx:
        async def __aenter__(self) -> AsyncMock:
            return conn

        async def __aexit__(self, *args: Any) -> None:
            pass

    pool.acquire = MagicMock(return_value=_Ctx())
    pool.close = AsyncMock(return_value=None)
    pool._conn = conn  # expose the mock connection for test assertions
    return pool
