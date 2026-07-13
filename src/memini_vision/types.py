"""Shared data types for memini-vision.

These dataclasses are the contract between the library and its two
consumers (videre-mcp, memini-ai). They are plain dataclasses (not
pydantic models) to keep the dependency surface minimal — only
``config.py`` needs pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """A stored image memory row (mirrors the ``memories_image`` table).

    The ``embedding`` field is ``None`` when the record is read back from
    the DB without selecting the vector column (the common case — vectors
    are large and rarely needed by the caller). When inserting, the
    caller provides the embedding via :meth:`ImageIndex.insert`.
    """

    id: str
    memory_id: str
    file_path: str
    sha256: str
    mime_type: str
    embedding: list[float] | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    caption: str | None = None
    embedding_model: str = "clip-ViT-B-32"
    tags: list[str] = field(default_factory=list)
    created_at: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """A cross-modal search hit.

    ``memory_id`` is the UUID of the ``memories`` row this image is
    attached to (1:1 FK). ``distance`` is the cosine distance from the
    CLIP text tower — lower is more similar.
    """

    memory_id: str
    image_id: str
    file_path: str
    sha256: str
    caption: str | None
    distance: float


@dataclass(frozen=True)
class SaveResult:
    """Response from :meth:`ImageStore.store`.

    The caller uses ``sha256`` as the content-addressed key and
    ``file_path`` as the absolute path where the bytes now live.
    ``dimensions`` is ``(width, height)`` or ``None`` if the image could
    not be parsed by Pillow (e.g. unknown format).
    """

    sha256: str
    file_path: Path
    mime_type: str
    file_size_bytes: int
    dimensions: tuple[int, int] | None
