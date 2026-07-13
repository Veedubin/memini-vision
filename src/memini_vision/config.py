"""Configuration for memini-vision via pydantic-settings.

All env vars use the ``MEMINI_IMAGE_`` prefix. The ``db_url`` field falls
back to ``MEMINI_DB_URL`` when ``MEMINI_IMAGE_DB_URL`` is empty — this lets
videre-mcp and memini-ai share a single PostgreSQL instance without
duplicating the connection string.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Canonical dimension of the ``memories_image.embedding`` column.
#: Both ViT-B/32 (512-dim, zero-padded) and ViT-L/14 (768-dim, native)
#: write to this single column — avoids a second table.
EMBEDDING_DIM = 768

#: Allowed CLIP model IDs. The validator enforces this allow-list so a
#: typo like ``clip-ViT-B-32`` vs ``clip-vit-b-32`` is caught at config load.
ALLOWED_CLIP_MODELS = frozenset({"clip-ViT-B-32", "clip-ViT-L-14"})

#: Allowed device specifiers.
ALLOWED_DEVICES = frozenset({"auto", "cpu", "cuda"})


def _clamp_max_file_size(v: int) -> int:
    """Clamp ``MEMINI_IMAGE_MAX_FILE_SIZE`` to a safe range.

    Minimum 1 KiB (reject 0 or negative), maximum 100 MiB (reject absurdly
    large values that would blow up the filesystem). The default 10 MiB
    sits comfortably inside this range.
    """
    min_bytes = 1024
    max_bytes = 100 * 1024 * 1024
    if v < min_bytes:
        return min_bytes
    if v > max_bytes:
        return max_bytes
    return v


class VisionConfig(BaseSettings):
    """Pydantic-settings for ``MEMINI_IMAGE_*`` environment variables.

    Read at module load time by the two consuming servers (videre-mcp,
    memini-ai). The library itself does NOT gate on ``search_enabled`` —
    it just exposes the flag. The consumer checks it before importing or
    calling vision code.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMINI_IMAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Master kill-switch. Default ``False`` — the entire subsystem is
    #: opt-in. The consumer (videre-mcp or memini-ai) checks this flag.
    search_enabled: bool = False

    #: CLIP model ID. ``clip-ViT-B-32`` (512-dim, ~150MB) is the default;
    #: ``clip-ViT-L-14`` (768-dim, ~890MB) is the opt-in upgrade.
    clip_model: str = "clip-ViT-B-32"

    #: Device: ``auto`` (CUDA if available, else CPU), ``cpu``, or ``cuda``.
    clip_device: str = "auto"

    #: Filesystem root for sharded image storage. ``~`` is expanded.
    #: Uses ``validation_alias`` so the env var is ``MEMINI_IMAGE_DIR``
    #: (not ``MEMINI_IMAGE_IMAGE_DIR`` which pydantic-settings' prefix
    #: would otherwise generate for a field named ``image_dir``).
    image_dir: str = Field(default="~/.memini-ai/images", alias="MEMINI_IMAGE_DIR")

    #: Maximum image file size in bytes. Clamped to [1 KiB, 100 MiB].
    max_file_size: int = 10 * 1024 * 1024

    #: PostgreSQL connection string. Empty string → fall back to
    #: ``MEMINI_DB_URL`` (the shared memini-ai connection).
    db_url: str = ""

    @field_validator("clip_model")
    @classmethod
    def _validate_clip_model(cls, v: str) -> str:
        if v not in ALLOWED_CLIP_MODELS:
            raise ValueError(f"clip_model must be one of {sorted(ALLOWED_CLIP_MODELS)}, got {v!r}")
        return v

    @field_validator("clip_device")
    @classmethod
    def _validate_clip_device(cls, v: str) -> str:
        if v not in ALLOWED_DEVICES:
            raise ValueError(f"clip_device must be one of {sorted(ALLOWED_DEVICES)}, got {v!r}")
        return v

    @field_validator("max_file_size")
    @classmethod
    def _validate_max_file_size(cls, v: int) -> int:
        return _clamp_max_file_size(v)

    @property
    def resolved_db_url(self) -> str:
        """Return ``db_url`` if set, else fall back to ``MEMINI_DB_URL``.

        This mirrors the spec's "falls back to MEMINI_DB_URL" requirement
        (Section 2.3 / Section 7.1). The fallback is a runtime property
        so ``MEMINI_DB_URL`` changes are picked up even if the
        ``VisionConfig`` instance was created before the env var was set.
        """
        if self.db_url:
            return self.db_url
        return os.environ.get("MEMINI_DB_URL", "")

    @property
    def resolved_image_dir(self) -> Path:
        """Return ``image_dir`` as an expanded ``Path``.

        ``~`` is expanded to the user's home directory. The path is NOT
        created here — ``ImageStore`` creates it on first write.
        """
        return Path(self.image_dir).expanduser()
