"""Tests for memini_vision.config — env var parsing, defaults, validation."""

from __future__ import annotations

import pytest

from memini_vision.config import ALLOWED_CLIP_MODELS, EMBEDDING_DIM, VisionConfig


class TestVisionConfigDefaults:
    def test_defaults(self) -> None:
        """All 6 fields have the spec-mandated defaults."""
        cfg = VisionConfig()
        assert cfg.search_enabled is False
        assert cfg.clip_model == "clip-ViT-B-32"
        assert cfg.clip_device == "auto"
        assert cfg.image_dir == "~/.memini-ai/images"
        assert cfg.max_file_size == 10 * 1024 * 1024
        assert cfg.db_url == ""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MEMINI_IMAGE_* env vars override defaults."""
        monkeypatch.setenv("MEMINI_IMAGE_SEARCH_ENABLED", "true")
        monkeypatch.setenv("MEMINI_IMAGE_CLIP_MODEL", "clip-ViT-L-14")
        monkeypatch.setenv("MEMINI_IMAGE_CLIP_DEVICE", "cuda")
        monkeypatch.setenv("MEMINI_IMAGE_DIR", "/tmp/test_images")
        monkeypatch.setenv("MEMINI_IMAGE_MAX_FILE_SIZE", "2048")
        monkeypatch.setenv("MEMINI_IMAGE_DB_URL", "postgresql://u:p@h/d")
        cfg = VisionConfig()
        assert cfg.search_enabled is True
        assert cfg.clip_model == "clip-ViT-L-14"
        assert cfg.clip_device == "cuda"
        assert cfg.image_dir == "/tmp/test_images"
        assert cfg.max_file_size == 2048
        assert cfg.db_url == "postgresql://u:p@h/d"

    def test_db_url_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolved_db_url falls back to MEMINI_DB_URL when IMAGE_DB_URL empty."""
        monkeypatch.setenv("MEMINI_DB_URL", "postgresql://shared@host/db")
        cfg = VisionConfig()  # db_url stays ""
        assert cfg.db_url == ""
        assert cfg.resolved_db_url == "postgresql://shared@host/db"

    def test_clip_model_validator_rejects_unknown(self) -> None:
        """An invalid clip_model raises ValueError at config load."""
        with pytest.raises(ValueError, match="clip_model must be one of"):
            VisionConfig(clip_model="clip-ViT-B-16")  # type: ignore[call-arg]

    def test_clip_device_validator_rejects_unknown(self) -> None:
        """An invalid clip_device raises ValueError at config load."""
        with pytest.raises(ValueError, match="clip_device must be one of"):
            VisionConfig(clip_device="tpu")  # type: ignore[call-arg]

    def test_max_file_size_clamped_to_min(self) -> None:
        """max_file_size below 1 KiB is clamped to 1024."""
        cfg = VisionConfig(max_file_size=0)  # type: ignore[call-arg]
        assert cfg.max_file_size == 1024

    def test_max_file_size_clamped_to_max(self) -> None:
        """max_file_size above 100 MiB is clamped to 104857600."""
        cfg = VisionConfig(max_file_size=999 * 1024 * 1024)  # type: ignore[call-arg]
        assert cfg.max_file_size == 100 * 1024 * 1024

    def test_constants(self) -> None:
        """EMBEDDING_DIM is 768 and the two CLIP models are in the allow-list."""
        assert EMBEDDING_DIM == 768
        assert "clip-ViT-B-32" in ALLOWED_CLIP_MODELS
        assert "clip-ViT-L-14" in ALLOWED_CLIP_MODELS
