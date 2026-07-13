"""Tests for memini_vision.embedder — model selection, dim, device, lazy load, zero-pad."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from memini_vision.config import EMBEDDING_DIM
from memini_vision.embedder import ClipEmbedder


class TestClipEmbedderModelSelection:
    def test_b32_model_name_and_dim(self, mock_clip_model: MagicMock) -> None:
        """ViT-B/32 embedder reports 512 native dim."""
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        assert emb.model_name == "clip-ViT-B-32"
        assert emb.native_dim == 512
        assert emb.model_dim == 512

    def test_l14_model_name_and_dim(self) -> None:
        """ViT-L/14 embedder reports 768 native dim."""
        emb = ClipEmbedder("clip-ViT-L-14", "cpu")
        assert emb.model_name == "clip-ViT-L-14"
        assert emb.native_dim == 768

    def test_unknown_model_rejected(self) -> None:
        """An unsupported model name raises ValueError at construction."""
        with pytest.raises(ValueError, match="model_name must be one of"):
            ClipEmbedder("clip-ViT-B-16", "cpu")


class TestClipEmbedderDevice:
    def test_device_auto_resolves_to_cpu_when_no_cuda(
        self, mock_clip_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When CUDA is unavailable, auto resolves to cpu."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        emb = ClipEmbedder("clip-ViT-B-32", "auto")
        assert emb.device == "auto"  # before load
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_clip_model
        ) as mock_st:
            emb.get_model()
            mock_st.assert_called_once()
            assert mock_st.call_args.kwargs.get("device") == "cpu"
        assert emb.device == "cpu"


class TestClipEmbedderLazyLoad:
    def test_model_not_loaded_at_init(self) -> None:
        """Constructing ClipEmbedder does NOT load the model."""
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        # Access private _model to confirm it's None
        assert emb._model is None

    def test_get_model_loads_once(self, mock_clip_model: MagicMock) -> None:
        """get_model loads on first call and caches on subsequent calls."""
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_clip_model
        ) as mock_st:
            m1 = emb.get_model()
            m2 = emb.get_model()
            assert m1 is m2
            assert mock_st.call_count == 1


class TestClipEmbedderZeroPad:
    def test_b32_image_vector_zero_padded_to_768(self, mock_clip_model: MagicMock) -> None:
        """B/32 image vectors (512-dim) are zero-padded to 768."""
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_clip_model):
            vec = emb.encode_image(Image.new("RGB", (16, 16)))
        assert len(vec) == EMBEDDING_DIM
        # First 512 dims from the mock (0.5), last 256 are zeros
        assert all(v == 0.0 for v in vec[512:])

    def test_b32_text_vector_zero_padded_to_768(self, mock_clip_model: MagicMock) -> None:
        """B/32 text vectors (512-dim) are zero-padded to 768."""
        emb = ClipEmbedder("clip-ViT-B-32", "cpu")
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_clip_model):
            vec = emb.encode_text("terminal traceback")
        assert len(vec) == EMBEDDING_DIM
        assert all(v == 0.0 for v in vec[512:])

    def test_l14_vector_not_padded(self) -> None:
        """L/14 (768-dim) vectors pass through without padding."""
        model = MagicMock()
        model.get_embedding_dimension.return_value = 768
        model.encode.return_value = np.ones(768, dtype=np.float32) * 0.3
        emb = ClipEmbedder("clip-ViT-L-14", "cpu")
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            vec = emb.encode_text("test")
        assert len(vec) == EMBEDDING_DIM
        assert all(v == 0.3 for v in vec)
