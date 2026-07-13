"""CLIP image/text embedding via sentence-transformers.

The model is loaded **lazily** on first call to :meth:`get_model` —
importing this module does NOT download the ~150MB ViT-B/32 weights.
This matches the spec's Section 6.5 (Lazy CLIP Model Load): server
startup stays fast; the first ``save_image_memory`` or ``query_images``
call incurs a 2-5s load penalty, and subsequent calls reuse the cached
model.

Dimension handling: ViT-B/32 produces 512-dim vectors; ViT-L/14 produces
768-dim. The ``memories_image.embedding`` column is ``vector(768)`` to
accommodate both. B/32 vectors are **zero-padded** to 768 at write time
(spec Section 4.3, Decision 2). This avoids a second table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from memini_vision.config import ALLOWED_CLIP_MODELS, EMBEDDING_DIM

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


#: Model → native output dimension. Used to decide whether zero-padding
#: is needed. Kept in sync with :data:`ALLOWED_CLIP_MODELS`.
_MODEL_DIMS: dict[str, int] = {
    "clip-ViT-B-32": 512,
    "clip-ViT-L-14": 768,
}


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` to ``cuda`` or ``cpu`` based on availability.

    Mirrors the memini-ai ``ModelManager._check_cuda_available`` pattern:
    check both ``torch.cuda.is_available()`` AND ``device_count() > 0``
    so a CPU-only torch install doesn't report a phantom GPU.
    """
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


class ClipEmbedder:
    """Lazy-loaded CLIP wrapper over sentence-transformers.

    Not a singleton — callers may instantiate multiple embedders with
    different models/devices (e.g. one per MCP server process). The
    first :meth:`get_model` call downloads the model to the
    sentence-transformers cache directory.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32", device: str = "auto") -> None:
        """Initialize the embedder (does NOT load the model).

        Args:
            model_name: One of :data:`ALLOWED_CLIP_MODELS`.
            device: ``auto``, ``cpu``, or ``cuda``.

        Raises:
            ValueError: If ``model_name`` is not in the allow-list.
        """
        if model_name not in ALLOWED_CLIP_MODELS:
            raise ValueError(
                f"model_name must be one of {sorted(ALLOWED_CLIP_MODELS)}, got {model_name!r}"
            )
        self._model_name = model_name
        self._device_pref = device
        self._model: SentenceTransformer | None = None
        self._resolved_device: str | None = None

    @property
    def model_name(self) -> str:
        """The configured CLIP model ID."""
        return self._model_name

    @property
    def device(self) -> str:
        """The resolved device (``cpu``/``cuda``). ``auto`` is resolved on first load."""
        if self._resolved_device is None:
            return self._device_pref
        return self._resolved_device

    @property
    def native_dim(self) -> int:
        """The native output dimension of the configured model.

        B/32 → 512, L/14 → 768. The DB column is always 768; B/32 vectors
        are zero-padded by :meth:`encode_image` / :meth:`encode_text`.
        """
        return _MODEL_DIMS[self._model_name]

    @property
    def model_dim(self) -> int:
        """Alias for :attr:`native_dim` (spec Section 2.2 naming)."""
        return self.native_dim

    def get_model(self) -> SentenceTransformer:
        """Load (or return cached) the CLIP model.

        First call downloads the model (~150MB B/32, ~890MB L/14) and
        caches it in the sentence-transformers cache directory. Subsequent
        calls return the cached instance with sub-millisecond overhead.

        Returns:
            The loaded ``SentenceTransformer`` instance.

        Raises:
            RuntimeError: If the model download/load fails.
        """
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        device = _resolve_device(self._device_pref)
        try:
            self._model = SentenceTransformer(self._model_name, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP model '{self._model_name}' on {device}. "
                f"Check your internet connection and try again."
            ) from e
        self._resolved_device = device
        return self._model

    def encode_image(self, image: Image.Image | str) -> list[float]:
        """Encode a PIL image (or path) to a 768-dim CLIP vector.

        For ViT-B/32, the native 512-dim vector is zero-padded to 768 so
        it fits the ``vector(768)`` column. ViT-L/14's 768-dim vector
        passes through unchanged.

        Args:
            image: A PIL ``Image.Image`` or a path to an image file.

        Returns:
            A 768-element ``list[float]``.

        Raises:
            RuntimeError: If the model failed to load.
        """
        model = self.get_model()
        img: Any = image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"expected PIL.Image or str path, got {type(image)}")

        vec = model.encode(img, convert_to_numpy=True)
        return self._pad_to_768(list(vec))

    def encode_text(self, text: str) -> list[float]:
        """Encode a text string to a 768-dim CLIP vector (text tower).

        The CLIP text tower shares the same embedding space as the image
        tower, so text vectors are directly comparable to image vectors
        via cosine similarity. This is the cross-modal bridge used by
        :class:`ImageQuery.search_by_text`.

        Args:
            text: Input text.

        Returns:
            A 768-element ``list[float]`` (zero-padded for B/32).
        """
        model = self.get_model()
        vec = model.encode(text, convert_to_numpy=True)
        return self._pad_to_768(list(vec))

    def _pad_to_768(self, vec: list[float]) -> list[float]:
        """Zero-pad a native-dim vector to EMBEDDING_DIM (768).

        If the vector is already 768-dim (L/14), it passes through. If
        it's 512-dim (B/32), 256 zeros are appended. Vectors longer than
        768 (shouldn't happen) are truncated.
        """
        dim = EMBEDDING_DIM
        if len(vec) == dim:
            return vec
        if len(vec) > dim:
            return list(vec[:dim])
        return list(vec) + [0.0] * (dim - len(vec))

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._resolved_device = None
