"""Filesystem sharded storage for image bytes.

Images are stored at ``<image_dir>/<sha256[:2]>/<sha256>.<ext>`` — the
first 2 hex chars of the SHA-256 form a shard directory to keep any
single directory from growing unboundedly (e.g. ``ab/abcdef...png``).
The store is content-addressed: the same bytes always map to the same
path, so re-storing is a no-op.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from memini_vision.types import SaveResult

#: MIME type → file extension map. Used to pick the shard filename suffix.
_MIME_TO_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}

#: Reverse map for reconstructing mime_type from a stored path if needed.
_EXT_TO_MIME: dict[str, str] = {v: k for k, v in _MIME_TO_EXT.items()}


def _ext_for_mime(mime_type: str) -> str:
    """Return the file extension for a MIME type, defaulting to ``png``."""
    return _MIME_TO_EXT.get(mime_type, "png")


def mime_for_ext(ext: str) -> str:
    """Return the canonical MIME type for a file extension (or ``image/png``)."""
    return _EXT_TO_MIME.get(ext.lower().lstrip("."), "image/png")


class ImageStore:
    """Content-addressed, sharded filesystem storage for images.

    The store does NOT load the image into memory for storage — it copies
    bytes from the source path to the shard path. Dimensions are read via
    Pillow's lazy header parser (``Image.open`` without ``load()``).
    """

    def __init__(self, image_dir: str | Path, *, max_file_size: int = 10 * 1024 * 1024) -> None:
        """Initialize the store with a root directory and size cap.

        Args:
            image_dir: Root directory. ``~`` is expanded. Created on first write.
            max_file_size: Max image file size in bytes. Files larger than
                this raise ``ValueError``. Default 10 MiB.
        """
        self._root = Path(image_dir).expanduser()
        self._max_file_size = max_file_size

    @property
    def root(self) -> Path:
        """The expanded root directory (may not exist yet)."""
        return self._root

    @property
    def max_file_size(self) -> int:
        """The configured max file size in bytes."""
        return self._max_file_size

    def _shard_path(self, sha256: str, ext: str) -> Path:
        """Return ``<root>/<sha256[:2]>/<sha256>.<ext>`` for a digest."""
        return self._root / sha256[:2] / f"{sha256}.{ext}"

    def store(self, source_path: str | Path, mime_type: str) -> SaveResult:
        """Copy an image file into the sharded store.

        Content-addressed: if the same bytes are already stored, the
        existing file is left untouched (the copy is skipped). The SHA-256
        is computed by streaming the file in 1 MiB chunks so large files
        don't blow up memory.

        Args:
            source_path: Path to the source image file.
            mime_type: MIME type (e.g. ``image/png``). Determines the
                file extension in the shard path.

        Returns:
            :class:`SaveResult` with sha256, absolute file_path, mime_type,
            file_size_bytes, and dimensions ``(width, height)`` (or None
            if Pillow could not parse the header).

        Raises:
            FileNotFoundError: If ``source_path`` does not exist.
            ValueError: If the file exceeds ``max_file_size``.
        """
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(f"source image not found: {src}")

        size = src.stat().st_size
        if size > self._max_file_size:
            raise ValueError(
                f"image file size {size} bytes exceeds max {self._max_file_size} bytes"
            )

        sha = self._sha256_file(src)
        ext = _ext_for_mime(mime_type)
        dest = self._shard_path(sha, ext)

        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        dims = self._read_dimensions(dest)

        return SaveResult(
            sha256=sha,
            file_path=dest.resolve(),
            mime_type=mime_type,
            file_size_bytes=size,
            dimensions=dims,
        )

    def store_bytes(self, image_bytes: bytes, mime_type: str) -> SaveResult:
        """Store raw image bytes (no source file on disk).

        Writes the bytes to a temp path, then delegates to :meth:`store`.
        This is the entry point for in-memory images (e.g. screenshots
        captured directly to bytes).

        Args:
            image_bytes: Raw image bytes.
            mime_type: MIME type for extension selection.

        Returns:
            :class:`SaveResult`.
        """
        if len(image_bytes) > self._max_file_size:
            raise ValueError(
                f"image bytes size {len(image_bytes)} exceeds max {self._max_file_size} bytes"
            )
        import tempfile

        ext = _ext_for_mime(mime_type)
        # NamedTemporaryFile for atomic write + cleanup
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            return self.store(tmp_path, mime_type)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def get_path(self, sha256: str) -> Path | None:
        """Return the stored path for a SHA-256, or ``None`` if absent.

        Probes the known extensions for the digest. Returns the first
        match.
        """
        shard_dir = self._root / sha256[:2]
        if not shard_dir.is_dir():
            return None
        stem = f"{sha256}."
        for entry in shard_dir.iterdir():
            if entry.is_file() and entry.name.startswith(stem):
                return entry.resolve()
        return None

    def exists(self, sha256: str) -> bool:
        """Return True if an image with this SHA-256 is stored."""
        return self.get_path(sha256) is not None

    def delete(self, sha256: str) -> bool:
        """Delete the stored image for a SHA-256.

        Returns True if a file was deleted, False if it was not present.
        Does NOT delete the shard directory (other digests may share the
        first-2-char prefix).
        """
        path = self.get_path(sha256)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        return True

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Stream a file through SHA-256 in 1 MiB chunks."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _read_dimensions(path: Path) -> tuple[int, int] | None:
        """Read image dimensions via Pillow's lazy header parse.

        Returns ``(width, height)`` or ``None`` if the format is
        unrecognized. Uses ``Image.open`` without ``load()`` so only the
        header is read — large images don't consume memory.
        """
        try:
            with Image.open(path) as img:
                return img.size  # (width, height)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    @staticmethod
    def dimensions_of(path: str | Path) -> tuple[int, int] | None:
        """Read dimensions of an arbitrary image file (static helper)."""
        return ImageStore._read_dimensions(Path(path))

    @staticmethod
    def mime_type_of(path: str | Path) -> str:
        """Guess the MIME type from a file extension (static helper)."""
        return mime_for_ext(Path(path).suffix)
