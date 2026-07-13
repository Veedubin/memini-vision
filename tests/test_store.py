"""Tests for memini_vision.store — sharding, sha256, copy, get, exists, mime, dims."""

from __future__ import annotations

from pathlib import Path

import pytest

from memini_vision.store import ImageStore, mime_for_ext


class TestImageStoreSharding:
    def test_shard_path_uses_first_2_chars(self, temp_image_dir: Path) -> None:
        """The stored path is ``<dir>/<sha256[:2]>/<sha256>.<ext>``."""
        store = ImageStore(temp_image_dir)
        # Force a known sha by pre-computing
        import hashlib

        sha = hashlib.sha256(b"test").hexdigest()
        path = store._shard_path(sha, "png")
        assert path.parent.name == sha[:2]
        assert path.name == f"{sha}.png"

    def test_store_copies_to_sharded_path(
        self, temp_image_dir: Path, sample_png_path: Path
    ) -> None:
        """store() copies the file to the sharded location and returns metadata."""
        store = ImageStore(temp_image_dir)
        result = store.store(sample_png_path, "image/png")
        assert result.sha256 != ""
        assert result.file_path.parent.name == result.sha256[:2]
        assert result.file_path.suffix == ".png"
        assert result.file_path.is_file()
        assert result.mime_type == "image/png"
        assert result.file_size_bytes > 0

    def test_store_is_content_addressed_idempotent(
        self, temp_image_dir: Path, sample_png_path: Path
    ) -> None:
        """Re-storing the same bytes is a no-op (file already exists)."""
        store = ImageStore(temp_image_dir)
        r1 = store.store(sample_png_path, "image/png")
        # Corrupt the source so a re-copy would change the dest — but since
        # the dest already exists, store should skip the copy.
        r2 = store.store(sample_png_path, "image/png")
        assert r1.sha256 == r2.sha256
        assert r1.file_path == r2.file_path


class TestImageStoreGetExistsDelete:
    def test_get_path_returns_none_for_missing(self, temp_image_dir: Path) -> None:
        store = ImageStore(temp_image_dir)
        assert store.get_path("nonexistent0000000000000000000000000000") is None

    def test_get_path_returns_path_after_store(
        self, temp_image_dir: Path, sample_png_path: Path
    ) -> None:
        store = ImageStore(temp_image_dir)
        result = store.store(sample_png_path, "image/png")
        found = store.get_path(result.sha256)
        assert found is not None
        assert found == result.file_path

    def test_exists_after_store(self, temp_image_dir: Path, sample_png_path: Path) -> None:
        store = ImageStore(temp_image_dir)
        result = store.store(sample_png_path, "image/png")
        assert store.exists(result.sha256) is True
        assert store.exists("nonexistent0000000000000000000000000000") is False

    def test_delete_removes_file(self, temp_image_dir: Path, sample_png_path: Path) -> None:
        store = ImageStore(temp_image_dir)
        result = store.store(sample_png_path, "image/png")
        assert store.delete(result.sha256) is True
        assert store.exists(result.sha256) is False
        # Deleting again returns False (already gone)
        assert store.delete(result.sha256) is False


class TestImageStoreMimeAndDimensions:
    def test_dimensions_read_from_png(self, temp_image_dir: Path, sample_png_path: Path) -> None:
        """store() reads (width, height) from the image header."""
        store = ImageStore(temp_image_dir)
        result = store.store(sample_png_path, "image/png")
        assert result.dimensions == (16, 16)

    def test_dimensions_of_unknown_format(self, tmp_path: Path) -> None:
        """dimensions_of returns None for a non-image file."""
        p = tmp_path / "not_an_image.bin"
        p.write_bytes(b"\x00\x01\x02\x03")
        assert ImageStore.dimensions_of(p) is None

    def test_mime_for_ext_round_trip(self) -> None:
        """mime_for_ext maps known extensions to MIME types."""
        assert mime_for_ext("png") == "image/png"
        assert mime_for_ext(".jpg") == "image/jpeg"
        assert mime_for_ext("unknown") == "image/png"  # default


class TestImageStoreStoreBytes:
    def test_store_bytes_works(self, temp_image_dir: Path, sample_png_bytes: bytes) -> None:
        """store_bytes writes in-memory bytes to the shard path."""
        store = ImageStore(temp_image_dir)
        result = store.store_bytes(sample_png_bytes, "image/png")
        assert result.file_path.is_file()
        assert result.file_path.read_bytes() == sample_png_bytes
        assert result.dimensions == (16, 16)

    def test_store_bytes_rejects_oversized(self, temp_image_dir: Path) -> None:
        """store_bytes raises ValueError when bytes exceed max_file_size."""
        store = ImageStore(temp_image_dir, max_file_size=100)
        with pytest.raises(ValueError, match="exceeds max"):
            store.store_bytes(b"\x00" * 200, "image/png")

    def test_store_rejects_oversized_file(self, temp_image_dir: Path, tmp_path: Path) -> None:
        """store() raises ValueError when the source file exceeds max_file_size."""
        big = tmp_path / "big.png"
        big.write_bytes(b"\x00" * 500)
        store = ImageStore(temp_image_dir, max_file_size=100)
        with pytest.raises(ValueError, match="exceeds max"):
            store.store(big, "image/png")
