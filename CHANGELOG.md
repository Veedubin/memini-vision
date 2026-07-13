# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added
- Initial release of `memini-vision`, a shared vision library.
- `VisionConfig` — pydantic-settings for `MEMINI_IMAGE_*` env vars (6 vars).
- `ImageStore` — filesystem sharded storage (`<sha256[:2]>/<sha256>.<ext>`).
- `ClipEmbedder` — lazy-loaded sentence-transformers CLIP wrapper with device
  handling and dim detection. Zero-pads ViT-B/32 (512-dim) to 768 for the
  `vector(768)` schema column; ViT-L/14 (768-dim) passes through natively.
- `ImageIndex` — asyncpg pool + CRUD for the `memories_image` table.
  Idempotent `ensure_schema()` via `CREATE TABLE IF NOT EXISTS`.
- `ImageQuery` — cross-modal text→image search using the CLIP text tower.
- `ImageRecord`, `SearchResult`, `SaveResult` dataclasses.
- 25 unit tests (config, store, embedder, index, query) with mocked CLIP
  model and mocked asyncpg pool — no 150MB download, no live DB required.