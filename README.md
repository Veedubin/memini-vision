# memini-vision

Shared vision library used by both `videre-mcp` (the MCP server that exposes
vision tools to agents) and `memini-ai-dev` (the semantic memory server that
fuses image results into its existing RRF). Owns image storage, CLIP
embedding, the `memories_image` table CRUD, and cross-modal query logic.

**No FastMCP, no CLI, no HTTP server.** Library only — the two consuming
servers register their own tools.

## Install

```bash
pip install memini-vision
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_IMAGE_SEARCH_ENABLED` | `false` | Master kill-switch (consumer checks flag) |
| `MEMINI_IMAGE_CLIP_MODEL` | `clip-ViT-B-32` | `clip-ViT-B-32` or `clip-ViT-L-14` |
| `MEMINI_IMAGE_CLIP_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `MEMINI_IMAGE_DIR` | `~/.memini-ai/images` | Filesystem image storage root |
| `MEMINI_IMAGE_MAX_FILE_SIZE` | `10485760` | Max image file size in bytes (10MB) |
| `MEMINI_IMAGE_DB_URL` | *(empty)* | PostgreSQL connection string (falls back to `MEMINI_DB_URL`) |

## Quickstart

```python
from memini_vision import VisionConfig, ImageStore, ClipEmbedder, ImageIndex, ImageQuery

cfg = VisionConfig()                     # reads MEMINI_IMAGE_* env vars
store = ImageStore(cfg.image_dir)        # filesystem sharded storage
embedder = ClipEmbedder(cfg.clip_model, cfg.clip_device)
index = ImageIndex(cfg.db_url)           # asyncpg pool, memories_image CRUD
query = ImageQuery(embedder, index)       # cross-modal CLIP search

# Encode + store an image (lazy model load on first call)
vec = embedder.encode_image(pil_image)    # list[float], zero-padded to 768 for B/32
record = await index.insert(memory_id, sha256, vec, file_path, mime_type)
results = await query.search_by_text("terminal traceback", limit=5)
```