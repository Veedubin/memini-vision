"""memini-vision — shared vision library for image storage, CLIP embedding,
and cross-modal search.

Used by both ``videre-mcp`` (the MCP server exposing vision tools to agents)
and ``memini-ai-dev`` (the semantic memory server fusing image results into
its existing RRF). No FastMCP, no CLI, no HTTP server — library only.
"""

from __future__ import annotations

__version__ = "0.1.1"
__all__ = [
    "ALLOWED_CLIP_MODELS",
    "EMBEDDING_DIM",
    "ClipEmbedder",
    "ImageIndex",
    "ImageQuery",
    "ImageRecord",
    "ImageStore",
    "SaveResult",
    "SearchResult",
    "VisionConfig",
]

from memini_vision.config import ALLOWED_CLIP_MODELS, EMBEDDING_DIM, VisionConfig
from memini_vision.embedder import ClipEmbedder
from memini_vision.index import ImageIndex
from memini_vision.query import ImageQuery
from memini_vision.store import ImageStore
from memini_vision.types import ImageRecord, SaveResult, SearchResult
