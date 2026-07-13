"""Cross-modal text→image and image→image search via the CLIP text tower.

The CLIP text tower shares the embedding space with the image tower, so
a text query can be directly compared to stored image embeddings via
cosine similarity. This is the bridge that lets an agent recall a
screenshot by describing it in natural language ("terminal showing a
Python traceback").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from memini_vision.queries import SEARCH_IMAGE_BY_VECTOR
from memini_vision.types import SearchResult

if TYPE_CHECKING:
    from memini_vision.embedder import ClipEmbedder
    from memini_vision.index import ImageIndex


class ImageQuery:
    """Cross-modal search over the ``memories_image`` table.

    Holds references to a :class:`ClipEmbedder` (for query encoding) and
    an :class:`ImageIndex` (for the asyncpg pool + SQL). The embedder's
    model is loaded lazily on first search (first call downloads the
    CLIP weights).
    """

    def __init__(self, embedder: ClipEmbedder, index: ImageIndex) -> None:
        """Initialize with an embedder and index.

        Args:
            embedder: A configured :class:`ClipEmbedder` (model loaded lazily).
            index: An :class:`ImageIndex` (pool created lazily).
        """
        self._embedder = embedder
        self._index = index

    async def search_by_text(self, text: str, limit: int = 10) -> list[SearchResult]:
        """Cross-modal text→image search.

        Encodes the query text with the CLIP text tower (same model that
        produced the stored image embeddings), then runs a cosine-distance
        nearest-neighbor query against the ``memories_image`` table.

        Args:
            text: Natural-language query (e.g. "terminal traceback").
            limit: Max results (default 10).

        Returns:
            List of :class:`SearchResult` ordered by ascending cosine
            distance (lower = more similar).
        """
        query_vec = self._embedder.encode_text(text)
        return await self._search_vector(query_vec, limit)

    async def search_by_image(
        self, image: Image.Image | str, limit: int = 10
    ) -> list[SearchResult]:
        """Image→image search (find similar stored images).

        Args:
            image: A PIL image or path to an image file.
            limit: Max results (default 10).

        Returns:
            List of :class:`SearchResult`.
        """
        query_vec = self._embedder.encode_image(image)
        return await self._search_vector(query_vec, limit)

    async def _search_vector(self, vec: list[float], limit: int) -> list[SearchResult]:
        """Run the cosine-distance search and map rows to :class:`SearchResult`."""
        pool = await self._index._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(SEARCH_IMAGE_BY_VECTOR, vec, limit)
        return [
            SearchResult(
                memory_id=str(row["memory_id"]),
                image_id=str(row["image_id"]),
                file_path=row["file_path"],
                sha256=row["sha256"],
                caption=row["caption"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]
