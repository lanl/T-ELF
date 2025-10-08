# Termite/embedding_store/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Optional, Tuple

class EmbeddingStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def ensure_index(
        self,
        index: str,
        dim: int,
        metric: str = "cosine",
        **kwargs,
    ) -> None:
        """Create the index/collection if missing."""

    @abstractmethod
    def upsert(
        self,
        index: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Mapping]] = None,
    ) -> None:
        """Insert or update vectors with optional metadata."""

    @abstractmethod
    def search(
        self,
        index: str,
        query: List[float],
        k: int = 5,
        **kwargs,
    ) -> List[Tuple[str, float, Mapping]]:
        """Return [(id, score, payload), ...]."""
