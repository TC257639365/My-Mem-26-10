"""Embedding backends with an in-process cache.

The hashing backend is only for deterministic smoke tests. Research runs should use
the OpenAI-compatible backend pointed at the frozen BGE-M3 service.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import requests

from .text import multilingual_tokens


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a two-dimensional, row-normalized float32 matrix."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"embedding matrix must be 2D, got shape={array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


@dataclass
class HashingEmbedder:
    """Cheap deterministic encoder used to verify the pipeline without model services."""

    dimensions: int = 256
    name: str = "hashing-smoke-v1"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in multilingual_tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimensions
                sign = 1.0 if (value >> 8) & 1 else -1.0
                matrix[row, column] += sign
        return _normalize(matrix)


@dataclass
class OpenAICompatibleEmbedder:
    base_url: str
    model: str
    api_key: str = ""
    batch_size: int = 32
    timeout: int = 120
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.name = f"openai-compatible:{self.model}"

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleEmbedder":
        base_url = os.environ.get("RELATION_VECTOR_EMBEDDING_BASE_URL", "").strip()
        model = os.environ.get("RELATION_VECTOR_EMBEDDING_MODEL", "").strip()
        if not base_url or not model:
            raise ValueError(
                "RELATION_VECTOR_EMBEDDING_BASE_URL and "
                "RELATION_VECTOR_EMBEDDING_MODEL are required"
            )
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("RELATION_VECTOR_EMBEDDING_API_KEY", ""),
            batch_size=int(os.environ.get("RELATION_VECTOR_EMBEDDING_BATCH_SIZE", "32")),
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(batch):
                raise RuntimeError(
                    f"embedding service returned {len(data)} vectors for {len(batch)} texts"
                )
            vectors.extend(item["embedding"] for item in data)
        return _normalize(np.asarray(vectors, dtype=np.float32))


class CachedEmbedder:
    def __init__(self, backend: Embedder):
        self.backend = backend
        self.name = backend.name
        self._cache: dict[str, np.ndarray] = {}
        self.backend_calls = 0
        self.backend_texts = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            self.backend_calls += 1
            self.backend_texts += len(missing)
            vectors = self.backend.embed(missing)
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=np.float32)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self._cache[text] for text in texts], axis=0)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

