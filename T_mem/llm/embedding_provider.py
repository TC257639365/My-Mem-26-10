"""Generic OpenAI-compatible embedding provider.

Calls ``POST {base_url}/embeddings`` with an OpenAI-style body
(``{"model": ..., "input": [...]}``), so any OpenAI-compatible endpoint
(OpenAI, vLLM, SiliconFlow, BGE-M3 deployments, ...) works as-is.

Configuration (environment variables, see ``.env.example``):

- ``T_MEM_EMBEDDING_BASE_URL`` optional endpoint; defaults to T_MEM_LLM_BASE_URL
- ``T_MEM_LLM_EMBEDDING_MODEL`` embedding model name (legacy alias:
                                ``T_MEM_EMBEDDING_MODEL``)
- ``T_MEM_EMBEDDING_API_KEY`` API key (fallback: ``T_MEM_LLM_API_KEY`` /
                              ``OPENAI_API_KEY``)
- ``T_MEM_EMBEDDING_BATCH_SIZE`` maximum texts per request (default: ``20``)
- ``T_MEM_EMBEDDING_DIMENSIONS`` optional vector dimension
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import List

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from T_mem.llm.model_config import (
    embedding_api_key,
    embedding_batch_size,
    embedding_base_url,
    embedding_dimensions,
    embedding_model,
)


_logger = logging.getLogger("T_mem.embedding")


def _base_url() -> str:
    return embedding_base_url()


def _api_key() -> str:
    return embedding_api_key()


class EmbeddingProvider:  # pylint: disable=too-few-public-methods
    """OpenAI-compatible embedding provider (POST {base_url}/embeddings)."""

    def __init__(
        self,
        base_url: str = "",
        model_name: str | None = None,
        timeout: int = 120,
        max_retries: int = 10,
        batch_size: int | None = None,
        **kwargs,
    ):
        self.base_url = (base_url or _base_url()).rstrip("/")
        # 统一从 cloud.env/local.env 选择模型，显式参数只用于调试覆盖。
        self.model_name = embedding_model(model_name)
        self.display_name = self.model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = embedding_batch_size(batch_size)
        self.dimensions = embedding_dimensions()

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _endpoint(self) -> str:
        return self.base_url + "/embeddings"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 各调用阶段原先拥有不同的 batch_size。统一在 Provider 边界再次切分，
        # 防止任何上游一次提交超过云模型限制的文本数量。
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_one_batch(texts[start : start + self.batch_size]))
        return vectors

    def _embed_one_batch(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        api_key = _api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": self.model_name, "input": texts}
        if self.dimensions is not None:
            data["dimensions"] = self.dimensions

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self._endpoint(),
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                )
                resp_json = resp.json()
                if (
                    resp.ok
                    and isinstance(resp_json, dict)
                    and isinstance(resp_json.get("data"), list)
                    and resp_json["data"]
                ):
                    sorted_data = sorted(
                        resp_json["data"], key=lambda x: x.get("index", 0)
                    )
                    return [item["embedding"] for item in sorted_data]
                is_rate_limit = resp.status_code == 429
                last_err = f"bad body: {str(resp_json)[:300]}"
                _logger.warning(
                    "[Embedding] attempt %d/%d: %s (status=%s)",
                    attempt + 1, self.max_retries, last_err, resp.status_code,
                )
            except Exception as e:  # noqa: BLE001
                is_rate_limit = False
                last_err = f"{type(e).__name__}: {e}"
                _logger.warning(
                    "[Embedding] attempt %d/%d request failed: %s",
                    attempt + 1, self.max_retries, last_err,
                )

            if attempt < self.max_retries - 1:
                if is_rate_limit:
                    wait = min(3 * (2 ** attempt) + random.uniform(0, 2), 30)
                else:
                    wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} retries: {last_err}"
        )

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        dot_product = np.dot(doc_vecs, query_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        denominator = query_norm * doc_norms
        denominator = np.where(denominator == 0, 1e-9, denominator)
        return dot_product / denominator

    def __repr__(self) -> str:
        return f"EmbeddingProvider(model={self.display_name}, endpoint={self._endpoint()})"
