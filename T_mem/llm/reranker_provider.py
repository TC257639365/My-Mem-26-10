"""Generic OpenAI-compatible reranker provider.

Calls ``POST {base_url}/rerank`` with an OpenAI-style body
(``{"model": ..., "query": ..., "documents": [...]}``), compatible with
endpoints that expose the standard rerank API (vLLM, Jina, SiliconFlow, ...).

Same-query batches issue a single request; mixed queries fall back to
per-pair calls (same contract as the previous internal rerankers).

Configuration (environment variables, see ``.env.example``):

- ``T_MEM_RERANKER_URL``      full rerank endpoint (recommended because
                              providers use different paths)
- ``T_MEM_RERANKER_BASE_URL`` legacy base URL; appends ``/rerank``
- ``T_MEM_RERANKER_MODEL``    reranker model name (default: ``bge-reranker-v2-m3``)
- ``T_MEM_RERANKER_API_KEY``  API key (fallback: ``T_MEM_LLM_API_KEY`` /
                              ``OPENAI_API_KEY``)
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from T_mem.llm.model_config import (
    reranker_api_key,
    reranker_base_url,
    reranker_model,
    reranker_url,
)


_logger = logging.getLogger("T_mem.reranker")


def _base_url() -> str:
    return reranker_base_url()


def _api_key() -> str:
    return reranker_api_key()


class RerankerProvider:  # pylint: disable=too-few-public-methods
    """Generic reranker via ``POST {base_url}/rerank``."""

    def __init__(
        self,
        base_url: str = "",
        endpoint_url: str = "",
        model_name: str | None = None,
        timeout: int = 120,
        max_retries: int = 10,
        **kwargs,
    ):
        self.base_url = (base_url or _base_url()).rstrip("/")
        # 统一从 cloud.env/local.env 选择模型，显式参数只用于调试覆盖。
        self.model_name = reranker_model(model_name)
        self.endpoint_url = endpoint_url.strip() or reranker_url(self.base_url)
        self.display_name = self.model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.instruction = kwargs.pop("instruction", None)

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
        return self.endpoint_url

    def rerank(
        self,
        queries: List[str],
        docs: List[str],
        instruction: str | None = None,
    ) -> List[float]:
        if not docs:
            return []
        if len(set(queries)) == 1:
            return self._rerank_one_query(queries[0], docs, instruction=instruction)
        scores = []
        for q, d in zip(queries, docs):
            one = self._rerank_one_query(q, [d], instruction=instruction)
            scores.append(one[0] if one else 0.0)
        return scores

    def __repr__(self) -> str:
        return f"RerankerProvider(model={self.display_name}, endpoint={self._endpoint()})"

    def _rerank_one_query(
        self,
        query: str,
        docs: List[str],
        instruction: str | None = None,
    ) -> List[float]:
        if not docs:
            return []

        headers = {"Content-Type": "application/json"}
        api_key = _api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict = {
            "model": self.model_name,
            "query": query,
            "documents": docs,
        }
        instr = instruction if instruction is not None else self.instruction
        if instr:
            payload["instruction"] = instr

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self._endpoint(),
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                resp_json = resp.json()

                if resp.ok:
                    # Accept both `results` (OpenAI-style) and `data` keys.
                    results = resp_json.get("results") or resp_json.get("data")
                    if isinstance(results, list):
                        sorted_back = sorted(
                            results, key=lambda x: x.get("index", 0)
                        )
                        scores = [
                            float(item.get("relevance_score", item.get("relevanceScore", 0.0)))
                            for item in sorted_back
                        ]
                        if len(scores) < len(docs):
                            scores += [0.0] * (len(docs) - len(scores))
                        return scores[: len(docs)]
                    last_err = f"unexpected body: {str(resp_json)[:300]}"
                else:
                    last_err = f"status={resp.status_code}, body={str(resp_json)[:300]}"

                is_rate_limit = resp.status_code == 429
                _logger.warning(
                    "[Reranker] attempt %d/%d: %s",
                    attempt + 1, self.max_retries, last_err,
                )
            except Exception as e:  # noqa: BLE001
                is_rate_limit = False
                last_err = f"{type(e).__name__}: {e}"
                _logger.warning(
                    "[Reranker] attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, last_err,
                )

            if attempt < self.max_retries - 1:
                if is_rate_limit:
                    wait = min(3 * (2 ** attempt) + random.uniform(0, 2), 30)
                else:
                    wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)

        _logger.error("[Reranker] all retries exhausted: %s; returning zeros", last_err)
        return [0.0] * len(docs)
