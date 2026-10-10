"""模型服务的统一环境变量配置。

My-Mem 只依赖环境变量，不在代码中保存 API Key 或具体云模型名称。
运行前 source local.env / cloud.env，即可在本地服务和云服务之间切换。

新的统一变量：

* T_MEM_LLM_BASE_URL / T_MEM_LLM_API_KEY / T_MEM_LLM_BASE_MODEL
* T_MEM_LLM_EMBEDDING_MODEL
* T_MEM_RERANKER_MODEL

Embedding 默认复用 LLM 的地址和 Key；Reranker 默认复用 LLM 的 Key。
若 Reranker 的路径不是通用的 ``/rerank``，应通过
``T_MEM_RERANKER_URL`` 提供完整请求地址。
"""

from __future__ import annotations

import os
from typing import Optional


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_RERANKER_MODEL = "bge-reranker-v2-m3"


def _first_env(*names: str) -> str:
    """返回第一个非空环境变量，便于平滑兼容旧配置名。"""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def llm_base_url() -> str:
    return (_first_env("T_MEM_LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")


def llm_api_key() -> str:
    return _first_env("T_MEM_LLM_API_KEY", "OPENAI_API_KEY")


def llm_model(explicit: Optional[str] = None) -> str:
    """基础生成模型；显式参数只用于临时调试覆盖。"""
    return (
        (explicit or "").strip()
        or _first_env("T_MEM_LLM_BASE_MODEL", "T_MEM_MEMORY_BUILD_MODEL")
        or DEFAULT_LLM_MODEL
    )


def embedding_base_url() -> str:
    # 云端通常在同一 OpenAI-compatible 根地址同时暴露 chat 和 embedding。
    return (_first_env("T_MEM_EMBEDDING_BASE_URL") or llm_base_url()).rstrip("/")


def embedding_api_key() -> str:
    return _first_env(
        "T_MEM_EMBEDDING_API_KEY", "T_MEM_LLM_API_KEY", "OPENAI_API_KEY"
    )


def embedding_model(explicit: Optional[str] = None) -> str:
    return (
        (explicit or "").strip()
        or _first_env("T_MEM_LLM_EMBEDDING_MODEL", "T_MEM_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )


def embedding_batch_size(explicit: Optional[int] = None) -> int:
    """单次 embedding 请求的文本条数；默认 20 兼容千问云模型限制。"""
    raw = str(explicit) if explicit is not None else _first_env(
        "T_MEM_EMBEDDING_BATCH_SIZE"
    )
    try:
        value = int(raw or "20")
    except ValueError as exc:
        raise ValueError("T_MEM_EMBEDDING_BATCH_SIZE 必须是正整数") from exc
    if value <= 0:
        raise ValueError("T_MEM_EMBEDDING_BATCH_SIZE 必须是正整数")
    return value


def embedding_dimensions() -> Optional[int]:
    """可选输出维度；不配置时让服务端使用模型默认值。"""
    raw = _first_env("T_MEM_EMBEDDING_DIMENSIONS")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("T_MEM_EMBEDDING_DIMENSIONS 必须是正整数") from exc
    if value <= 0:
        raise ValueError("T_MEM_EMBEDDING_DIMENSIONS 必须是正整数")
    return value


def reranker_base_url() -> str:
    return (_first_env("T_MEM_RERANKER_BASE_URL") or llm_base_url()).rstrip("/")


def reranker_url(base_url: Optional[str] = None) -> str:
    """Reranker 完整地址。

    服务商对 rerank 路径尚未统一（例如 /rerank 与 /reranks 并存），
    因此优先接受完整 URL；未配置时才保留旧项目的 /rerank 行为。
    """
    fallback_base = (base_url or reranker_base_url()).strip().rstrip("/")
    return _first_env("T_MEM_RERANKER_URL") or fallback_base + "/rerank"


def reranker_api_key() -> str:
    return _first_env(
        "T_MEM_RERANKER_API_KEY", "T_MEM_LLM_API_KEY", "OPENAI_API_KEY"
    )


def reranker_model(explicit: Optional[str] = None) -> str:
    return (
        (explicit or "").strip()
        or _first_env("T_MEM_RERANKER_MODEL")
        or DEFAULT_RERANKER_MODEL
    )
