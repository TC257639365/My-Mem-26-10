"""把改造后的 T-Mem Stage 6 输出转换为 My-Mem 的 TaskCase。"""

from __future__ import annotations

import json
from pathlib import Path

from my_mem.schemas import CandidateMemory, EvidenceDocument, TaskCase


def load_cases(
    search_results_path: str | Path,
    *,
    evidence_pool: list[EvidenceDocument] | None = None,
) -> list[TaskCase]:
    """读取 T-Mem `search_results.json` 中的结构化 candidates。

    原始 T-Mem 只保存面向 QA 的 context 字符串，旧结果没有 `candidates`。
    此处选择明确报错，而不是从 context 反向解析，因为反向解析会丢失稳定 ID、
    检索通道、分数和 provenance。
    """
    path = Path(search_results_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[TaskCase] = []
    # 同一受控证据池先挂到每个 case；后续可替换成逐 case 的外部工具环境。
    pool = evidence_pool or []
    for user_key, records in payload.items():
        for index, record in enumerate(records or []):
            raw_candidates = record.get("candidates")
            if raw_candidates is None:
                raise ValueError(
                    f"{path} record {user_key}/{index} has no structured candidates; "
                    "rerun Stage 6 from the My-Mem fork"
                )
            cases.append(
                TaskCase(
                    case_id=f"{user_key}:{index}",
                    task=record.get("query", ""),
                    candidates=[CandidateMemory.model_validate(c) for c in raw_candidates],
                    evidence_pool=pool,
                    metadata={"source": str(path), "user_key": user_key},
                )
            )
    return cases


def load_case_file(path: str | Path) -> list[TaskCase]:
    """读取已经整理成 TaskCase 的受控实验输入。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    return [TaskCase.model_validate(case) for case in raw]


def load_evidence_pool(path: str | Path) -> list[EvidenceDocument]:
    """读取独立证据池，供 T-Mem 检索结果补充 EvidenceDocument。

    支持直接的 JSON 数组，也支持 `{\"evidence_pool\": [...]}` 包装格式。
    证据池与候选记忆分文件保存，便于固定候选后单独替换证据条件。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("evidence_pool", [])
    return [EvidenceDocument.model_validate(document) for document in raw]
