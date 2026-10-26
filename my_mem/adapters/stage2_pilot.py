"""把 stage2 12-task pilot 数据转换为 My-Mem 运行输入。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from my_mem.schemas import CandidateMemory, EvidenceDocument, TaskCase


EVIDENCE_CONDITIONS = ("E0", "Eplus", "Eminus", "Enoise")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(dataset_dir: str | Path) -> dict[str, str]:
    root = Path(dataset_dir)
    return {
        name: _sha256(root / name)
        for name in ("tasks.json", "gold.json", "evidence_packs.json")
    }


def load_stage2_test1_cases(
    dataset_dir: str | Path,
    *,
    condition: str,
) -> list[TaskCase]:
    """加载一个 evidence condition；不会把 condition Gold 暴露给模型。"""

    if condition not in EVIDENCE_CONDITIONS:
        raise ValueError(f"unknown evidence condition: {condition}")
    root = Path(dataset_dir)
    tasks_doc = _read_json(root / "tasks.json")
    evidence_doc = _read_json(root / "evidence_packs.json")
    evidence_by_task = {item["task_id"]: item for item in evidence_doc["tasks"]}

    cases: list[TaskCase] = []
    for task in tasks_doc["tasks"]:
        task_id = task["task_id"]
        candidates = []
        for raw in task["candidates"]:
            provenance = raw["provenance"]
            candidates.append(
                CandidateMemory(
                    memory_id=raw["candidate_id"],
                    memory_type="candidate_memory",
                    content=raw["text"],
                    timestamp=provenance.get("timestamp"),
                    source_turn_ids=provenance["source_turn_ids"],
                    metadata={"provenance": provenance},
                )
            )

        condition_pack = evidence_by_task[task_id]["conditions"][condition]
        pool = []
        for index, raw in enumerate(condition_pack["evidence"], start=1):
            # 原始 evidence_id、condition 名和 relation 都是 Gold 信号。
            # 对模型只提供中性的 document_id、正文、通道和清洗后的来源。
            document_id = f"{task_id}:document:{index:02d}"
            pool.append(
                EvidenceDocument(
                    document_id=document_id,
                    content=raw["content"],
                    source_type=raw["source_type"],
                    provenance=f"controlled:{task_id}:source:{index:02d}",
                    metadata={
                        "target_candidate_id": raw["candidate_id"],
                        "gold_gap_id": raw["gap_id"],
                    },
                )
            )

        cases.append(
            TaskCase(
                case_id=task_id,
                task=task["query"],
                candidates=candidates,
                evidence_pool=pool,
                metadata={
                    "experiment_id": "stage2_test1",
                    "dataset_version": tasks_doc["dataset_version"],
                    "source_record_id": task["source_record_id"],
                    "relation_type": task["relation_type"],
                    "query_kind_group": task["query_kind"],
                    "evidence_condition": condition,
                    "channel_status": condition_pack["channel_status"],
                },
            )
        )
    return cases


def load_stage2_gold(dataset_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(dataset_dir) / "gold.json")
