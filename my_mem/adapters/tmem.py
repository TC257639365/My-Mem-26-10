"""Convert the forked T-Mem Stage 6 output into My-Mem task cases."""

from __future__ import annotations

import json
from pathlib import Path

from my_mem.schemas import CandidateMemory, EvidenceDocument, TaskCase


def load_cases(
    search_results_path: str | Path,
    *,
    evidence_pool: list[EvidenceDocument] | None = None,
) -> list[TaskCase]:
    """Load structured candidates emitted by My-Mem's fork of T-Mem Stage 6.

    Upstream T-Mem records created before this fork do not contain `candidates`.
    We fail clearly instead of parsing the display-oriented context string,
    because that would lose ids, scores and provenance.
    """
    path = Path(search_results_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[TaskCase] = []
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
    """Load a standalone list of TaskCase objects used by controlled experiments."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    return [TaskCase.model_validate(case) for case in raw]

