"""语义模块协议：便于替换模型、构造基线并进行消融。"""

from __future__ import annotations

from typing import Protocol

from my_mem.schemas import (
    AcquiredEvidence,
    CandidateMemory,
    InfluenceHypothesis,
    TaskFrame,
    ValidatedInfluence,
)


class TaskFrameBuilder(Protocol):
    """Query -> 候选无关的公共 TaskFrame。"""

    async def build(self, task: str) -> TaskFrame: ...


class HypothesisBuilder(Protocol):
    """Task + TaskFrame + CandidateMemory -> InfluenceHypothesis。"""

    async def build(
        self,
        task: str,
        task_frame: TaskFrame,
        candidates: list[CandidateMemory],
    ) -> list[InfluenceHypothesis]: ...


class InfluenceValidator(Protocol):
    """原始记忆 + hypothesis + 新证据 -> ValidatedInfluence。"""

    async def validate(
        self,
        task: str,
        candidate: CandidateMemory,
        hypothesis: InfluenceHypothesis,
        evidence: list[AcquiredEvidence],
    ) -> ValidatedInfluence: ...
