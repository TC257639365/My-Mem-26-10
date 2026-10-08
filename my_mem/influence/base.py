"""Interfaces that make semantic components replaceable and easy to ablate."""

from __future__ import annotations

from typing import Protocol

from my_mem.schemas import (
    AcquiredEvidence,
    CandidateMemory,
    InfluenceHypothesis,
    ValidatedInfluence,
)


class HypothesisBuilder(Protocol):
    async def build(
        self, task: str, candidates: list[CandidateMemory]
    ) -> list[InfluenceHypothesis]: ...


class InfluenceValidator(Protocol):
    async def validate(
        self,
        task: str,
        candidate: CandidateMemory,
        hypothesis: InfluenceHypothesis,
        evidence: list[AcquiredEvidence],
    ) -> ValidatedInfluence: ...

