"""LLM implementations of hypothesis construction and influence validation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from my_mem.influence.prompts import HYPOTHESIS_PROMPT, VALIDATION_PROMPT
from my_mem.schemas import (
    AcquiredEvidence,
    CandidateMemory,
    InfluenceHypothesis,
    ValidatedInfluence,
)


class _HypothesisEnvelope(BaseModel):
    hypotheses: list[InfluenceHypothesis]


class _ValidationEnvelope(BaseModel):
    result: ValidatedInfluence


class LLMHypothesisBuilder:
    def __init__(self, provider: Any):
        self.provider = provider

    async def build(
        self, task: str, candidates: list[CandidateMemory]
    ) -> list[InfluenceHypothesis]:
        prompt = HYPOTHESIS_PROMPT.format(
            task=task,
            candidates_json=json.dumps(
                [candidate.model_dump(mode="json") for candidate in candidates],
                ensure_ascii=False,
            ),
        )
        raw = await self.provider.generate(
            prompt,
            response_format={"type": "json_object"},
            call_site="influence.hypothesis",
        )
        result = _HypothesisEnvelope.model_validate_json(raw)
        expected = {candidate.memory_id for candidate in candidates}
        received = {hypothesis.candidate_id for hypothesis in result.hypotheses}
        if received != expected:
            raise ValueError(
                f"hypothesis candidate ids do not match input: expected={expected}, received={received}"
            )
        return result.hypotheses


class LLMInfluenceValidator:
    def __init__(self, provider: Any):
        self.provider = provider

    async def validate(
        self,
        task: str,
        candidate: CandidateMemory,
        hypothesis: InfluenceHypothesis,
        evidence: list[AcquiredEvidence],
    ) -> ValidatedInfluence:
        prompt = VALIDATION_PROMPT.format(
            task=task,
            candidate_json=candidate.model_dump_json(),
            hypothesis_json=hypothesis.model_dump_json(),
            evidence_json=json.dumps(
                [item.model_dump(mode="json") for item in evidence],
                ensure_ascii=False,
            ),
        )
        raw = await self.provider.generate(
            prompt,
            response_format={"type": "json_object"},
            call_site="influence.validation",
        )
        result = _ValidationEnvelope.model_validate_json(raw).result
        if result.candidate_id != hypothesis.candidate_id:
            raise ValueError("validator returned a different candidate_id")
        allowed = {item.evidence_id for item in evidence} | {f"memory:{candidate.memory_id}"}
        if not set(result.evidence_ids).issubset(allowed):
            raise ValueError("validator cited evidence that was not supplied")
        return result

