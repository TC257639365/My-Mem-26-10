"""HypothesisBuilder 和 InfluenceValidator 的 LLM 实现。"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from my_mem.influence.prompts import HYPOTHESIS_PROMPT, VALIDATION_PROMPT
from my_mem.schemas import (
    AcquiredEvidence,
    CandidateMemory,
    InfluenceHypothesis,
    ValidatedInfluence,
)


class _HypothesisEnvelope(BaseModel):
    # 顶层 envelope 固定 JSON 形状，避免模型直接返回裸数组。
    hypotheses: list[InfluenceHypothesis]


class _ValidationEnvelope(BaseModel):
    result: ValidatedInfluence


_EnvelopeT = TypeVar("_EnvelopeT", bound=BaseModel)
_SCHEMA_MAX_ATTEMPTS = 3


async def _generate_typed_json(
    provider: Any,
    prompt: str,
    envelope: type[_EnvelopeT],
    *,
    call_site: str,
) -> _EnvelopeT:
    """生成并校验结构化输出，类型不匹配时把错误反馈给模型重试。"""

    schema_json = json.dumps(envelope.model_json_schema(), ensure_ascii=False)
    base_prompt = (
        f"{prompt}\n\nRequired JSON Schema (follow field types exactly):\n"
        f"{schema_json}"
    )
    attempt_prompt = base_prompt
    last_error: ValidationError | None = None

    for attempt in range(1, _SCHEMA_MAX_ATTEMPTS + 1):
        raw = await provider.generate(
            attempt_prompt,
            response_format={"type": "json_object"},
            call_site=call_site,
        )
        try:
            return envelope.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            if attempt == _SCHEMA_MAX_ATTEMPTS:
                break
            errors = json.dumps(
                exc.errors(include_url=False, include_input=False),
                ensure_ascii=False,
            )
            # 带回上一次输出，模型才能只修正错误字段；限制长度避免异常响应
            # 在重试时无限放大 prompt。
            attempt_prompt = (
                f"{base_prompt}\n\n"
                "Your previous JSON failed schema validation. Correct it and return "
                "the complete JSON object only. Do not replace numeric scores with "
                "labels such as high, medium, or low.\n"
                f"Validation errors:\n{errors}\n"
                f"Previous JSON:\n{raw[:12000]}"
            )

    assert last_error is not None
    raise last_error


class LLMHypothesisBuilder:
    def __init__(self, provider: Any):
        self.provider = provider

    async def build(
        self, task: str, candidates: list[CandidateMemory]
    ) -> list[InfluenceHypothesis]:
        # 只把 task 和候选记忆送给第一轮模型；固定证据池此时不可见。
        prompt = HYPOTHESIS_PROMPT.format(
            task=task,
            candidates_json=json.dumps(
                [candidate.model_dump(mode="json") for candidate in candidates],
                ensure_ascii=False,
            ),
        )
        result = await _generate_typed_json(
            self.provider,
            prompt,
            _HypothesisEnvelope,
            call_site="influence.hypothesis",
        )
        # 这里除字段、枚举和数值范围外，还校验候选 ID 集合。
        expected = {candidate.memory_id for candidate in candidates}
        received = {hypothesis.candidate_id for hypothesis in result.hypotheses}
        # 必须一条候选对应一条 hypothesis，防止模型漏判或虚构 memory_id。
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
        # 验证阶段重新提供原始 candidate，不能只依赖 LLM 自己生成的 hypothesis 转述。
        prompt = VALIDATION_PROMPT.format(
            task=task,
            candidate_json=candidate.model_dump_json(),
            hypothesis_json=hypothesis.model_dump_json(),
            evidence_json=json.dumps(
                [item.model_dump(mode="json") for item in evidence],
                ensure_ascii=False,
            ),
        )
        result = (
            await _generate_typed_json(
                self.provider,
                prompt,
                _ValidationEnvelope,
                call_site="influence.validation",
            )
        ).result
        if result.candidate_id != hypothesis.candidate_id:
            raise ValueError("validator returned a different candidate_id")
        # 允许引用原始记忆或本轮实际取回的证据，禁止凭空生成 evidence_id。
        allowed = {item.evidence_id for item in evidence} | {f"memory:{candidate.memory_id}"}
        if not set(result.evidence_ids).issubset(allowed):
            raise ValueError("validator cited evidence that was not supplied")
        return result
