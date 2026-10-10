"""检索、影响推理、证据获取和验证模块之间共享的数据契约。

这里的结构约束有意比 Prompt 更严格：LLM 可以提出假设，但只有通过
Pydantic 校验的数据才能流入下一模块，从而避免自然语言字段悄悄漂移。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    # 禁止未知字段，Prompt 和代码的数据协议不一致时立即失败，而不是静默丢字段。
    model_config = ConfigDict(extra="forbid")


class InfluenceLevel(str, Enum):
    """候选记忆对当前回答或行动的实质影响强度。"""

    HIGH = "high"
    LOW = "low"
    NONE = "none"


class EvidenceSourceType(str, Enum):
    """Gap 的获取通道，而不是命题的语义类别。"""

    PERSONAL_MEMORY_STORE = "personal_memory_store"
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    ENVIRONMENT_TOOL = "environment_tool"
    USER = "user"


class QueryKind(str, Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    MIXED = "mixed"


class DimensionScope(str, Enum):
    CORE = "core"
    SECONDARY = "secondary"


class ClaimBasis(str, Enum):
    TASK = "task"
    MEMORY = "memory"
    INFERENCE = "inference"
    GAP = "gap"


class GapStatus(str, Enum):
    OPEN = "open"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class EvidenceRelation(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    INSUFFICIENT = "insufficient"


class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    IRRELEVANT = "irrelevant"
    UNRESOLVED = "unresolved"


class CandidateMemory(StrictModel):
    """T-Mem 候选发现层交给 My-Mem 的最小记忆对象。"""

    memory_id: str
    memory_type: str = "item"
    content: str
    timestamp: str | None = None
    source_turn_ids: list[str] = Field(default_factory=list)
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseObligation(StrictModel):
    """当前输入自然产生的一项响应责任；priority 越高越应优先满足。"""

    obligation_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class TaskDimension(StrictModel):
    """所有候选共享的、仅由 Query 生成的初始影响维度。"""

    dimension_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scope: DimensionScope
    priority: float = Field(ge=0.0, le=1.0)


class TaskFrame(StrictModel):
    """候选无关的公共任务框架，不允许在候选评估期间原地修改。"""

    query_kind: QueryKind
    explicit_facts: list[str] = Field(default_factory=list)
    focal_response_goal: str = Field(min_length=1)
    response_obligations: list[ResponseObligation] = Field(min_length=1)
    common_dimensions: list[TaskDimension] = Field(min_length=1)
    baseline_response: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "TaskFrame":
        obligation_ids = [item.obligation_id for item in self.response_obligations]
        dimension_ids = [item.dimension_id for item in self.common_dimensions]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("response obligation ids must be unique")
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("common dimension ids must be unique")
        return self


class EvidenceDocument(StrictModel):
    """实验预先提供的可检索证据；provenance 用于追踪事实来源。"""

    document_id: str
    content: str
    source_type: EvidenceSourceType
    provenance: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceGap(StrictModel):
    """可被检索和判真的缺失命题，而不是笼统的“仍不确定”。"""

    gap_id: str
    proposition: str
    source_type: EvidenceSourceType
    retrieval_query: str
    decision_relevance: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GapStatus = GapStatus.OPEN


class ProposedDimension(StrictModel):
    """单个候选在本任务内提出的临时维度，不写回公共 TaskFrame。"""

    dimension_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scope: DimensionScope
    priority: float = Field(ge=0.0, le=1.0)
    memory_grounding: str = Field(min_length=1)
    query_bridge: str = Field(min_length=1)
    decision_delta: str = Field(min_length=1)


class BridgeStep(StrictModel):
    """单轮内显式展开的桥接步骤，并标明每步的事实基础。"""

    claim: str = Field(min_length=1)
    basis: ClaimBasis
    confidence: float = Field(ge=0.0, le=1.0)
    gap_id: str | None = None

    @model_validator(mode="after")
    def gap_basis_is_consistent(self) -> "BridgeStep":
        if self.basis == ClaimBasis.GAP and not self.gap_id:
            raise ValueError("a gap-based bridge step must cite gap_id")
        if self.basis != ClaimBasis.GAP and self.gap_id is not None:
            raise ValueError("only a gap-based bridge step may cite gap_id")
        return self


class InfluenceHypothesis(StrictModel):
    """一条候选记忆影响当前任务的待验证解释。"""

    candidate_id: str
    influence_level: InfluenceLevel
    effect_target: str = ""
    matched_dimension_ids: list[str] = Field(default_factory=list)
    proposed_dimension: ProposedDimension | None = None
    mechanism: list[str] = Field(default_factory=list)
    bridge_steps: list[BridgeStep] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    task_supported_claims: list[str] = Field(default_factory=list)
    memory_supported_claims: list[str] = Field(default_factory=list)
    inferred_claims: list[str] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    possible_action_change: str | None = None
    impact: float = Field(ge=0.0, le=1.0)
    bridge_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def level_and_claims_are_consistent(self) -> "InfluenceHypothesis":
        """强制三分类、impact 和 gap 语义保持一致。"""

        if self.influence_level == InfluenceLevel.NONE:
            if self.impact >= 0.20:
                raise ValueError("none influence requires impact < 0.20")
            if self.gaps:
                raise ValueError("none influence cannot request evidence gaps")
            if self.possible_action_change:
                raise ValueError("none influence cannot claim a possible action change")
            if self.proposed_dimension is not None:
                raise ValueError("none influence cannot create a novel dimension")
            if self.bridge_steps:
                raise ValueError("none influence cannot claim an influence bridge")
        elif self.influence_level == InfluenceLevel.LOW:
            if not 0.20 <= self.impact < 0.60:
                raise ValueError("low influence requires 0.20 <= impact < 0.60")
            if not self.possible_action_change:
                raise ValueError("low influence must state its minor action or response change")
            if not self.matched_dimension_ids and self.proposed_dimension is None:
                raise ValueError("low influence must target a common or proposed dimension")
            if not self.bridge_steps:
                raise ValueError("low influence must provide bridge steps")
        elif self.influence_level == InfluenceLevel.HIGH:
            if self.impact < 0.60:
                raise ValueError("high influence requires impact >= 0.60")
            if not self.possible_action_change:
                raise ValueError("high influence must state its material action or response change")
            if not self.matched_dimension_ids and self.proposed_dimension is None:
                raise ValueError("high influence must target a common or proposed dimension")
            if not self.bridge_steps:
                raise ValueError("high influence must provide bridge steps")

        if len(self.matched_dimension_ids) != len(set(self.matched_dimension_ids)):
            raise ValueError("matched dimension ids must be unique")
        if (
            self.proposed_dimension is not None
            and self.proposed_dimension.dimension_id in self.matched_dimension_ids
        ):
            raise ValueError("a proposed dimension cannot also be a matched common dimension")

        supported = self.task_supported_claims + self.memory_supported_claims
        supported_keys = {_claim_key(claim) for claim in supported if _claim_key(claim)}
        inferred_keys = {_claim_key(claim) for claim in self.inferred_claims if _claim_key(claim)}
        overlap = supported_keys & inferred_keys
        if overlap:
            raise ValueError("inferred_claims cannot repeat task- or memory-supported facts")

        gap_keys: set[str] = set()
        for gap in self.gaps:
            key = _claim_key(gap.proposition)
            if key in supported_keys:
                raise ValueError(
                    "evidence gaps cannot repeat task- or memory-supported facts"
                )
            if key and key in gap_keys:
                raise ValueError("duplicate evidence-gap propositions are not allowed")
            if key:
                gap_keys.add(key)

        known_gap_ids = {gap.gap_id for gap in self.gaps}
        for step in self.bridge_steps:
            if step.gap_id is not None and step.gap_id not in known_gap_ids:
                raise ValueError("bridge step cites an unknown gap_id")
        return self


class AcquiredEvidence(StrictModel):
    """EvidenceRetriever 针对某个 gap 实际取回的证据。"""

    evidence_id: str
    gap_id: str
    document_id: str
    content: str
    source_type: EvidenceSourceType
    provenance: str
    relation: EvidenceRelation = EvidenceRelation.INSUFFICIENT
    retrieval_score: float = 0.0


class ValidatedInfluence(StrictModel):
    """验证后的影响结论，是本轮最小实现的最终语义产物。"""

    candidate_id: str
    status: ValidationStatus
    influence_level: InfluenceLevel
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_gap_ids: list[str] = Field(default_factory=list)
    action_implication: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validation_is_consistent(self) -> "ValidatedInfluence":
        # 防止模型只凭自信度把 hypothesis 宣布为已验证。
        if self.status == ValidationStatus.VALIDATED and not self.evidence_ids:
            raise ValueError("validated influence must cite acquired evidence")
        if self.status in (ValidationStatus.REJECTED, ValidationStatus.IRRELEVANT):
            if self.influence_level != InfluenceLevel.NONE:
                raise ValueError("rejected or irrelevant results must have none influence")
        elif self.influence_level == InfluenceLevel.NONE:
            raise ValueError("validated or unresolved results cannot have none influence")
        return self


class TaskCase(StrictModel):
    """Controller 的完整输入：任务、候选记忆以及本轮允许访问的证据池。"""

    case_id: str
    task: str
    candidates: list[CandidateMemory]
    evidence_pool: list[EvidenceDocument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InfluenceRunResult(StrictModel):
    """一次运行的完整轨迹，用于评测、消融和错误分析。"""

    case_id: str
    task: str
    task_frame: TaskFrame
    hypotheses: list[InfluenceHypothesis]
    acquired_evidence: list[AcquiredEvidence]
    validated_influences: list[ValidatedInfluence]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


_QUESTION_PREFIXES = (
    "whether",
    "if",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "the",
    "是否",
    "能否",
    "有没有",
)


def _claim_key(value: str) -> str:
    """生成保守的命题比较键，只拦截近乎逐字重复的 supported fact。"""

    normalized = value.casefold().strip()
    changed = True
    while changed:
        changed = False
        for prefix in _QUESTION_PREFIXES:
            if normalized.startswith(prefix + " "):
                normalized = normalized[len(prefix) :].lstrip()
                changed = True
                break
            if prefix in ("是否", "能否", "有没有") and normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].lstrip()
                changed = True
                break
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", normalized))
