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


class ImpactLevel(str, Enum):
    """桥接成立时，候选对当前回答或行动的条件影响上界。"""

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


class BridgeStatus(str, Enum):
    """候选到任务的最小影响桥当前是否已经成立。"""

    SUPPORTED = "supported"
    OPEN = "open"
    REFUTED = "refuted"


class GapStatus(str, Enum):
    OPEN = "open"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class EvidenceRelation(str, Enum):
    UNJUDGED = "unjudged"
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
    open_dimensions: list[str] = Field(default_factory=list)
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


class CandidateDimensionChallenge(StrictModel):
    """候选在影响定级前对 F0 的映射或一次局部维度挑战。"""

    candidate_id: str
    matched_dimension_ids: list[str] = Field(default_factory=list)
    proposed_dimension: ProposedDimension | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def dimension_ids_are_unique(self) -> "CandidateDimensionChallenge":
        if len(self.matched_dimension_ids) != len(set(self.matched_dimension_ids)):
            raise ValueError("matched dimension ids must be unique")
        if (
            self.proposed_dimension is not None
            and self.proposed_dimension.dimension_id in self.matched_dimension_ids
        ):
            raise ValueError("a proposed dimension cannot also be a matched dimension")
        return self


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
    potential_impact: ImpactLevel
    effect_target: str = ""
    target_dimension_ids: list[str] = Field(default_factory=list)
    mechanism: list[str] = Field(default_factory=list)
    bridge_steps: list[BridgeStep] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    task_supported_claims: list[str] = Field(default_factory=list)
    memory_supported_claims: list[str] = Field(default_factory=list)
    inferred_claims: list[str] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    possible_action_change: str | None = None
    bridge_status: BridgeStatus
    bridge_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def impact_bridge_and_claims_are_consistent(self) -> "InfluenceHypothesis":
        """分别约束条件影响上界、桥接状态和证据缺口。"""

        if len(self.target_dimension_ids) != len(set(self.target_dimension_ids)):
            raise ValueError("target dimension ids must be unique")

        if self.potential_impact == ImpactLevel.NONE:
            if self.possible_action_change:
                raise ValueError("none potential impact cannot claim an action change")
            if self.bridge_status != BridgeStatus.REFUTED:
                raise ValueError("none potential impact requires a refuted bridge")
        elif self.potential_impact == ImpactLevel.LOW:
            if not self.possible_action_change:
                raise ValueError("low potential impact must state its minor action change")
            if not self.target_dimension_ids:
                raise ValueError("low potential impact must target a reviewed dimension")
            if not self.bridge_steps:
                raise ValueError("low potential impact must provide bridge steps")
        elif self.potential_impact == ImpactLevel.HIGH:
            if not self.possible_action_change:
                raise ValueError("high potential impact must state its material action change")
            if not self.target_dimension_ids:
                raise ValueError("high potential impact must target a reviewed dimension")
            if not self.bridge_steps:
                raise ValueError("high potential impact must provide bridge steps")

        if self.bridge_status == BridgeStatus.OPEN:
            if not self.gaps:
                raise ValueError("an open bridge must contain at least one evidence gap")
            if not any(step.basis == ClaimBasis.GAP for step in self.bridge_steps):
                raise ValueError("an open bridge must cite a gap-based bridge step")
        elif self.gaps:
            raise ValueError("only an open bridge may request evidence gaps")

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
    # Retriever 只负责取得证据，不能把“没判断”伪装成 insufficient。
    relation: EvidenceRelation = EvidenceRelation.UNJUDGED
    relation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    relation_rationale: str | None = None
    grounding_spans: list[str] = Field(default_factory=list)
    retrieval_score: float = 0.0

    @model_validator(mode="after")
    def relation_judgment_is_complete(self) -> "AcquiredEvidence":
        if self.relation == EvidenceRelation.UNJUDGED:
            if self.relation_confidence is not None or self.relation_rationale:
                raise ValueError("unjudged evidence cannot contain relation judgment")
        else:
            if self.relation_confidence is None or not self.relation_rationale:
                raise ValueError("judged evidence requires confidence and rationale")
            if self.relation in (EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE):
                if not self.grounding_spans:
                    raise ValueError("support/refute evidence requires grounding spans")
            missing = [span for span in self.grounding_spans if span not in self.content]
            if missing:
                raise ValueError(f"evidence grounding spans not found in content: {missing}")
        return self


class EvidenceRelationAssessment(StrictModel):
    """Evidence relation judge 对一条已取得证据的独立语义判断。"""

    evidence_id: str
    gap_id: str
    relation: EvidenceRelation
    rationale: str = Field(min_length=1)
    grounding_spans: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def relation_must_be_judged(self) -> "EvidenceRelationAssessment":
        if self.relation == EvidenceRelation.UNJUDGED:
            raise ValueError("relation judge cannot return unjudged")
        if self.relation in (EvidenceRelation.SUPPORT, EvidenceRelation.REFUTE):
            if not self.grounding_spans:
                raise ValueError("support/refute assessment requires grounding spans")
        return self


class ValidatedInfluence(StrictModel):
    """验证后的影响结论，是本轮最小实现的最终语义产物。"""

    candidate_id: str
    status: ValidationStatus
    potential_impact: ImpactLevel
    validated_impact: ImpactLevel | None = None
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_gap_ids: list[str] = Field(default_factory=list)
    action_implication: str | None = None
    validation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validation_is_consistent(self) -> "ValidatedInfluence":
        # 防止模型只凭自信度把 hypothesis 宣布为已验证。
        if self.status == ValidationStatus.VALIDATED:
            if not self.evidence_ids:
                raise ValueError("validated influence must cite memory or acquired evidence")
            if self.validated_impact not in (ImpactLevel.HIGH, ImpactLevel.LOW):
                raise ValueError("validated influence requires high or low validated impact")
            if self.potential_impact == ImpactLevel.NONE:
                raise ValueError("none potential impact cannot become validated influence")
            if (
                self.potential_impact == ImpactLevel.LOW
                and self.validated_impact == ImpactLevel.HIGH
            ):
                raise ValueError("validated impact cannot exceed the conditional upper bound")
        if self.status in (ValidationStatus.REJECTED, ValidationStatus.IRRELEVANT):
            if self.validated_impact != ImpactLevel.NONE:
                raise ValueError("rejected or irrelevant results require none validated impact")
        if self.status == ValidationStatus.IRRELEVANT:
            if self.potential_impact != ImpactLevel.NONE:
                raise ValueError("irrelevant result requires none potential impact")
        if self.status == ValidationStatus.REJECTED:
            if self.potential_impact == ImpactLevel.NONE:
                raise ValueError("none potential impact is irrelevant, not rejected")
        if self.status == ValidationStatus.UNRESOLVED:
            if self.potential_impact == ImpactLevel.NONE:
                raise ValueError("none potential impact cannot remain unresolved")
            if self.validated_impact is not None:
                raise ValueError("unresolved results cannot claim a validated impact")
            if not self.unresolved_gap_ids:
                raise ValueError("unresolved results must list unresolved gap ids")
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

    schema_version: str = "stage2-v2"
    case_id: str
    task: str
    task_frame: TaskFrame
    dimension_challenges: list[CandidateDimensionChallenge]
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
