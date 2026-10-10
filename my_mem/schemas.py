"""检索、影响推理、证据获取和验证模块之间共享的数据契约。

这里的结构约束有意比 Prompt 更严格：LLM 可以提出假设，但只有通过
Pydantic 校验的数据才能流入下一模块，从而避免自然语言字段悄悄漂移。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    # 禁止未知字段，Prompt 和代码的数据协议不一致时立即失败，而不是静默丢字段。
    model_config = ConfigDict(extra="forbid")


class InfluenceStatus(str, Enum):
    DIRECT = "direct"
    CONDITIONAL = "conditional"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


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


class EvidenceDocument(StrictModel):
    """实验预先提供的可检索证据；provenance 用于追踪事实来源。"""

    document_id: str
    content: str
    source_type: str
    provenance: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceGap(StrictModel):
    """可被检索和判真的缺失命题，而不是笼统的“仍不确定”。"""

    gap_id: str
    proposition: str
    source_type: str
    retrieval_query: str
    decision_relevance: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GapStatus = GapStatus.OPEN


class InfluenceHypothesis(StrictModel):
    """一条候选记忆影响当前任务的待验证解释。"""

    candidate_id: str
    status: InfluenceStatus
    effect_target: str = ""
    mechanism: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    memory_supported_claims: list[str] = Field(default_factory=list)
    inferred_claims: list[str] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    possible_action_change: str | None = None
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def irrelevant_has_no_gaps(self) -> "InfluenceHypothesis":
        # 已判为无关的候选不应继续消耗证据预算。
        if self.status == InfluenceStatus.IRRELEVANT and self.gaps:
            raise ValueError("irrelevant hypotheses cannot request evidence")
        return self


class AcquiredEvidence(StrictModel):
    """EvidenceRetriever 针对某个 gap 实际取回的证据。"""

    evidence_id: str
    gap_id: str
    document_id: str
    content: str
    source_type: str
    provenance: str
    relation: EvidenceRelation = EvidenceRelation.INSUFFICIENT
    retrieval_score: float = 0.0


class ValidatedInfluence(StrictModel):
    """验证后的影响结论，是本轮最小实现的最终语义产物。"""

    candidate_id: str
    status: ValidationStatus
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_gap_ids: list[str] = Field(default_factory=list)
    action_implication: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validated_requires_evidence(self) -> "ValidatedInfluence":
        # 防止模型只凭自信度把 hypothesis 宣布为已验证。
        if self.status == ValidationStatus.VALIDATED and not self.evidence_ids:
            raise ValueError("validated influence must cite acquired evidence")
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
    hypotheses: list[InfluenceHypothesis]
    acquired_evidence: list[AcquiredEvidence]
    validated_influences: list[ValidatedInfluence]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
