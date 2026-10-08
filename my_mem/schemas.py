"""Typed contracts between retrieval, reasoning, evidence and validation.

The contracts are intentionally stricter than the prompts.  Model-generated
text may propose hypotheses, but only validated objects cross module boundaries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
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
    memory_id: str
    memory_type: str = "item"
    content: str
    timestamp: str | None = None
    source_turn_ids: list[str] = Field(default_factory=list)
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceDocument(StrictModel):
    document_id: str
    content: str
    source_type: str
    provenance: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceGap(StrictModel):
    gap_id: str
    proposition: str
    source_type: str
    retrieval_query: str
    decision_relevance: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GapStatus = GapStatus.OPEN


class InfluenceHypothesis(StrictModel):
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
        if self.status == InfluenceStatus.IRRELEVANT and self.gaps:
            raise ValueError("irrelevant hypotheses cannot request evidence")
        return self


class AcquiredEvidence(StrictModel):
    evidence_id: str
    gap_id: str
    document_id: str
    content: str
    source_type: str
    provenance: str
    relation: EvidenceRelation = EvidenceRelation.INSUFFICIENT
    retrieval_score: float = 0.0


class ValidatedInfluence(StrictModel):
    candidate_id: str
    status: ValidationStatus
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_gap_ids: list[str] = Field(default_factory=list)
    action_implication: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validated_requires_evidence(self) -> "ValidatedInfluence":
        if self.status == ValidationStatus.VALIDATED and not self.evidence_ids:
            raise ValueError("validated influence must cite acquired evidence")
        return self


class TaskCase(StrictModel):
    case_id: str
    task: str
    candidates: list[CandidateMemory]
    evidence_pool: list[EvidenceDocument] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InfluenceRunResult(StrictModel):
    case_id: str
    task: str
    hypotheses: list[InfluenceHypothesis]
    acquired_evidence: list[AcquiredEvidence]
    validated_influences: list[ValidatedInfluence]
    diagnostics: dict[str, Any] = Field(default_factory=dict)

