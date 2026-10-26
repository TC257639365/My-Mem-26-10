"""Small, dependency-light contracts shared by all experiment profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Side = Literal["memory", "query"]


@dataclass(frozen=True)
class TextView:
    owner_id: str
    side: Side
    kind: str
    text: str
    pattern_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationBinding:
    owner_id: str
    side: Side
    pattern_id: str
    role_text: str
    source_assertions: tuple[str, ...] = ()
    target_requirements: tuple[str, ...] = ()
    hard_constraints: tuple[str, ...] = ()
    open_conditions: tuple[str, ...] = ()
    join_constraints: tuple[str, ...] = ()
    confidence: float | None = None
    legacy_mixed_conditions: bool = False
    entity_scope: dict[str, Any] = field(default_factory=dict)
    temporal_scope: dict[str, Any] = field(default_factory=dict)
    audit_status: str | None = None


@dataclass(frozen=True)
class PatternDefinition:
    pattern_id: str
    name: str
    effect_domain: str
    mechanism: str
    source_pattern: str
    target_pattern: str
    applicability_boundary: str
    source_specification: str = ""
    target_requirements: tuple[str, ...] = ()
    join_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    content: str
    subject: str
    source_memory_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRecord:
    memory_id: str
    source_memory_id: str | None
    candidate_role: str | None
    retrieval_label: str | None
    label_status: str | None
    gold_status: str | None
    negative_type: str | None
    relation_pattern_id: str | None


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    query_id: str
    query: str
    language: str
    task_family: str
    scenario_group_id: str
    pattern_id: str | None
    candidates: tuple[CandidateRecord, ...]
    strict_relevant_memory_ids: tuple[str, ...]
    provisional_relevant_memory_ids: tuple[str, ...]
    assumed_reviewed_relevant_memory_ids: tuple[str, ...]
    judged_negative_memory_ids: tuple[str, ...]
    unjudged_memory_ids: tuple[str, ...]


@dataclass
class ExperimentData:
    memories: dict[str, MemoryRecord]
    cases: list[QueryCase]
    memory_views: dict[str, list[TextView]]
    query_views: dict[str, list[TextView]]
    memory_bindings: dict[str, list[RelationBinding]]
    query_bindings: dict[str, list[RelationBinding]]
    patterns: dict[str, PatternDefinition]
    memory_representation_records: set[str] = field(default_factory=set)
    query_representation_records: set[str] = field(default_factory=set)
    memory_binding_records: set[str] = field(default_factory=set)
    query_binding_records: set[str] = field(default_factory=set)
    source_to_canonical: dict[str, str] = field(default_factory=dict)
    input_files: dict[str, str] = field(default_factory=dict)
    data_warnings: list[dict[str, Any]] = field(default_factory=list)

    def views(self, side: Side, owner_id: str, kinds: tuple[str, ...]) -> list[TextView]:
        table = self.memory_views if side == "memory" else self.query_views
        allowed = set(kinds)
        return [view for view in table.get(owner_id, []) if view.kind in allowed]


@dataclass(frozen=True)
class CandidateScore:
    memory_id: str
    score: float | None
    route: str
    trace: dict[str, Any]


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    memory_id: str
    score: float | None
    route: str
    candidate_role: str | None
    retrieval_label: str | None
    label_status: str | None
    gold_status: str | None
    negative_type: str | None
    relation_pattern_id: str | None
    trace: dict[str, Any]


@dataclass(frozen=True)
class CaseResult:
    profile: str
    case_id: str
    query_id: str
    task_family: str
    language: str
    pattern_id: str | None
    ranked_candidates: tuple[RankedCandidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

