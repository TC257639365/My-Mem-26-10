"""Adapters for the stage3 controlled-pool and auxiliary relation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    CandidateRecord,
    ExperimentData,
    MemoryRecord,
    PatternDefinition,
    QueryCase,
    RelationBinding,
    TextView,
)


REPRESENTATION_KIND = {
    "same_length_unstructured_rewrite": "rewrite",
    "prospective_trigger": "prospective_trigger",
    "instance_source_factor": "source_factor",
    "source_role_descriptor": "source_role",
    "query_expansion": "query_expansion",
    "query_requirement_factor": "query_requirement",
    "target_role_descriptor": "target_role",
}

FROZEN_MEMORY_VIEW_KIND = {
    "raw": "raw",
    "subject": "subject",
    "rewrite": "rewrite",
    "source_factor": "source_factor",
    "source_role": "source_role",
    "source_assertion": "source_assertion",
}

FROZEN_QUERY_VIEW_KIND = {
    "raw": "raw",
    "ordinary_expansion": "query_expansion",
    "target_factor": "query_requirement",
    "target_role": "target_role",
    "hard_constraint": "hard_constraint",
    "open_condition": "open_condition",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return read_jsonl(path)


def _canonical_id(owner_id: str, mapping: dict[str, str], memories: dict[str, MemoryRecord]) -> str:
    if owner_id in memories:
        return owner_id
    return mapping.get(owner_id, owner_id)


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_experiment_data(
    *,
    test_cases_path: str | Path,
    memory_catalog_path: str | Path,
    representations_path: str | Path | None = None,
    source_bindings_path: str | Path | None = None,
    query_requirements_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    memory_auxiliary_path: str | Path | None = None,
    query_auxiliary_path: str | Path | None = None,
) -> ExperimentData:
    test_cases_path = Path(test_cases_path)
    memory_catalog_path = Path(memory_catalog_path)
    optional_paths = {
        "representations": Path(representations_path) if representations_path else None,
        "source_bindings": Path(source_bindings_path) if source_bindings_path else None,
        "query_requirements": Path(query_requirements_path) if query_requirements_path else None,
        "registry": Path(registry_path) if registry_path else None,
        "memory_auxiliary": Path(memory_auxiliary_path) if memory_auxiliary_path else None,
        "query_auxiliary": Path(query_auxiliary_path) if query_auxiliary_path else None,
    }

    raw_cases = read_jsonl(test_cases_path)
    raw_memories = read_jsonl(memory_catalog_path)

    source_to_canonical: dict[str, str] = {}
    for raw_case in raw_cases:
        for candidate in raw_case.get("candidates", []):
            source_id = candidate.get("source_memory_id")
            canonical_id = candidate["canonical_memory_id"]
            if source_id:
                previous = source_to_canonical.setdefault(source_id, canonical_id)
                if previous != canonical_id:
                    raise ValueError(
                        f"source memory {source_id!r} maps to multiple canonical IDs: "
                        f"{previous!r}, {canonical_id!r}"
                    )

    memories: dict[str, MemoryRecord] = {}
    memory_views: dict[str, list[TextView]] = defaultdict(list)
    for raw in raw_memories:
        memory_id = raw["canonical_memory_id"]
        source_ids = tuple(
            sorted(
                source_id
                for source_id, canonical_id in source_to_canonical.items()
                if canonical_id == memory_id
            )
        )
        memories[memory_id] = MemoryRecord(
            memory_id=memory_id,
            content=raw["content"],
            subject=raw.get("subject") or "",
            source_memory_ids=source_ids,
            metadata={"content_sha256_20": raw.get("content_sha256_20")},
        )
        memory_views[memory_id].append(
            TextView(memory_id, "memory", "raw", raw["content"], provenance={"source": "catalog"})
        )
        if raw.get("subject"):
            memory_views[memory_id].append(
                TextView(
                    memory_id,
                    "memory",
                    "subject",
                    raw["subject"],
                    provenance={"source": "catalog"},
                )
            )

    cases: list[QueryCase] = []
    query_views: dict[str, list[TextView]] = defaultdict(list)
    for raw in raw_cases:
        candidates = tuple(
            CandidateRecord(
                memory_id=item["canonical_memory_id"],
                source_memory_id=item.get("source_memory_id"),
                candidate_role=item.get("candidate_role"),
                retrieval_label=item.get("retrieval_label"),
                label_status=item.get("label_status"),
                gold_status=item.get("gold_status"),
                negative_type=item.get("negative_type"),
                relation_pattern_id=item.get("relation_pattern_id"),
            )
            for item in raw.get("candidates", [])
        )
        case = QueryCase(
            case_id=raw["case_id"],
            query_id=raw["query_id"],
            query=raw["query"],
            language=raw.get("language") or "unknown",
            task_family=raw.get("task_family") or "unknown",
            scenario_group_id=raw.get("scenario_group_id") or raw["case_id"],
            pattern_id=raw.get("pattern_id"),
            candidates=candidates,
            strict_relevant_memory_ids=tuple(raw.get("strict_relevant_memory_ids", [])),
            provisional_relevant_memory_ids=tuple(
                raw.get("provisional_relevant_memory_ids", [])
            ),
            assumed_reviewed_relevant_memory_ids=tuple(
                raw.get("assumed_reviewed_relevant_memory_ids", [])
            ),
            judged_negative_memory_ids=tuple(raw.get("judged_negative_memory_ids", [])),
            unjudged_memory_ids=tuple(raw.get("unjudged_memory_ids", [])),
        )
        cases.append(case)
        query_views[case.query_id].append(
            TextView(
                case.query_id,
                "query",
                "raw",
                case.query,
                provenance={"source": "test_cases"},
            )
        )

    memory_representation_records: set[str] = set()
    query_representation_records: set[str] = set()
    memory_bindings: dict[str, list[RelationBinding]] = defaultdict(list)
    memory_binding_records: set[str] = set()
    query_bindings: dict[str, list[RelationBinding]] = defaultdict(list)
    query_binding_records: set[str] = set()
    data_warnings: list[dict[str, Any]] = []
    auxiliary_memory_ids: set[str] = set()
    auxiliary_query_ids: set[str] = set()
    for raw in _optional_jsonl(optional_paths["representations"]):
        kind = REPRESENTATION_KIND.get(raw.get("representation_type", ""))
        if kind is None or not str(raw.get("generated_text", "")).strip():
            continue
        side = raw.get("input_side")
        owner_id = str(raw.get("input_id", ""))
        isolation = str(raw.get("input_isolation") or "")
        if side == "memory":
            if isolation and not isolation.startswith("memory_only"):
                raise ValueError(
                    f"memory representation {owner_id!r} violates input isolation: {isolation!r}"
                )
            owner_id = _canonical_id(owner_id, source_to_canonical, memories)
            memory_representation_records.add(owner_id)
            target = memory_views
        elif side == "query":
            if isolation and not isolation.startswith("query_only"):
                raise ValueError(
                    f"query representation {owner_id!r} violates input isolation: {isolation!r}"
                )
            query_representation_records.add(owner_id)
            target = query_views
        else:
            continue
        target[owner_id].append(
            TextView(
                owner_id=owner_id,
                side=side,
                kind=kind,
                text=raw["generated_text"],
                provenance={
                    "generator_model": raw.get("generator_model"),
                    "run_id": raw.get("run_id"),
                    "gold_status": raw.get("gold_status"),
                    "input_isolation": raw.get("input_isolation"),
                },
            )
        )

    # stage3_auxiliary_v2 keeps every memory-side view and binding in one isolated
    # record.  It is loaded directly rather than flattened into the legacy three-file
    # schema, so audit and boundary fields remain available in traces.
    for raw in _optional_jsonl(optional_paths["memory_auxiliary"]):
        owner_id = _canonical_id(
            str(raw.get("canonical_memory_id", "")), source_to_canonical, memories
        )
        if owner_id not in memories:
            raise ValueError(f"unknown canonical memory in auxiliary data: {owner_id!r}")
        if owner_id in auxiliary_memory_ids:
            raise ValueError(f"duplicate memory auxiliary record: {owner_id!r}")
        auxiliary_memory_ids.add(owner_id)
        expected_hash = hashlib.sha256(memories[owner_id].content.encode("utf-8")).hexdigest()
        if raw.get("content_sha256") != expected_hash:
            raise ValueError(f"memory auxiliary content hash mismatch: {owner_id!r}")
        provenance = dict(raw.get("generation_provenance") or {})
        isolation = str(provenance.get("input_isolation") or "")
        if isolation and not isolation.startswith("memory_only"):
            raise ValueError(
                f"memory auxiliary {owner_id!r} violates input isolation: {isolation!r}"
            )
        memory_representation_records.add(owner_id)
        rewrite = str(raw.get("same_length_unstructured_rewrite") or "").strip()
        if rewrite:
            memory_views[owner_id].append(
                TextView(owner_id, "memory", "rewrite", rewrite, provenance=provenance)
            )
        for index, factor in enumerate(raw.get("source_factors") or []):
            factor_provenance = {**provenance, "factor_index": index}
            factor_text = str(factor.get("factor_text") or "").strip()
            role_text = str(factor.get("source_role_descriptor") or "").strip()
            if factor_text:
                memory_views[owner_id].append(
                    TextView(
                        owner_id, "memory", "source_factor", factor_text,
                        provenance=factor_provenance,
                    )
                )
            if role_text:
                memory_views[owner_id].append(
                    TextView(
                        owner_id, "memory", "source_role", role_text,
                        provenance=factor_provenance,
                    )
                )
        memories[owner_id].metadata.update(
            {
                "entity_scope": raw.get("entity_scope") or {},
                "temporal_scope": raw.get("temporal_scope") or {},
                "auxiliary_audit_status": provenance.get("audit_status"),
                "unmatched_reason": raw.get("unmatched_reason"),
            }
        )

        bindings = raw.get("candidate_bindings") or []
        unmatched_reason = str(raw.get("unmatched_reason") or "").strip()
        if bindings and unmatched_reason:
            data_warnings.append(
                {
                    "code": "binding_and_unmatched_reason",
                    "owner_id": owner_id,
                    "message": "candidate_bindings take precedence over unmatched_reason",
                }
            )
        memory_binding_records.add(owner_id)
        for item in bindings:
            memory_bindings[owner_id].append(
                RelationBinding(
                    owner_id=owner_id,
                    side="memory",
                    pattern_id=item["pattern_id"],
                    role_text=str(item.get("source_role_descriptor") or ""),
                    source_assertions=_tuple_strings(item.get("source_assertions")),
                    confidence=item.get("extraction_confidence"),
                    entity_scope=raw.get("entity_scope") or {},
                    temporal_scope=raw.get("temporal_scope") or {},
                    audit_status=provenance.get("audit_status"),
                )
            )

    for raw in _optional_jsonl(optional_paths["query_auxiliary"]):
        owner_id = str(raw.get("query_id", ""))
        if owner_id not in query_views:
            raise ValueError(f"unknown query in auxiliary data: {owner_id!r}")
        if owner_id in auxiliary_query_ids:
            raise ValueError(f"duplicate query auxiliary record: {owner_id!r}")
        auxiliary_query_ids.add(owner_id)
        query_text = next(view.text for view in query_views[owner_id] if view.kind == "raw")
        expected_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        if raw.get("query_sha256") != expected_hash:
            raise ValueError(f"query auxiliary content hash mismatch: {owner_id!r}")
        provenance = dict(raw.get("generation_provenance") or {})
        isolation = str(provenance.get("input_isolation") or "")
        if isolation and not isolation.startswith("query_only"):
            raise ValueError(
                f"query auxiliary {owner_id!r} violates input isolation: {isolation!r}"
            )
        query_representation_records.add(owner_id)
        for index, expansion in enumerate(raw.get("ordinary_query_expansions") or []):
            text = str(expansion or "").strip()
            if text:
                query_views[owner_id].append(
                    TextView(
                        owner_id, "query", "query_expansion", text,
                        provenance={**provenance, "expansion_index": index},
                    )
                )
        for index, factor in enumerate(raw.get("target_factors") or []):
            factor_provenance = {**provenance, "factor_index": index}
            factor_text = str(factor.get("factor_text") or "").strip()
            role_text = str(factor.get("target_role_descriptor") or "").strip()
            if factor_text:
                query_views[owner_id].append(
                    TextView(
                        owner_id, "query", "query_requirement", factor_text,
                        provenance={
                            **factor_provenance,
                            "hard_constraints": factor.get("hard_constraints") or [],
                            "open_conditions": factor.get("open_conditions") or [],
                        },
                    )
                )
            if role_text:
                query_views[owner_id].append(
                    TextView(
                        owner_id, "query", "target_role", role_text,
                        provenance=factor_provenance,
                    )
                )

        bindings = raw.get("candidate_bindings") or []
        unmatched_reason = str(raw.get("unmatched_reason") or "").strip()
        if bindings and unmatched_reason:
            data_warnings.append(
                {
                    "code": "binding_and_unmatched_reason",
                    "owner_id": owner_id,
                    "message": "candidate_bindings take precedence over unmatched_reason",
                }
            )
        query_binding_records.add(owner_id)
        for item in bindings:
            query_bindings[owner_id].append(
                RelationBinding(
                    owner_id=owner_id,
                    side="query",
                    pattern_id=item["pattern_id"],
                    role_text=str(item.get("target_role_descriptor") or ""),
                    target_requirements=_tuple_strings(item.get("target_requirements")),
                    hard_constraints=_tuple_strings(item.get("hard_constraints")),
                    open_conditions=_tuple_strings(item.get("open_conditions")),
                    confidence=item.get("activation_confidence"),
                    audit_status=provenance.get("audit_status"),
                )
            )

    for raw in _optional_jsonl(optional_paths["source_bindings"]):
        isolation = str(raw.get("input_isolation") or "")
        if isolation and not isolation.startswith("memory_only"):
            raise ValueError(
                f"source binding {raw.get('memory_id')!r} violates input isolation: {isolation!r}"
            )
        owner_id = _canonical_id(str(raw["memory_id"]), source_to_canonical, memories)
        memory_binding_records.add(owner_id)
        candidate_bindings = raw.get("candidate_bindings") or []
        if not candidate_bindings:
            candidate_bindings = [
                {
                    "pattern_id": pattern_id,
                    "source_role_descriptor": raw.get("source_role_descriptor", ""),
                    "source_assertions": raw.get("source_assertions"),
                    "applicability_conditions": raw.get("applicability_conditions", []),
                    "join_constraints": raw.get("join_constraints", []),
                    "extraction_confidence": raw.get("extraction_confidence"),
                }
                for pattern_id in raw.get("candidate_pattern_ids", [])
            ]
        for item in candidate_bindings:
            source_assertions = item.get("source_assertions")
            legacy = source_assertions is None
            if source_assertions is None:
                source_assertions = item.get("applicability_conditions", [])
            binding = RelationBinding(
                owner_id=owner_id,
                side="memory",
                pattern_id=item["pattern_id"],
                role_text=item.get("source_role_descriptor")
                or raw.get("source_role_descriptor", ""),
                source_assertions=_tuple_strings(source_assertions),
                target_requirements=_tuple_strings(item.get("target_requirements")),
                join_constraints=_tuple_strings(item.get("join_constraints")),
                confidence=item.get("extraction_confidence"),
                legacy_mixed_conditions=legacy,
            )
            memory_bindings[owner_id].append(binding)

    for raw in _optional_jsonl(optional_paths["query_requirements"]):
        isolation = str(raw.get("input_isolation") or "")
        if isolation and not isolation.startswith("query_only"):
            raise ValueError(
                f"query requirement {raw.get('query_id')!r} violates input isolation: {isolation!r}"
            )
        owner_id = str(raw["query_id"])
        query_binding_records.add(owner_id)
        for pattern_id in raw.get("candidate_pattern_ids", []):
            query_bindings[owner_id].append(
                RelationBinding(
                    owner_id=owner_id,
                    side="query",
                    pattern_id=pattern_id,
                    role_text=raw.get("target_role_descriptor", ""),
                    target_requirements=_tuple_strings(
                        raw.get("target_requirements") or raw.get("hard_constraints")
                    ),
                    join_constraints=_tuple_strings(raw.get("join_constraints")),
                    legacy_mixed_conditions=False,
                )
            )

    patterns: dict[str, PatternDefinition] = {}
    registry_path_obj = optional_paths["registry"]
    if registry_path_obj is not None and registry_path_obj.exists():
        registry = read_json(registry_path_obj)
        for raw in registry.get("patterns", []):
            pattern = PatternDefinition(
                pattern_id=raw["pattern_id"],
                name=raw["name"],
                effect_domain=raw.get("effect_domain", ""),
                mechanism=raw.get("mechanism", ""),
                source_pattern=raw["source_pattern"],
                target_pattern=raw["target_pattern"],
                applicability_boundary=raw.get("applicability_boundary", ""),
                source_specification=raw.get("source_specification", ""),
                target_requirements=_tuple_strings(raw.get("target_requirements")),
                join_constraints=_tuple_strings(raw.get("join_constraints")),
            )
            patterns[pattern.pattern_id] = pattern

    unknown_pattern_ids = sorted(
        {
            binding.pattern_id
            for table in (memory_bindings, query_bindings)
            for bindings in table.values()
            for binding in bindings
        }
        - set(patterns)
    )
    if patterns and unknown_pattern_ids:
        raise ValueError(
            "auxiliary bindings reference unknown patterns: " + ", ".join(unknown_pattern_ids)
        )

    input_files = {
        "test_cases": file_sha256(test_cases_path),
        "memory_catalog": file_sha256(memory_catalog_path),
    }
    for name, path in optional_paths.items():
        if path is not None and path.exists():
            input_files[name] = file_sha256(path)

    return ExperimentData(
        memories=memories,
        cases=cases,
        memory_views=dict(memory_views),
        query_views=dict(query_views),
        memory_bindings=dict(memory_bindings),
        query_bindings=dict(query_bindings),
        patterns=patterns,
        memory_representation_records=memory_representation_records,
        query_representation_records=query_representation_records,
        memory_binding_records=memory_binding_records,
        query_binding_records=query_binding_records,
        source_to_canonical=source_to_canonical,
        input_files=input_files,
        data_warnings=data_warnings,
    )


def default_data_paths(project_root: str | Path | None = None) -> dict[str, Path | None]:
    project = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    auxiliary = project / "results" / "stage3_test1" / "auxiliary_v2"
    return {
        "test_cases_path": project / "results" / "stage3_test1" / "test_cases.jsonl",
        "memory_catalog_path": project / "results" / "stage3_test1" / "memory_catalog.jsonl",
        "representations_path": None,
        "source_bindings_path": None,
        "query_requirements_path": None,
        "memory_auxiliary_path": auxiliary / "memory_auxiliary.jsonl",
        "query_auxiliary_path": auxiliary / "query_auxiliary.jsonl",
        "registry_path": auxiliary / "relation_patterns_v2.json",
    }


def unique_candidate_memory_ids(cases: Iterable[QueryCase]) -> set[str]:
    return {candidate.memory_id for case in cases for candidate in case.candidates}


def load_frozen_experiment_data(dataset_root: str | Path) -> ExperimentData:
    """Load the isolation-partitioned My-Mem first-stage experiment package."""

    root = Path(dataset_root)
    validation = read_json(root / "validation_report.json")
    if validation.get("status") != "passed":
        raise ValueError(
            f"frozen experiment dataset is not ready: {validation.get('status')!r}"
        )

    paths = {
        "memories": root / "index_safe" / "memories.jsonl",
        "memory_views": root / "index_safe" / "memory_views.jsonl",
        "memory_bindings": root / "index_safe" / "memory_bindings.jsonl",
        "queries": root / "query_only" / "queries.jsonl",
        "query_views": root / "query_only" / "query_views.jsonl",
        "query_bindings": root / "query_only" / "query_bindings.jsonl",
        "cases": root / "evaluation_only" / "cases.jsonl",
        "registry": root / "config" / "relation_patterns_v2.json",
        "manifest": root / "manifest.json",
        "validation": root / "validation_report.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen experiment files: " + ", ".join(missing))

    raw_memories = read_jsonl(paths["memories"])
    raw_queries = read_jsonl(paths["queries"])
    raw_cases = read_jsonl(paths["cases"])
    query_by_id = {row["query_id"]: row for row in raw_queries}

    memories: dict[str, MemoryRecord] = {}
    for raw in raw_memories:
        memory_id = raw["memory_id"]
        if raw.get("input_isolation") != "memory_only_index_safe":
            raise ValueError(f"unsafe memory input boundary: {memory_id!r}")
        memories[memory_id] = MemoryRecord(
            memory_id=memory_id,
            content=raw["content"],
            subject=raw.get("subject") or "",
            metadata={
                "language": raw.get("language"),
                "content_sha256": raw.get("content_sha256"),
                "scene_id": raw.get("scene_id"),
                "item_id": raw.get("item_id"),
            },
        )

    memory_views: dict[str, list[TextView]] = defaultdict(list)
    for raw in read_jsonl(paths["memory_views"]):
        memory_id = raw["memory_id"]
        if memory_id not in memories:
            raise ValueError(f"unknown memory view owner: {memory_id!r}")
        kind = FROZEN_MEMORY_VIEW_KIND.get(raw["view_type"])
        if kind is None:
            continue
        provenance = dict(raw.get("provenance") or {})
        provenance["view_id"] = raw["view_id"]
        memory_views[memory_id].append(
            TextView(memory_id, "memory", kind, raw["text"], provenance=provenance)
        )

    cases: list[QueryCase] = []
    source_to_canonical: dict[str, str] = {}
    for raw in raw_cases:
        if raw["query_id"] not in query_by_id:
            raise ValueError(f"case references unknown query: {raw['query_id']!r}")
        candidates = []
        for item in raw.get("candidates", []):
            source_id = item.get("source_memory_id")
            canonical_id = item["canonical_memory_id"]
            if source_id:
                previous = source_to_canonical.setdefault(source_id, canonical_id)
                if previous != canonical_id:
                    raise ValueError(f"inconsistent source mapping: {source_id!r}")
            candidates.append(
                CandidateRecord(
                    memory_id=canonical_id,
                    source_memory_id=source_id,
                    candidate_role=item.get("candidate_role"),
                    retrieval_label=item.get("retrieval_label"),
                    label_status=item.get("label_status"),
                    gold_status=item.get("gold_status"),
                    negative_type=item.get("negative_type"),
                    relation_pattern_id=item.get("relation_pattern_id"),
                )
            )
        cases.append(
            QueryCase(
                case_id=raw["case_id"],
                query_id=raw["query_id"],
                query=raw["query"],
                language=raw.get("language") or "unknown",
                task_family=raw.get("task_family") or "unknown",
                scenario_group_id=raw.get("scenario_group_id") or raw["case_id"],
                pattern_id=raw.get("pattern_id"),
                candidates=tuple(candidates),
                strict_relevant_memory_ids=tuple(raw.get("strict_relevant_memory_ids", [])),
                provisional_relevant_memory_ids=tuple(
                    raw.get("provisional_relevant_memory_ids", [])
                ),
                assumed_reviewed_relevant_memory_ids=tuple(
                    raw.get("assumed_reviewed_relevant_memory_ids", [])
                ),
                judged_negative_memory_ids=tuple(raw.get("judged_negative_memory_ids", [])),
                unjudged_memory_ids=tuple(raw.get("unjudged_memory_ids", [])),
            )
        )

    for memory_id, memory in list(memories.items()):
        memories[memory_id] = MemoryRecord(
            memory_id=memory.memory_id,
            content=memory.content,
            subject=memory.subject,
            source_memory_ids=tuple(
                sorted(
                    source_id
                    for source_id, canonical_id in source_to_canonical.items()
                    if canonical_id == memory_id
                )
            ),
            metadata=memory.metadata,
        )

    query_views: dict[str, list[TextView]] = defaultdict(list)
    for raw in read_jsonl(paths["query_views"]):
        query_id = raw["query_id"]
        if query_id not in query_by_id:
            raise ValueError(f"unknown query view owner: {query_id!r}")
        kind = FROZEN_QUERY_VIEW_KIND.get(raw["view_type"])
        if kind is None:
            continue
        provenance = dict(raw.get("provenance") or {})
        provenance["view_id"] = raw["view_id"]
        query_views[query_id].append(
            TextView(query_id, "query", kind, raw["text"], provenance=provenance)
        )

    memory_bindings: dict[str, list[RelationBinding]] = defaultdict(list)
    memory_binding_records: set[str] = set()
    for raw in read_jsonl(paths["memory_bindings"]):
        memory_id = raw["memory_id"]
        isolation = str(raw.get("input_isolation") or "")
        if not isolation.startswith("memory_only"):
            raise ValueError(f"unsafe memory binding boundary: {memory_id!r}")
        memory_binding_records.add(memory_id)
        provenance = raw.get("generation_provenance") or {}
        memories[memory_id].metadata.update(
            {
                "entity_scope": raw.get("entity_scope") or {},
                "temporal_scope": raw.get("temporal_scope") or {},
                "auxiliary_audit_status": provenance.get("audit_status"),
                "unmatched_reason": raw.get("unmatched_reason"),
            }
        )
        for item in raw.get("candidate_bindings") or []:
            memory_bindings[memory_id].append(
                RelationBinding(
                    owner_id=memory_id,
                    side="memory",
                    pattern_id=item["pattern_id"],
                    role_text=str(item.get("source_role_descriptor") or ""),
                    source_assertions=_tuple_strings(item.get("source_assertions")),
                    confidence=item.get("extraction_confidence"),
                    entity_scope=raw.get("entity_scope") or {},
                    temporal_scope=raw.get("temporal_scope") or {},
                    audit_status=provenance.get("audit_status"),
                )
            )

    query_bindings: dict[str, list[RelationBinding]] = defaultdict(list)
    query_binding_records: set[str] = set()
    for raw in read_jsonl(paths["query_bindings"]):
        query_id = raw["query_id"]
        isolation = str(raw.get("input_isolation") or "")
        if not isolation.startswith("query_only"):
            raise ValueError(f"unsafe query binding boundary: {query_id!r}")
        query_binding_records.add(query_id)
        provenance = raw.get("generation_provenance") or {}
        for item in raw.get("candidate_bindings") or []:
            query_bindings[query_id].append(
                RelationBinding(
                    owner_id=query_id,
                    side="query",
                    pattern_id=item["pattern_id"],
                    role_text=str(item.get("target_role_descriptor") or ""),
                    target_requirements=_tuple_strings(item.get("target_requirements")),
                    hard_constraints=_tuple_strings(item.get("hard_constraints")),
                    open_conditions=_tuple_strings(item.get("open_conditions")),
                    confidence=item.get("activation_confidence"),
                    audit_status=provenance.get("audit_status"),
                )
            )

    patterns: dict[str, PatternDefinition] = {}
    for raw in read_json(paths["registry"]).get("patterns", []):
        pattern = PatternDefinition(
            pattern_id=raw["pattern_id"],
            name=raw["name"],
            effect_domain=raw.get("effect_domain", ""),
            mechanism=raw.get("mechanism", ""),
            source_pattern=raw["source_pattern"],
            target_pattern=raw["target_pattern"],
            applicability_boundary=raw.get("applicability_boundary", ""),
            source_specification=raw.get("source_specification", ""),
            target_requirements=_tuple_strings(raw.get("target_requirements")),
            join_constraints=_tuple_strings(raw.get("join_constraints")),
        )
        patterns[pattern.pattern_id] = pattern

    unknown = sorted(
        {
            binding.pattern_id
            for table in (memory_bindings, query_bindings)
            for bindings in table.values()
            for binding in bindings
        }
        - set(patterns)
    )
    if unknown:
        raise ValueError("frozen bindings reference unknown patterns: " + ", ".join(unknown))

    return ExperimentData(
        memories=memories,
        cases=cases,
        memory_views=dict(memory_views),
        query_views=dict(query_views),
        memory_bindings=dict(memory_bindings),
        query_bindings=dict(query_bindings),
        patterns=patterns,
        memory_representation_records=set(memories),
        query_representation_records=set(query_by_id),
        memory_binding_records=memory_binding_records,
        query_binding_records=query_binding_records,
        source_to_canonical=source_to_canonical,
        input_files={name: file_sha256(path) for name, path in paths.items()},
    )

