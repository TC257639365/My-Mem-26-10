"""Dataset readiness checks and auxiliary-generation worklists."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .contracts import ExperimentData
from .datasets import unique_candidate_memory_ids
from .profiles import DEFAULT_PROFILES, ProfileSpec


def profile_readiness(data: ExperimentData, profile: ProfileSpec) -> dict[str, Any]:
    if profile.family == "union":
        members = [profile_readiness(data, DEFAULT_PROFILES[name]) for name in profile.union_members]
        return {
            "profile": profile.name,
            "ready": all(item["ready"] for item in members),
            "union_members": {item["profile"]: item for item in members},
            "missing_memory_view_count": sum(
                item["missing_memory_view_count"] for item in members
            ),
            "missing_query_view_count": sum(
                item["missing_query_view_count"] for item in members
            ),
            "missing_memory_binding_record_count": sum(
                item["missing_memory_binding_record_count"] for item in members
            ),
            "missing_query_binding_record_count": sum(
                item["missing_query_binding_record_count"] for item in members
            ),
            "legacy_mixed_binding_count": sum(
                item["legacy_mixed_binding_count"] for item in members
            ),
        }
    candidate_ids = unique_candidate_memory_ids(data.cases)
    query_ids = {case.query_id for case in data.cases}
    missing_memory_views: dict[str, list[str]] = {}
    missing_query_views: dict[str, list[str]] = {}
    missing_memory_binding_records: list[str] = []
    missing_query_binding_records: list[str] = []
    legacy_mixed_binding_count = 0

    if profile.family == "dense":
        for memory_id in sorted(candidate_ids):
            missing = [
                kind
                for kind in profile.memory_view_kinds
                if not data.views("memory", memory_id, (kind,))
            ]
            if missing:
                missing_memory_views[memory_id] = missing
        for query_id in sorted(query_ids):
            missing = [
                kind
                for kind in profile.query_view_kinds
                if not data.views("query", query_id, (kind,))
            ]
            if missing:
                missing_query_views[query_id] = missing

    if profile.family == "registry":
        missing_memory_binding_records = sorted(candidate_ids - data.memory_binding_records)
        missing_query_binding_records = sorted(query_ids - data.query_binding_records)
        legacy_mixed_binding_count = sum(
            binding.legacy_mixed_conditions
            for bindings in data.memory_bindings.values()
            for binding in bindings
        )

    ready = not any(
        (
            missing_memory_views,
            missing_query_views,
            missing_memory_binding_records,
            missing_query_binding_records,
            legacy_mixed_binding_count,
        )
    )
    return {
        "profile": profile.name,
        "ready": ready,
        "missing_memory_view_count": len(missing_memory_views),
        "missing_query_view_count": len(missing_query_views),
        "missing_memory_binding_record_count": len(missing_memory_binding_records),
        "missing_query_binding_record_count": len(missing_query_binding_records),
        "legacy_mixed_binding_count": legacy_mixed_binding_count,
        "missing_memory_views": missing_memory_views,
        "missing_query_views": missing_query_views,
        "missing_memory_binding_records": missing_memory_binding_records,
        "missing_query_binding_records": missing_query_binding_records,
    }


def build_preflight_report(
    data: ExperimentData,
    profiles: list[ProfileSpec] | None = None,
) -> dict[str, Any]:
    candidate_ids = unique_candidate_memory_ids(data.cases)
    query_ids = {case.query_id for case in data.cases}
    legacy_bindings = [
        binding
        for bindings in data.memory_bindings.values()
        for binding in bindings
        if binding.legacy_mixed_conditions
    ]
    task_counts = Counter(case.task_family for case in data.cases)
    language_counts = Counter(case.language for case in data.cases)
    memory_view_counts = Counter(
        view.kind for views in data.memory_views.values() for view in views
    )
    query_view_counts = Counter(
        view.kind for views in data.query_views.values() for view in views
    )
    audit_status_counts = Counter(
        binding.audit_status or "unspecified"
        for table in (data.memory_bindings, data.query_bindings)
        for bindings in table.values()
        for binding in bindings
    )

    report: dict[str, Any] = {
        "summary": {
            "case_count": len(data.cases),
            "candidate_memory_count": len(candidate_ids),
            "catalog_memory_count": len(data.memories),
            "query_count": len(query_ids),
            "pattern_count": len(data.patterns),
            "task_family_counts": dict(task_counts),
            "language_counts": dict(language_counts),
        },
        "auxiliary_coverage": {
            "memory_representation_records": len(
                candidate_ids & data.memory_representation_records
            ),
            "memory_representation_target": len(candidate_ids),
            "query_representation_records": len(query_ids & data.query_representation_records),
            "query_representation_target": len(query_ids),
            "memory_binding_records": len(candidate_ids & data.memory_binding_records),
            "memory_binding_target": len(candidate_ids),
            "query_binding_records": len(query_ids & data.query_binding_records),
            "query_binding_target": len(query_ids),
            "legacy_mixed_condition_binding_count": len(legacy_bindings),
            "memory_view_counts": dict(memory_view_counts),
            "query_view_counts": dict(query_view_counts),
            "binding_audit_status_counts": dict(audit_status_counts),
        },
        "data_warnings": data.data_warnings,
        "worklist": {
            "memory_ids_missing_representation_record": sorted(
                candidate_ids - data.memory_representation_records
            ),
            "memory_ids_missing_binding_record": sorted(
                candidate_ids - data.memory_binding_records
            ),
            "query_ids_missing_representation_record": sorted(
                query_ids - data.query_representation_records
            ),
            "query_ids_missing_binding_record": sorted(query_ids - data.query_binding_records),
        },
        "input_files": data.input_files,
    }
    if profiles is not None:
        report["profiles"] = {
            profile.name: profile_readiness(data, profile) for profile in profiles
        }
    return report


def assert_profiles_ready(
    data: ExperimentData,
    profiles: list[ProfileSpec],
    *,
    allow_incomplete: bool,
) -> dict[str, dict[str, Any]]:
    readiness = {profile.name: profile_readiness(data, profile) for profile in profiles}
    failed = [name for name, item in readiness.items() if not item["ready"]]
    if failed and not allow_incomplete:
        summary = ", ".join(
            f"{name}(missing memory views={readiness[name]['missing_memory_view_count']}, "
            f"query views={readiness[name]['missing_query_view_count']}, "
            f"memory bindings={readiness[name]['missing_memory_binding_record_count']}, "
            f"query bindings={readiness[name]['missing_query_binding_record_count']}, "
            f"legacy mixed bindings={readiness[name]['legacy_mixed_binding_count']})"
            for name in failed
        )
        raise ValueError(
            "profiles are not ready: "
            + summary
            + "; run preflight or pass allow_incomplete=True for diagnostic abstention"
        )
    return readiness

