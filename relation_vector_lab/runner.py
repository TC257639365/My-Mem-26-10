"""One runner for all incremental P0-P4 profiles."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .contracts import CaseResult, ExperimentData, RankedCandidate
from .embeddings import CachedEmbedder, Embedder
from .evaluation import evaluate_results
from .preflight import assert_profiles_ready, build_preflight_report
from .profiles import DEFAULT_PROFILES, ProfileSpec, get_profiles
from .scoring import (
    rank_scores,
    rank_union,
    score_dense,
    score_keyword,
    score_registry,
)


def _expand_dependencies(profiles: list[ProfileSpec]) -> list[ProfileSpec]:
    names = [profile.name for profile in profiles]
    for profile in profiles:
        for member in profile.union_members:
            if member not in names:
                names.append(member)
    return get_profiles(names)


def _case_result(
    profile: ProfileSpec,
    case,
    ranking,
    *,
    metadata: dict[str, Any] | None = None,
) -> CaseResult:
    ranked = tuple(
        RankedCandidate(
            rank=rank,
            memory_id=candidate.memory_id,
            score=score.score,
            route=score.route,
            candidate_role=candidate.candidate_role,
            retrieval_label=candidate.retrieval_label,
            label_status=candidate.label_status,
            gold_status=candidate.gold_status,
            negative_type=candidate.negative_type,
            relation_pattern_id=candidate.relation_pattern_id,
            trace=score.trace,
        )
        for rank, (candidate, score) in enumerate(ranking, start=1)
    )
    return CaseResult(
        profile=profile.name,
        case_id=case.case_id,
        query_id=case.query_id,
        task_family=case.task_family,
        language=case.language,
        pattern_id=case.pattern_id,
        ranked_candidates=ranked,
        metadata=metadata or {},
    )


def run_profiles(
    data: ExperimentData,
    profile_names: list[str],
    *,
    embedder: Embedder,
    allow_incomplete: bool = False,
    union_depth: int = 5,
    view_budget: int | None = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, list[CaseResult]], dict[str, Any]]:
    if view_budget is not None and view_budget < 1:
        raise ValueError("view_budget must be at least 1 or None")
    requested = get_profiles(profile_names)
    expanded = _expand_dependencies(requested)
    readiness = assert_profiles_ready(data, expanded, allow_incomplete=allow_incomplete)
    cached = CachedEmbedder(embedder)
    total_case_steps = max(1, len(data.cases) * len(expanded))
    completed_case_steps = 0

    def report(profile: ProfileSpec, case_index: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "profile": profile.name,
                "profile_stage": profile.stage,
                "case_index": case_index,
                "case_count": len(data.cases),
                "completed_case_steps": completed_case_steps,
                "total_case_steps": total_case_steps,
                "fraction": completed_case_steps / total_case_steps,
            }
        )

    atomic_rankings: dict[str, dict[str, Any]] = {}
    results_by_profile: dict[str, list[CaseResult]] = {}
    profile_runtime: dict[str, dict[str, float | int]] = {}
    for profile in expanded:
        if profile.family == "union":
            continue
        profile_results: list[CaseResult] = []
        atomic_rankings[profile.name] = {}
        started = time.perf_counter()
        cache_before = cached.cache_size
        for case_index, case in enumerate(data.cases, start=1):
            if profile.family == "keyword":
                scores = score_keyword(data, case)
            elif profile.family == "dense":
                scores = score_dense(
                    data, case, profile, cached, view_budget=view_budget
                )
            elif profile.family == "registry":
                scores = score_registry(
                    data, case, profile, cached, view_budget=view_budget
                )
            else:  # pragma: no cover - guarded by ProfileFamily
                raise ValueError(f"unsupported profile family: {profile.family}")
            ranking = rank_scores(case, scores)
            atomic_rankings[profile.name][case.case_id] = ranking
            profile_results.append(
                _case_result(
                    profile,
                    case,
                    ranking,
                    metadata={"profile_ready": readiness[profile.name]["ready"]},
                )
            )
            completed_case_steps += 1
            report(profile, case_index)
        results_by_profile[profile.name] = profile_results
        profile_runtime[profile.name] = {
            "wall_seconds": time.perf_counter() - started,
            "new_unique_embedding_texts": cached.cache_size - cache_before,
        }

    for profile in expanded:
        if profile.family != "union":
            continue
        profile_results = []
        started = time.perf_counter()
        for case_index, case in enumerate(data.cases, start=1):
            members = {
                name: atomic_rankings[name][case.case_id] for name in profile.union_members
            }
            scores = rank_union(case, members, depth=union_depth)
            ranking = rank_scores(case, scores)
            profile_results.append(
                _case_result(
                    profile,
                    case,
                    ranking,
                    metadata={"union_members": profile.union_members, "union_depth": union_depth},
                )
            )
            completed_case_steps += 1
            report(profile, case_index)
        results_by_profile[profile.name] = profile_results
        profile_runtime[profile.name] = {
            "wall_seconds": time.perf_counter() - started,
            "new_unique_embedding_texts": 0,
        }

    requested_names = set(profile_names)
    requested_results = {
        name: results for name, results in results_by_profile.items() if name in requested_names
    }
    run_metadata = {
        "embedder": cached.name,
        "allow_incomplete": allow_incomplete,
        "union_depth": union_depth,
        "view_budget_per_side": view_budget,
        "requested_profiles": profile_names,
        "expanded_profiles": [profile.name for profile in expanded],
        "readiness": readiness,
        "profile_runtime": profile_runtime,
        "embedding_cache": {
            "unique_texts": cached.cache_size,
            "backend_calls": cached.backend_calls,
            "backend_texts": cached.backend_texts,
        },
    }
    return requested_results, run_metadata


def _jsonable_result(result: CaseResult) -> dict[str, Any]:
    return asdict(result)


def write_run_artifacts(
    output_dir: str | Path,
    *,
    data: ExperimentData,
    results_by_profile: dict[str, list[CaseResult]],
    run_metadata: dict[str, Any],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "relation-vector-lab-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "input_files": data.input_files,
        **run_metadata,
    }
    preflight = build_preflight_report(
        data,
        [DEFAULT_PROFILES[name] for name in run_metadata["expanded_profiles"]],
    )
    metrics = evaluate_results(data.cases, results_by_profile)

    paths = {
        "manifest": output / "run_manifest.json",
        "preflight": output / "preflight_report.json",
        "results": output / "retrieval_results.jsonl",
        "metrics": output / "metrics_summary.json",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["preflight"].write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with paths["results"].open("w", encoding="utf-8") as handle:
        for profile in run_metadata["requested_profiles"]:
            for result in results_by_profile[profile]:
                handle.write(json.dumps(_jsonable_result(result), ensure_ascii=False) + "\n")
    paths["metrics"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths

