"""Audit-aware retrieval metrics and relation-counterfactual diagnostics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from .contracts import CaseResult, QueryCase


AUDIT_MODES = ("strict", "provisional", "assumed_reviewed")


def relevant_ids(case: QueryCase, audit_mode: str) -> set[str]:
    if audit_mode == "strict":
        return set(case.strict_relevant_memory_ids)
    if audit_mode == "provisional":
        return set(case.provisional_relevant_memory_ids)
    if audit_mode == "assumed_reviewed":
        return set(case.assumed_reviewed_relevant_memory_ids)
    raise ValueError(f"unknown audit mode: {audit_mode}")


def _retrieval_metrics(
    cases_by_id: dict[str, QueryCase],
    results: Iterable[CaseResult],
    *,
    audit_mode: str,
    ks: tuple[int, ...],
) -> dict[str, float | int]:
    hits: dict[int, list[float]] = defaultdict(list)
    recalls: dict[int, list[float]] = defaultdict(list)
    reciprocal_ranks: list[float] = []
    scored_candidate_counts: list[int] = []
    candidate_counts: list[int] = []
    evaluable = 0
    for result in results:
        case = cases_by_id[result.case_id]
        relevant = relevant_ids(case, audit_mode)
        if not relevant:
            continue
        evaluable += 1
        # Abstention is not retrieval.  Keeping score=None candidates in Top-k would
        # turn their deterministic ID order into false hits.
        ranking = [
            candidate.memory_id
            for candidate in result.ranked_candidates
            if candidate.score is not None
        ]
        scored_candidate_counts.append(len(ranking))
        candidate_counts.append(len(result.ranked_candidates))
        for k in ks:
            retrieved = set(ranking[:k])
            matched = len(relevant & retrieved)
            hits[k].append(float(matched > 0))
            recalls[k].append(matched / len(relevant))
        first_rank = next(
            (rank for rank, memory_id in enumerate(ranking, start=1) if memory_id in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

    metrics: dict[str, float | int] = {"evaluable_queries": evaluable}
    for k in ks:
        metrics[f"hit@{k}"] = mean(hits[k]) if hits[k] else 0.0
        metrics[f"recall@{k}"] = mean(recalls[k]) if recalls[k] else 0.0
    metrics["mrr"] = mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    metrics["mean_scored_candidates"] = (
        mean(scored_candidate_counts) if scored_candidate_counts else 0.0
    )
    metrics["scored_candidate_rate"] = (
        sum(scored_candidate_counts) / sum(candidate_counts) if candidate_counts else 0.0
    )
    return metrics


def _relation_diagnostics(results: Iterable[CaseResult]) -> dict[str, float | int]:
    margins: list[float] = []
    positive_wins = 0
    scored = 0
    for result in results:
        if result.task_family != "relation_reuse_counterfactual":
            continue
        positive = [
            item.score
            for item in result.ranked_candidates
            if item.candidate_role == "positive" and item.score is not None
        ]
        counterfactuals = [
            item.score
            for item in result.ranked_candidates
            if item.candidate_role
            in {"subject_counterfactual", "condition_counterfactual", "hard_negative"}
            and item.score is not None
        ]
        if not positive or not counterfactuals:
            continue
        margin = max(positive) - max(counterfactuals)
        margins.append(margin)
        positive_wins += margin > 0
        scored += 1
    return {
        "scored_scenarios": scored,
        "positive_above_all_counterfactuals_rate": positive_wins / scored if scored else 0.0,
        "mean_positive_counterfactual_margin": mean(margins) if margins else 0.0,
    }


def _best_relevant_rank(result: CaseResult, relevant: set[str]) -> int:
    fallback = len(result.ranked_candidates) + 1
    return next(
        (
            rank
            for rank, item in enumerate(
                (candidate for candidate in result.ranked_candidates if candidate.score is not None),
                start=1,
            )
            if item.memory_id in relevant
        ),
        fallback,
    )


def _compare_profiles(
    cases_by_id: dict[str, QueryCase],
    baseline: list[CaseResult],
    variant: list[CaseResult],
    *,
    audit_mode: str,
) -> dict[str, float | int]:
    baseline_by_case = {item.case_id: item for item in baseline}
    deltas: list[int] = []
    for result in variant:
        if result.case_id not in baseline_by_case:
            continue
        relevant = relevant_ids(cases_by_id[result.case_id], audit_mode)
        if not relevant:
            continue
        before = _best_relevant_rank(baseline_by_case[result.case_id], relevant)
        after = _best_relevant_rank(result, relevant)
        deltas.append(after - before)
    return {
        "comparable_queries": len(deltas),
        "improved_rate": sum(delta < 0 for delta in deltas) / len(deltas) if deltas else 0.0,
        "unchanged_rate": sum(delta == 0 for delta in deltas) / len(deltas) if deltas else 0.0,
        "worsened_rate": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else 0.0,
        "mean_best_relevant_rank_delta": mean(deltas) if deltas else 0.0,
    }


def _union_diagnostics(
    cases_by_id: dict[str, QueryCase], results: Iterable[CaseResult], *, audit_mode: str
) -> dict[str, int]:
    added_by_route: dict[str, set[tuple[str, str]]] = defaultdict(set)
    union_relevant: set[tuple[str, str]] = set()
    for result in results:
        relevant = relevant_ids(cases_by_id[result.case_id], audit_mode)
        for item in result.ranked_candidates:
            if item.score is None or item.memory_id not in relevant:
                continue
            key = (result.case_id, item.memory_id)
            union_relevant.add(key)
            ranks = item.trace.get("route_ranks", {})
            if "p0_vector" not in ranks:
                for route in ("p2_factor", "p3_registry_auto"):
                    if route in ranks:
                        added_by_route[route].add(key)
    return {
        "union_relevant_pairs": len(union_relevant),
        "factor_relevant_pairs_outside_raw_top_r": len(added_by_route["p2_factor"]),
        "registry_relevant_pairs_outside_raw_top_r": len(
            added_by_route["p3_registry_auto"]
        ),
    }


def evaluate_results(
    cases: list[QueryCase],
    results_by_profile: dict[str, list[CaseResult]],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    cases_by_id = {case.case_id: case for case in cases}
    output: dict[str, object] = {}
    for profile, results in results_by_profile.items():
        slices: dict[str, object] = {}
        for audit_mode in AUDIT_MODES:
            slices[audit_mode] = _retrieval_metrics(
                cases_by_id, results, audit_mode=audit_mode, ks=ks
            )
        for family in sorted({case.task_family for case in cases}):
            family_results = [item for item in results if item.task_family == family]
            slices[f"family:{family}"] = {
                mode: _retrieval_metrics(
                    cases_by_id, family_results, audit_mode=mode, ks=ks
                )
                for mode in AUDIT_MODES
            }
        output[profile] = {
            "retrieval": slices,
            "relation_diagnostics": _relation_diagnostics(results),
        }
        if profile == "p4_union":
            output[profile]["union_diagnostics"] = {
                mode: _union_diagnostics(cases_by_id, results, audit_mode=mode)
                for mode in AUDIT_MODES
            }

    if "p0_vector" in results_by_profile:
        baseline = results_by_profile["p0_vector"]
        output["_comparisons_to_p0_vector"] = {
            profile: {
                mode: _compare_profiles(
                    cases_by_id, baseline, results, audit_mode=mode
                )
                for mode in AUDIT_MODES
            }
            for profile, results in results_by_profile.items()
            if profile != "p0_vector"
        }
    return output

