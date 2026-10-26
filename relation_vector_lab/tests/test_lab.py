from __future__ import annotations

from relation_vector_lab.embeddings import HashingEmbedder
from pathlib import Path

from relation_vector_lab.datasets import (
    default_data_paths,
    load_experiment_data,
    load_frozen_experiment_data,
)
from relation_vector_lab.evaluation import evaluate_results
from relation_vector_lab.preflight import build_preflight_report
from relation_vector_lab.profiles import DEFAULT_PROFILES, get_profiles
from relation_vector_lab.runner import run_profiles
from relation_vector_lab.scoring import bm25_scores
from relation_vector_lab.text import multilingual_tokens


def test_multilingual_tokens_keep_chinese() -> None:
    tokens = multilingual_tokens("我想学习吉他")
    assert "c1:我" in tokens
    assert "c2:我想" in tokens


def test_bm25_prefers_matching_document() -> None:
    scores = bm25_scores("安静的乐器", ["安静地学习乐器", "今天去跑步"])
    assert scores[0] > scores[1]


def test_hashing_embedder_is_normalized_and_repeatable() -> None:
    embedder = HashingEmbedder(dimensions=64)
    first = embedder.embed(["same text", "另一个文本"])
    second = embedder.embed(["same text", "另一个文本"])
    assert first.shape == (2, 64)
    assert (first == second).all()
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-6


def test_profile_ladder_is_registered() -> None:
    required = {
        "p0_keyword",
        "p0_vector",
        "p1_rewrite",
        "p2_factor",
        "p3_registry_auto",
        "p4_union",
    }
    assert required <= set(DEFAULT_PROFILES)
    assert [item.name for item in get_profiles(["p0_vector", "p2_factor"])] == [
        "p0_vector",
        "p2_factor",
    ]


def _load_default_data():
    return load_experiment_data(**default_data_paths())


def test_auxiliary_v2_is_fully_loaded_and_ready() -> None:
    data = _load_default_data()
    profiles = get_profiles(
        ["p1_rewrite", "p1_query_expansion", "p2_factor", "p3_registry_auto", "p4_union"]
    )
    report = build_preflight_report(data, profiles)
    assert report["auxiliary_coverage"]["memory_representation_records"] == 405
    assert report["auxiliary_coverage"]["query_representation_records"] == 80
    assert report["auxiliary_coverage"]["legacy_mixed_condition_binding_count"] == 0
    assert all(item["ready"] for item in report["profiles"].values())
    assert report["data_warnings"] == []


def test_incremental_profiles_run_end_to_end_with_boundary_trace() -> None:
    data = _load_default_data()
    names = [
        "p0_vector",
        "p1_rewrite",
        "p1_query_expansion",
        "p2_factor",
        "p3_registry_auto",
        "p3_registry_oracle_m",
        "p3_registry_oracle_q",
        "p3_registry_oracle_both",
        "p3_registry_swap",
        "p3_registry_shuffle",
        "p4_union",
    ]
    results, metadata = run_profiles(data, names, embedder=HashingEmbedder(dimensions=64))
    assert set(results) == set(names)
    assert all(len(items) == 80 for items in results.values())
    assert all(item["ready"] for item in metadata["readiness"].values())
    assert metadata["view_budget_per_side"] == 1
    traces = [
        candidate.trace
        for result in results["p3_registry_auto"]
        for candidate in result.ranked_candidates
        if candidate.score is not None
    ]
    assert traces
    assert any("join_constraints_pending" in trace for trace in traces)
    assert any("memory_audit_status" in trace for trace in traces)


def test_abstained_candidates_do_not_count_as_retrieved() -> None:
    data = _load_default_data()
    results, _ = run_profiles(
        data,
        ["p1_prospective_trigger"],
        embedder=HashingEmbedder(dimensions=32),
        allow_incomplete=True,
    )
    metrics = evaluate_results(data.cases, results)
    strict = metrics["p1_prospective_trigger"]["retrieval"]["strict"]
    assert strict["scored_candidate_rate"] == 0.0
    assert strict["hit@1"] == 0.0
    assert strict["hit@5"] == 0.0


def test_partitioned_frozen_dataset_runs_with_progress() -> None:
    project = Path(__file__).resolve().parents[2]
    dataset = project / "results" / "stage3_test1" / "My-Mem-test" / "dataset"
    data = load_frozen_experiment_data(dataset)
    events = []
    results, metadata = run_profiles(
        data,
        ["p2_factor"],
        embedder=HashingEmbedder(dimensions=32),
        view_budget=1,
        progress_callback=events.append,
    )
    assert len(results["p2_factor"]) == 80
    assert metadata["readiness"]["p2_factor"]["ready"] is True
    assert len(events) == 80
    assert events[-1]["fraction"] == 1.0
    scored = next(
        candidate
        for result in results["p2_factor"]
        for candidate in result.ranked_candidates
        if candidate.score is not None
    )
    assert scored.trace["memory_view_id"]
    assert scored.trace["query_view_id"]

