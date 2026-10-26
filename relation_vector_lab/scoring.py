"""Atomic lexical, dense, registry and rank-union scorers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from .contracts import (
    CandidateRecord,
    CandidateScore,
    ExperimentData,
    QueryCase,
    RelationBinding,
    TextView,
)
from .embeddings import CachedEmbedder
from .profiles import ProfileSpec
from .text import multilingual_tokens, serialize_view


def _embedding_text(view: TextView) -> str:
    if view.kind in {"raw", "subject", "rewrite", "prospective_trigger", "query_expansion"}:
        return view.text
    return serialize_view(view.kind, view.text)


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    tokenized = [multilingual_tokens(document) for document in documents]
    query_tokens = multilingual_tokens(query)
    if not documents:
        return []
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores: list[float] = []
    total_docs = len(documents)
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_norm = len(tokens) / average_length if average_length else 0.0
        score = 0.0
        for token in query_tokens:
            df = document_frequency.get(token, 0)
            if not df:
                continue
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            tf = frequencies[token]
            denominator = tf + k1 * (1.0 - b + b * length_norm)
            if denominator:
                score += idf * tf * (k1 + 1.0) / denominator
        scores.append(float(score))
    return scores


def score_keyword(data: ExperimentData, case: QueryCase) -> list[CandidateScore]:
    documents = [data.memories[candidate.memory_id].content for candidate in case.candidates]
    scores = bm25_scores(case.query, documents)
    return [
        CandidateScore(
            memory_id=candidate.memory_id,
            score=score,
            route="keyword",
            trace={"query_view": "raw", "memory_view": "raw"},
        )
        for candidate, score in zip(case.candidates, scores)
    ]


def _max_cosine(
    embedder: CachedEmbedder,
    memory_views: list[TextView],
    query_views: list[TextView],
) -> tuple[float | None, dict[str, object]]:
    if not memory_views or not query_views:
        return None, {
            "abstain_reason": "missing_memory_view" if not memory_views else "missing_query_view"
        }
    memory_texts = [_embedding_text(view) for view in memory_views]
    query_texts = [_embedding_text(view) for view in query_views]
    memory_vectors = embedder.embed(memory_texts)
    query_vectors = embedder.embed(query_texts)
    matrix = np.matmul(memory_vectors, query_vectors.T)
    flat_index = int(np.argmax(matrix))
    memory_index, query_index = np.unravel_index(flat_index, matrix.shape)
    return float(matrix[memory_index, query_index]), {
        "memory_view_kind": memory_views[memory_index].kind,
        "memory_view_id": memory_views[memory_index].provenance.get("view_id"),
        "memory_view_text": memory_views[memory_index].text,
        "query_view_kind": query_views[query_index].kind,
        "query_view_id": query_views[query_index].provenance.get("view_id"),
        "query_view_text": query_views[query_index].text,
        "aggregation": "max",
    }


def score_dense(
    data: ExperimentData,
    case: QueryCase,
    profile: ProfileSpec,
    embedder: CachedEmbedder,
    view_budget: int | None = 1,
) -> list[CandidateScore]:
    query_views = data.views("query", case.query_id, profile.query_view_kinds)
    if view_budget is not None:
        query_views = query_views[:view_budget]
    results: list[CandidateScore] = []
    for candidate in case.candidates:
        memory_views = data.views("memory", candidate.memory_id, profile.memory_view_kinds)
        if view_budget is not None:
            memory_views = memory_views[:view_budget]
        score, trace = _max_cosine(embedder, memory_views, query_views)
        trace["view_budget_per_side"] = view_budget
        results.append(CandidateScore(candidate.memory_id, score, profile.name, trace))
    return results


def _binding_text(binding: RelationBinding, *, memory_side: bool) -> str:
    fields = [binding.role_text]
    if memory_side:
        fields.extend(binding.source_assertions)
    else:
        fields.extend(binding.target_requirements)
        fields.extend(binding.hard_constraints)
        fields.extend(binding.open_conditions)
    return " | ".join(field.strip() for field in fields if field and field.strip())


def _fallback_text(data: ExperimentData, owner_id: str, *, memory_side: bool) -> str | None:
    kinds = ("source_role", "source_factor", "raw") if memory_side else (
        "target_role",
        "query_requirement",
        "raw",
    )
    views = data.views("memory" if memory_side else "query", owner_id, kinds)
    return views[0].text if views else None


def _pattern_ids_for_memory(
    candidate: CandidateRecord,
    bindings: list[RelationBinding],
    mode: str,
) -> set[str]:
    automatic = {binding.pattern_id for binding in bindings}
    if mode in {"oracle_memory", "oracle_both"} and candidate.relation_pattern_id:
        return {candidate.relation_pattern_id}
    return automatic


def _pattern_ids_for_query(case: QueryCase, bindings: list[RelationBinding], mode: str) -> set[str]:
    automatic = {binding.pattern_id for binding in bindings}
    if mode in {"oracle_query", "oracle_both"} and case.pattern_id:
        return {case.pattern_id}
    return automatic


def _descriptor_lookup(data: ExperimentData, *, shuffled: bool) -> dict[str, str]:
    ids = sorted(data.patterns)
    if not shuffled or len(ids) < 2:
        return {pattern_id: pattern_id for pattern_id in ids}
    shifted = ids[1:] + ids[:1]
    return dict(zip(ids, shifted))


def _cosine_pair(embedder: CachedEmbedder, left: str, right: str) -> float:
    vectors = embedder.embed([left, right])
    return float(np.dot(vectors[0], vectors[1]))


def score_registry(
    data: ExperimentData,
    case: QueryCase,
    profile: ProfileSpec,
    embedder: CachedEmbedder,
    view_budget: int | None = 1,
) -> list[CandidateScore]:
    query_bindings = data.query_bindings.get(case.query_id, [])
    if view_budget is not None and profile.registry_mode not in {"oracle_query", "oracle_both"}:
        query_bindings = query_bindings[:view_budget]
    query_ids = _pattern_ids_for_query(case, query_bindings, profile.registry_mode)
    descriptor_lookup = _descriptor_lookup(
        data, shuffled=profile.registry_shuffle_descriptors
    )
    results: list[CandidateScore] = []

    for candidate in case.candidates:
        memory_bindings = data.memory_bindings.get(candidate.memory_id, [])
        if view_budget is not None and profile.registry_mode not in {
            "oracle_memory",
            "oracle_both",
        }:
            memory_bindings = memory_bindings[:view_budget]
        memory_ids = _pattern_ids_for_memory(candidate, memory_bindings, profile.registry_mode)
        shared = sorted(memory_ids & query_ids & set(data.patterns))
        if not shared:
            results.append(
                CandidateScore(
                    candidate.memory_id,
                    None,
                    profile.name,
                    {
                        "abstain_reason": "no_shared_pattern",
                        "memory_pattern_ids": sorted(memory_ids),
                        "query_pattern_ids": sorted(query_ids),
                        "view_budget_per_side": view_budget,
                    },
                )
            )
            continue

        best_score: float | None = None
        best_trace: dict[str, object] = {}
        for pattern_id in shared:
            descriptor_id = descriptor_lookup[pattern_id]
            pattern = data.patterns[descriptor_id]
            memory_binding = next(
                (item for item in memory_bindings if item.pattern_id == pattern_id), None
            )
            query_binding = next(
                (item for item in query_bindings if item.pattern_id == pattern_id), None
            )
            memory_text = (
                _binding_text(memory_binding, memory_side=True)
                if memory_binding
                else _fallback_text(data, candidate.memory_id, memory_side=True)
            )
            query_text = (
                _binding_text(query_binding, memory_side=False)
                if query_binding
                else _fallback_text(data, case.query_id, memory_side=False)
            )
            if not memory_text or not query_text:
                continue
            source_pattern = (
                pattern.target_pattern
                if profile.registry_swap_roles
                else pattern.source_pattern
            )
            target_pattern = (
                pattern.source_pattern
                if profile.registry_swap_roles
                else pattern.target_pattern
            )
            source_fit = _cosine_pair(embedder, memory_text, source_pattern)
            target_fit = _cosine_pair(embedder, query_text, target_pattern)
            score = min(source_fit, target_fit)
            if best_score is None or score > best_score:
                best_score = score
                best_trace = {
                    "matched_pattern_id": pattern_id,
                    "descriptor_pattern_id": descriptor_id,
                    "source_fit": source_fit,
                    "target_fit": target_fit,
                    "aggregation": "min_then_max",
                    "memory_pattern_ids": sorted(memory_ids),
                    "query_pattern_ids": sorted(query_ids),
                    "oracle_mode": profile.registry_mode,
                    "swap_roles": profile.registry_swap_roles,
                    "shuffled_descriptors": profile.registry_shuffle_descriptors,
                    "legacy_mixed_memory_conditions": bool(
                        memory_binding and memory_binding.legacy_mixed_conditions
                    ),
                    "source_assertions": list(memory_binding.source_assertions)
                    if memory_binding
                    else [],
                    "target_requirements": list(query_binding.target_requirements)
                    if query_binding
                    else [],
                    "hard_constraints": list(query_binding.hard_constraints)
                    if query_binding
                    else [],
                    "open_conditions": list(query_binding.open_conditions)
                    if query_binding
                    else [],
                    "entity_scope": memory_binding.entity_scope if memory_binding else {},
                    "temporal_scope": memory_binding.temporal_scope
                    if memory_binding
                    else {},
                    "join_constraints_pending": list(pattern.join_constraints),
                    "memory_audit_status": memory_binding.audit_status
                    if memory_binding
                    else None,
                    "query_audit_status": query_binding.audit_status
                    if query_binding
                    else None,
                    "view_budget_per_side": view_budget,
                }
        if best_score is None:
            best_trace = {
                "abstain_reason": "missing_role_text",
                "shared_pattern_ids": shared,
                "view_budget_per_side": view_budget,
            }
        results.append(CandidateScore(candidate.memory_id, best_score, profile.name, best_trace))
    return results


def rank_scores(
    case: QueryCase,
    scores: Iterable[CandidateScore],
) -> list[tuple[CandidateRecord, CandidateScore]]:
    score_by_memory = {item.memory_id: item for item in scores}
    pairs = [
        (
            candidate,
            score_by_memory.get(
                candidate.memory_id,
                CandidateScore(
                    candidate.memory_id,
                    None,
                    "missing",
                    {"abstain_reason": "no_score"},
                ),
            ),
        )
        for candidate in case.candidates
    ]
    return sorted(
        pairs,
        key=lambda pair: (
            pair[1].score is None,
            -(pair[1].score if pair[1].score is not None else 0.0),
            pair[0].memory_id,
        ),
    )


def rank_union(
    case: QueryCase,
    member_rankings: dict[str, list[tuple[CandidateRecord, CandidateScore]]],
    *,
    depth: int,
) -> list[CandidateScore]:
    route_ranks: dict[str, dict[str, int]] = {}
    for route, ranking in member_rankings.items():
        route_ranks[route] = {
            candidate.memory_id: rank
            for rank, (candidate, score) in enumerate(ranking, start=1)
            if rank <= depth and score.score is not None
        }
    results: list[CandidateScore] = []
    for candidate in case.candidates:
        ranks = {
            route: values[candidate.memory_id]
            for route, values in route_ranks.items()
            if candidate.memory_id in values
        }
        score = max((1.0 / rank for rank in ranks.values()), default=None)
        results.append(
            CandidateScore(
                candidate.memory_id,
                score,
                "p4_union",
                {"route_ranks": ranks, "union_depth": depth, "fusion": "best_reciprocal_rank"},
            )
        )
    return results

