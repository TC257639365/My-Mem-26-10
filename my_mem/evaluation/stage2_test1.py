"""stage2_test1 的候选级、证据级和任务级评分。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable


def _classification_report(
    expected: list[str],
    predicted: list[str],
    labels: Iterable[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    f1_values = []
    active_f1_values = []
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(expected, predicted))
        fp = sum(gold != label and pred == label for gold, pred in zip(expected, predicted))
        fn = sum(gold == label and pred != label for gold, pred in zip(expected, predicted))
        support = sum(gold == label for gold in expected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_values.append(f1)
        if support or fp:
            active_f1_values.append(f1)
    correct = sum(gold == pred for gold, pred in zip(expected, predicted))
    return {
        "accuracy": correct / len(expected) if expected else 0.0,
        "macro_f1": (
            sum(active_f1_values) / len(active_f1_values)
            if active_f1_values
            else 0.0
        ),
        "macro_f1_all_declared_labels": (
            sum(f1_values) / len(f1_values) if f1_values else 0.0
        ),
        "count": len(expected),
        "per_label": per_label,
        "confusion": dict(Counter(f"{gold}->{pred}" for gold, pred in zip(expected, predicted))),
    }


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        if len(token) > 2
    }


def _lexical_alignment(expected_name: str, actual_text: str) -> float:
    expected = _tokens(expected_name)
    actual = _tokens(actual_text)
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score_stage2_test1(
    gold_doc: dict[str, Any],
    run_doc: dict[str, Any],
    *,
    condition: str | None = None,
) -> dict[str, Any]:
    """对一次固定配置运行评分；不执行 LLM-as-judge。"""

    condition = condition or run_doc.get("config", {}).get("evidence_condition")
    if condition not in {"E0", "Eplus", "Eminus", "Enoise"}:
        raise ValueError("a valid evidence condition is required for scoring")
    raw_results = run_doc.get("results", run_doc if isinstance(run_doc, list) else [])
    result_by_task = {item["case_id"]: item for item in raw_results}
    annotations = gold_doc["annotations"]
    expected_task_ids = {item["task_id"] for item in annotations}
    result_task_ids = [item["case_id"] for item in raw_results]

    expected_potential: list[str] = []
    predicted_potential: list[str] = []
    expected_bridge: list[str] = []
    predicted_bridge: list[str] = []
    expected_validated: list[str] = []
    predicted_validated: list[str] = []
    expected_relation: list[str] = []
    predicted_relation: list[str] = []
    per_candidate: list[dict[str, Any]] = []
    role_totals: Counter[str] = Counter()
    role_state_correct: Counter[str] = Counter()
    role_validated_correct: Counter[str] = Counter()
    query_kind_total = 0
    query_kind_correct = 0
    high_total = high_correct = 0
    high_open_total = high_open_correct = 0
    hard_negative_total = hard_negative_false_positive = 0
    unrelated_total = unrelated_false_positive = 0
    proposal_tp = proposal_fp = proposal_fn = 0
    source_gap_total = source_gap_correct = 0
    unexpected_gap_total = unexpected_gap_count = 0
    evidence_expected_total = evidence_acquired_total = 0
    relation_items = relation_grounded = relation_judged = 0
    citation_total = citation_valid = citation_decisive = decisive_citation_total = 0
    frame_alignment_scores: list[float] = []
    proposal_alignment_scores: list[float] = []
    missing_tasks: list[str] = []
    missing_hypotheses: list[str] = []
    missing_validations: list[str] = []
    missing_challenges: list[str] = []
    extra_candidate_predictions: list[str] = []

    for annotation in annotations:
        task_id = annotation["task_id"]
        result = result_by_task.get(task_id)
        if result is None:
            missing_tasks.append(task_id)
            hypotheses: dict[str, Any] = {}
            validations: dict[str, Any] = {}
            challenges: dict[str, Any] = {}
            evidence_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        else:
            hypotheses = {item["candidate_id"]: item for item in result["hypotheses"]}
            validations = {
                item["candidate_id"]: item for item in result["validated_influences"]
            }
            challenges = {
                item["candidate_id"]: item for item in result["dimension_challenges"]
            }
            expected_candidate_ids = {
                item["candidate_id"] for item in annotation["candidates"]
            }
            for stage_name, values in (
                ("hypothesis", set(hypotheses)),
                ("validation", set(validations)),
                ("challenge", set(challenges)),
            ):
                extra_candidate_predictions.extend(
                    f"{task_id}:{stage_name}:{candidate_id}"
                    for candidate_id in sorted(values - expected_candidate_ids)
                )
            gap_owner = {
                gap["gap_id"]: hypothesis["candidate_id"]
                for hypothesis in result["hypotheses"]
                for gap in hypothesis["gaps"]
            }
            evidence_by_candidate = defaultdict(list)
            for evidence in result["acquired_evidence"]:
                owner = gap_owner.get(evidence["gap_id"])
                if owner:
                    evidence_by_candidate[owner].append(evidence)

            expected_kind = annotation["query_kind"]
            actual_kind = result["task_frame"]["query_kind"]
            query_kind_total += 1
            if expected_kind == "request_mixed":
                query_kind_correct += actual_kind in {"request", "mixed"}
            else:
                query_kind_correct += actual_kind == expected_kind

            actual_dimensions = [
                " ".join(
                    [item["dimension_id"], item["name"], item["description"]]
                )
                for item in result["task_frame"]["common_dimensions"]
            ]
            for expected_dimension in annotation["oracle_common_dimensions"]:
                frame_alignment_scores.append(
                    max(
                        (
                            _lexical_alignment(expected_dimension, text)
                            for text in actual_dimensions
                        ),
                        default=0.0,
                    )
                )

        for gold in annotation["candidates"]:
            candidate_id = gold["candidate_id"]
            role = gold["candidate_role"]
            hypothesis = hypotheses.get(candidate_id)
            validation = validations.get(candidate_id)
            challenge = challenges.get(candidate_id)
            if hypothesis is None:
                missing_hypotheses.append(candidate_id)
            if validation is None:
                missing_validations.append(candidate_id)
            if challenge is None:
                missing_challenges.append(candidate_id)
            candidate_evidence = evidence_by_candidate.get(candidate_id, [])
            gold_potential = gold["potential_impact"]
            gold_bridge = gold["bridge_status_before_acquisition"]
            pred_potential = hypothesis["potential_impact"] if hypothesis else "__missing__"
            pred_bridge = hypothesis["bridge_status"] if hypothesis else "__missing__"
            gold_validated_raw = gold["validated_impact_by_condition"][condition]
            gold_validated = gold_validated_raw if gold_validated_raw is not None else "unresolved"
            if validation is None:
                pred_validated = "__missing__"
            else:
                actual = validation["validated_impact"]
                pred_validated = actual if actual is not None else "unresolved"

            expected_potential.append(gold_potential)
            predicted_potential.append(pred_potential)
            expected_bridge.append(gold_bridge)
            predicted_bridge.append(pred_bridge)
            expected_validated.append(gold_validated)
            predicted_validated.append(pred_validated)
            role_totals[role] += 1
            role_state_correct[role] += (
                pred_potential == gold_potential and pred_bridge == gold_bridge
            )
            role_validated_correct[role] += pred_validated == gold_validated

            if gold_potential == "high":
                high_total += 1
                high_correct += pred_potential == "high"
            if gold_potential == "high" and gold_bridge == "open":
                high_open_total += 1
                high_open_correct += pred_potential == "high" and pred_bridge == "open"
            if role == "D":
                hard_negative_total += 1
                hard_negative_false_positive += pred_potential != "none"
            if role == "F":
                unrelated_total += 1
                unrelated_false_positive += pred_potential != "none"

            expected_proposal = gold["gold_proposed_dimension"] is not None
            actual_proposal = bool(challenge and challenge.get("proposed_dimension"))
            proposal_tp += expected_proposal and actual_proposal
            proposal_fp += not expected_proposal and actual_proposal
            proposal_fn += expected_proposal and not actual_proposal
            if expected_proposal and actual_proposal:
                proposal = challenge["proposed_dimension"]
                proposal_text = " ".join(
                    [
                        proposal["dimension_id"],
                        proposal["name"],
                        proposal["description"],
                        proposal["decision_delta"],
                    ]
                )
                proposal_alignment_scores.append(
                    _lexical_alignment(gold["gold_proposed_dimension"], proposal_text)
                )

            expected_gaps = gold["required_gaps"]
            actual_gaps = hypothesis["gaps"] if hypothesis else []
            if expected_gaps:
                source_gap_total += 1
                expected_sources = {item["source_type"] for item in expected_gaps}
                actual_sources = {item["source_type"] for item in actual_gaps}
                source_gap_correct += bool(expected_sources & actual_sources)
            else:
                unexpected_gap_total += 1
                unexpected_gap_count += bool(actual_gaps)

            expected_evidence_relation_value = gold["evidence_relation_by_condition"][condition]
            if expected_gaps and condition != "E0":
                evidence_expected_total += 1
                evidence_acquired_total += bool(candidate_evidence)
                relations = {item["relation"] for item in candidate_evidence}
                if not relations:
                    predicted = "__missing__"
                elif len(relations) == 1:
                    predicted = next(iter(relations))
                else:
                    predicted = "__mixed__"
                expected_relation.append(expected_evidence_relation_value)
                predicted_relation.append(predicted)
                relation_items += len(candidate_evidence)
                for item in candidate_evidence:
                    if item["relation"] != "unjudged":
                        relation_judged += 1
                        relation_grounded += bool(item["grounding_spans"])

            if validation:
                acquired_ids = {item["evidence_id"] for item in candidate_evidence}
                allowed_ids = acquired_ids | {f"memory:{candidate_id}"}
                for evidence_id in validation["evidence_ids"]:
                    citation_total += 1
                    citation_valid += evidence_id in allowed_ids
                    matching = [
                        item for item in candidate_evidence if item["evidence_id"] == evidence_id
                    ]
                    if matching:
                        decisive_citation_total += 1
                        expected_decisive = (
                            "support" if pred_validated in {"high", "low"} else "refute"
                        )
                        citation_decisive += matching[0]["relation"] == expected_decisive

            per_candidate.append(
                {
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "role": role,
                    "gold": {
                        "potential_impact": gold_potential,
                        "bridge_status": gold_bridge,
                        "validated_impact": gold_validated,
                        "evidence_relation": expected_evidence_relation_value,
                    },
                    "prediction": {
                        "potential_impact": pred_potential,
                        "bridge_status": pred_bridge,
                        "validated_impact": pred_validated,
                        "evidence_relations": sorted(
                            {item["relation"] for item in candidate_evidence}
                        ),
                    },
                    "state_correct": (
                        pred_potential == gold_potential and pred_bridge == gold_bridge
                    ),
                    "validated_correct": pred_validated == gold_validated,
                }
            )

    potential_report = _classification_report(
        expected_potential, predicted_potential, ("high", "low", "none")
    )
    bridge_report = _classification_report(
        expected_bridge, predicted_bridge, ("supported", "open", "refuted")
    )
    validated_report = _classification_report(
        expected_validated, predicted_validated, ("high", "low", "none", "unresolved")
    )
    relation_report = _classification_report(
        expected_relation,
        predicted_relation,
        ("support", "refute", "insufficient"),
    )
    proposal_precision = _safe_div(proposal_tp, proposal_tp + proposal_fp)
    proposal_recall = _safe_div(proposal_tp, proposal_tp + proposal_fn)
    proposal_f1 = (
        2 * proposal_precision * proposal_recall / (proposal_precision + proposal_recall)
        if proposal_precision + proposal_recall
        else 0.0
    )

    return {
        "experiment_id": "stage2_test1",
        "scorer_version": "stage2-test1-scorer-v1",
        "evidence_condition": condition,
        "dataset_version": gold_doc.get("dataset_version"),
        "run_config": run_doc.get("config", {}),
        "run_completeness": {
            "expected_tasks": len(annotations),
            "completed_tasks": len(result_by_task),
            "missing_tasks": missing_tasks,
            "extra_tasks": sorted(set(result_task_ids) - expected_task_ids),
            "duplicate_task_results": sorted(
                task_id
                for task_id, count in Counter(result_task_ids).items()
                if count > 1
            ),
            "failed_cases": run_doc.get("errors", []),
            "missing_hypotheses": missing_hypotheses,
            "missing_validations": missing_validations,
            "missing_challenges": missing_challenges,
            "extra_candidate_predictions": extra_candidate_predictions,
        },
        "potential_impact": potential_report,
        "bridge_status": bridge_report,
        "validated_impact": validated_report,
        "evidence_relation": relation_report,
        "key_metrics": {
            "high_potential_recall": _safe_div(high_correct, high_total),
            "high_open_exact_recall": _safe_div(high_open_correct, high_open_total),
            "hard_negative_false_positive_rate": _safe_div(
                hard_negative_false_positive, hard_negative_total
            ),
            "unrelated_false_positive_rate": _safe_div(
                unrelated_false_positive, unrelated_total
            ),
            "novel_dimension_proposal_precision": proposal_precision,
            "novel_dimension_proposal_recall": proposal_recall,
            "novel_dimension_proposal_f1": proposal_f1,
            "gap_source_type_accuracy": _safe_div(source_gap_correct, source_gap_total),
            "unexpected_gap_rate": _safe_div(
                unexpected_gap_count, unexpected_gap_total
            ),
            "controlled_evidence_acquisition_coverage": _safe_div(
                evidence_acquired_total, evidence_expected_total
            ),
            "relation_judgment_grounding_rate": _safe_div(
                relation_grounded, relation_judged
            ),
            "relation_judgment_coverage": _safe_div(
                relation_judged, relation_items
            ),
            "citation_id_validity": _safe_div(citation_valid, citation_total),
            "decisive_evidence_citation_rate": _safe_div(
                citation_decisive, decisive_citation_total
            ),
            "query_kind_group_accuracy": _safe_div(query_kind_correct, query_kind_total),
        },
        "role_breakdown": {
            role: {
                "count": role_totals[role],
                "state_exact_accuracy": _safe_div(
                    role_state_correct[role], role_totals[role]
                ),
                "validated_accuracy": _safe_div(
                    role_validated_correct[role], role_totals[role]
                ),
            }
            for role in "ABCDEF"
        },
        "semantic_alignment_diagnostics": {
            "method": "token-overlap diagnostic only; not a substitute for semantic adjudication",
            "oracle_common_dimension_mean_token_recall": (
                sum(frame_alignment_scores) / len(frame_alignment_scores)
                if frame_alignment_scores
                else 0.0
            ),
            "proposed_dimension_mean_token_recall": (
                sum(proposal_alignment_scores) / len(proposal_alignment_scores)
                if proposal_alignment_scores
                else 0.0
            ),
        },
        "per_candidate": per_candidate,
    }
