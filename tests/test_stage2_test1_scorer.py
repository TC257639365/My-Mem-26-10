import json
import unittest
from pathlib import Path

from my_mem.evaluation import score_stage2_test1


DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "ProcessedDataset"
    / "stage2_influence_pilot_12"
)


def _perfect_run(gold):
    results = []
    for annotation in gold["annotations"]:
        challenges = []
        hypotheses = []
        evidence = []
        validated = []
        for item in annotation["candidates"]:
            candidate_id = item["candidate_id"]
            proposal = None
            if item["gold_proposed_dimension"]:
                name = item["gold_proposed_dimension"]
                proposal = {
                    "dimension_id": name,
                    "name": name,
                    "description": name,
                    "scope": "core",
                    "priority": 1.0,
                    "memory_grounding": "fixture",
                    "query_bridge": "fixture",
                    "decision_delta": "fixture",
                }
            challenges.append(
                {
                    "candidate_id": candidate_id,
                    "matched_dimension_ids": [],
                    "proposed_dimension": proposal,
                    "rationale": "fixture",
                }
            )
            gaps = []
            for gap_index, gold_gap in enumerate(item["required_gaps"], start=1):
                generated_gap_id = f"{candidate_id}:generated:{gap_index}"
                gaps.append(
                    {
                        "gap_id": generated_gap_id,
                        "proposition": gold_gap["proposition"],
                        "source_type": gold_gap["source_type"],
                        "retrieval_query": "fixture",
                        "decision_relevance": "fixture",
                        "priority": 1.0,
                        "status": "open",
                    }
                )
                evidence.append(
                    {
                        "evidence_id": f"{generated_gap_id}:doc",
                        "gap_id": generated_gap_id,
                        "document_id": "doc",
                        "content": "supporting fixture",
                        "source_type": gold_gap["source_type"],
                        "provenance": "fixture",
                        "relation": "support",
                        "relation_confidence": 1.0,
                        "relation_rationale": "fixture",
                        "grounding_spans": ["supporting fixture"],
                        "retrieval_score": 1.0,
                    }
                )
            hypotheses.append(
                {
                    "candidate_id": candidate_id,
                    "potential_impact": item["potential_impact"],
                    "bridge_status": item["bridge_status_before_acquisition"],
                    "gaps": gaps,
                }
            )
            validated_impact = item["validated_impact_by_condition"]["Eplus"]
            validated.append(
                {
                    "candidate_id": candidate_id,
                    "potential_impact": item["potential_impact"],
                    "validated_impact": validated_impact,
                    "evidence_ids": (
                        [f"{gaps[0]['gap_id']}:doc"]
                        if gaps
                        else (
                            [f"memory:{candidate_id}"]
                            if validated_impact in {"high", "low"}
                            else []
                        )
                    ),
                }
            )
        common_dimensions = [
            {
                "dimension_id": name,
                "name": name,
                "description": name,
            }
            for name in annotation["oracle_common_dimensions"]
        ]
        query_kind = annotation["query_kind"]
        if query_kind == "request_mixed":
            query_kind = "mixed"
        results.append(
            {
                "case_id": annotation["task_id"],
                "task_frame": {
                    "query_kind": query_kind,
                    "common_dimensions": common_dimensions,
                },
                "dimension_challenges": challenges,
                "hypotheses": hypotheses,
                "acquired_evidence": evidence,
                "validated_influences": validated,
            }
        )
    return {
        "config": {"evidence_condition": "Eplus"},
        "results": results,
        "errors": [],
    }


class Stage2Test1ScorerTests(unittest.TestCase):
    def test_perfect_fixture_scores_one_on_core_metrics(self):
        gold = json.loads((DATASET_DIR / "gold.json").read_text(encoding="utf-8"))
        report = score_stage2_test1(gold, _perfect_run(gold))
        self.assertEqual(report["potential_impact"]["macro_f1"], 1.0)
        self.assertEqual(report["bridge_status"]["macro_f1"], 1.0)
        self.assertEqual(report["validated_impact"]["macro_f1"], 1.0)
        self.assertEqual(report["evidence_relation"]["macro_f1"], 1.0)
        self.assertEqual(report["key_metrics"]["high_open_exact_recall"], 1.0)
        self.assertEqual(report["key_metrics"]["hard_negative_false_positive_rate"], 0.0)
        self.assertEqual(report["key_metrics"]["controlled_evidence_acquisition_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
