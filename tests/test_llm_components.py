import asyncio
import json
import unittest

from my_mem.influence.llm_components import (
    LLMDimensionChallengeBuilder,
    LLMEvidenceRelationJudge,
    LLMHypothesisBuilder,
    LLMInfluenceValidator,
    LLMTaskFrameBuilder,
)
from my_mem.schemas import (
    AcquiredEvidence,
    CandidateDimensionChallenge,
    CandidateMemory,
    EvidenceGap,
    InfluenceHypothesis,
    TaskFrame,
)


class SequencedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return json.dumps(next(self.responses))


class LLMComponentTests(unittest.TestCase):
    def _task_frame(self):
        return TaskFrame(
            query_kind="question",
            focal_response_goal="decide whether to buy the plant",
            response_obligations=[
                {
                    "obligation_id": "answer",
                    "description": "give a recommendation",
                    "priority": 1.0,
                    "rationale": "the task asks whether to buy",
                }
            ],
            common_dimensions=[
                {
                    "dimension_id": "placement",
                    "name": "placement feasibility",
                    "description": "whether the plant can be placed safely",
                    "scope": "core",
                    "priority": 0.8,
                }
            ],
            open_dimensions=["candidate-grounded safety constraints"],
            baseline_response="give a recommendation from task facts",
        )

    def _challenge(self):
        return CandidateDimensionChallenge(
            candidate_id="m1",
            matched_dimension_ids=["placement"],
            rationale="The dog's behavior may affect safe placement.",
        )

    def test_dimension_challenge_precedes_impact_classification(self):
        response = {
            "dimension_challenges": [
                {
                    "candidate_id": "m1",
                    "matched_dimension_ids": [],
                    "proposed_dimension": {
                        "dimension_id": "novel:m1:ingestion",
                        "name": "ingestion safety",
                        "description": "whether the dog could ingest the plant",
                        "scope": "core",
                        "priority": 0.9,
                        "memory_grounding": "The dog chews household objects.",
                        "query_bridge": "The new plant would be a reachable object.",
                        "decision_delta": "Avoid or isolate an unsafe plant.",
                    },
                    "rationale": "F0 omitted candidate-grounded ingestion risk.",
                }
            ]
        }
        provider = SequencedProvider([response])
        result = asyncio.run(
            LLMDimensionChallengeBuilder(provider).build(
                "Should I buy this plant?",
                self._task_frame(),
                [CandidateMemory(memory_id="m1", content="The dog chews things")],
            )
        )
        self.assertEqual(result[0].proposed_dimension.dimension_id, "novel:m1:ingestion")
        self.assertIn("before assigning any influence level", provider.calls[0][0])
        self.assertEqual(
            provider.calls[0][1]["call_site"], "influence.dimension_challenge"
        )

    def test_hypothesis_reasks_and_keeps_high_impact_with_open_bridge(self):
        invalid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "potential_impact": "high",
                    "target_dimension_ids": ["placement"],
                    "bridge_status": "open",
                    "bridge_confidence": "medium",
                    "possible_action_change": "place the plant out of reach",
                    "bridge_steps": [],
                    "gaps": [],
                }
            ]
        }
        valid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "potential_impact": "high",
                    "effect_target": "safe placement",
                    "target_dimension_ids": ["placement"],
                    "mechanism": ["dog may ingest a reachable plant"],
                    "bridge_steps": [
                        {
                            "claim": "The dog may reach this plant.",
                            "basis": "gap",
                            "confidence": 0.3,
                            "gap_id": "g1",
                        }
                    ],
                    "task_supported_claims": ["The user is considering the plant."],
                    "memory_supported_claims": ["The dog chews things."],
                    "inferred_claims": ["The dog could chew a reachable plant."],
                    "gaps": [
                        {
                            "gap_id": "g1",
                            "proposition": "The plant would be reachable by the dog.",
                            "source_type": "environment_tool",
                            "retrieval_query": "plant placement dog reachability",
                            "decision_relevance": "determines whether ingestion risk applies",
                            "priority": 0.8,
                            "status": "open",
                        }
                    ],
                    "possible_action_change": "avoid or isolate the plant",
                    "bridge_status": "open",
                    "bridge_confidence": 0.3,
                }
            ]
        }
        provider = SequencedProvider([invalid, valid])
        builder = LLMHypothesisBuilder(provider)
        result = asyncio.run(
            builder.build(
                "Should I buy this plant?",
                self._task_frame(),
                [CandidateMemory(memory_id="m1", content="The dog chews things")],
                [self._challenge()],
            )
        )
        self.assertEqual(result[0].potential_impact, "high")
        self.assertEqual(result[0].bridge_status, "open")
        self.assertEqual(result[0].bridge_confidence, 0.3)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("Keep conditional impact separate", provider.calls[0][0])
        self.assertIn("failed schema validation", provider.calls[1][0])

    def test_builds_statement_frame_with_open_dimensions_and_skeleton(self):
        response = {
            "task_frame": {
                "query_kind": "statement",
                "explicit_facts": ["The user slept poorly."],
                "focal_response_goal": "respond to the user's current experience",
                "response_obligations": [
                    {
                        "obligation_id": "acknowledge",
                        "description": "acknowledge the difficult experience",
                        "priority": 0.9,
                        "rationale": "the user disclosed a current difficulty",
                    }
                ],
                "common_dimensions": [
                    {
                        "dimension_id": "wellbeing",
                        "name": "current wellbeing",
                        "description": "severity and immediate implications",
                        "scope": "core",
                        "priority": 0.9,
                    }
                ],
                "open_dimensions": ["unmentioned personal constraints"],
                "baseline_response": "acknowledge, then offer proportionate support",
            }
        }
        provider = SequencedProvider([response])
        frame = asyncio.run(LLMTaskFrameBuilder(provider).build("I barely slept."))
        self.assertEqual(frame.query_kind, "statement")
        self.assertEqual(frame.open_dimensions, ["unmentioned personal constraints"])
        self.assertIn("response skeleton", provider.calls[0][0])

    def test_evidence_relation_judge_records_grounded_assessment(self):
        response = {
            "assessment": {
                "evidence_id": "g1:d1",
                "gap_id": "g1",
                "relation": "support",
                "rationale": "The evidence directly states the queried link.",
                "grounding_spans": ["applies to this project"],
                "confidence": 0.95,
            }
        }
        provider = SequencedProvider([response])
        gap = EvidenceGap(
            gap_id="g1",
            proposition="The rule applies to this project.",
            source_type="external_knowledge",
            retrieval_query="project rule",
            decision_relevance="changes the workload advice",
        )
        evidence = AcquiredEvidence(
            evidence_id="g1:d1",
            gap_id="g1",
            document_id="d1",
            content="The current handbook says the rule applies to this project.",
            source_type="external_knowledge",
            provenance="fixture:d1",
        )
        judged = asyncio.run(LLMEvidenceRelationJudge(provider).judge(gap, evidence))
        self.assertEqual(judged.relation, "support")
        self.assertEqual(judged.relation_confidence, 0.95)
        self.assertEqual(judged.grounding_spans, ["applies to this project"])
        self.assertEqual(provider.calls[0][1]["call_site"], "evidence.relation")

    def test_open_bridge_validation_must_cite_supporting_evidence(self):
        response = {
            "result": {
                "candidate_id": "m1",
                "status": "validated",
                "potential_impact": "high",
                "validated_impact": "high",
                "conclusion": "The bridge is supported.",
                "evidence_ids": ["g1:d1"],
                "unresolved_gap_ids": [],
                "action_implication": "Avoid the unsafe option.",
                "validation_confidence": 0.95,
            }
        }
        hypothesis = InfluenceHypothesis(
            candidate_id="m1",
            potential_impact="high",
            target_dimension_ids=["placement"],
            bridge_steps=[
                {
                    "claim": "The rule applies.",
                    "basis": "gap",
                    "confidence": 0.4,
                    "gap_id": "g1",
                }
            ],
            gaps=[
                {
                    "gap_id": "g1",
                    "proposition": "The rule applies.",
                    "source_type": "external_knowledge",
                    "retrieval_query": "rule applies",
                    "decision_relevance": "changes action",
                }
            ],
            possible_action_change="Avoid the unsafe option.",
            bridge_status="open",
            bridge_confidence=0.4,
        )
        evidence = AcquiredEvidence(
            evidence_id="g1:d1",
            gap_id="g1",
            document_id="d1",
            content="The current rule applies.",
            source_type="external_knowledge",
            provenance="fixture:d1",
            relation="support",
            relation_confidence=0.95,
            relation_rationale="Direct statement.",
            grounding_spans=["rule applies"],
        )
        validator = LLMInfluenceValidator(SequencedProvider([response]))
        result = asyncio.run(
            validator.validate(
                "Should I act?",
                CandidateMemory(memory_id="m1", content="Relevant memory."),
                hypothesis,
                [evidence],
            )
        )
        self.assertEqual(result.validated_impact, "high")

    def test_open_bridge_validation_rejects_non_support_citation(self):
        response = {
            "result": {
                "candidate_id": "m1",
                "status": "validated",
                "potential_impact": "high",
                "validated_impact": "high",
                "conclusion": "Incorrectly validated.",
                "evidence_ids": ["g1:d1"],
                "unresolved_gap_ids": [],
                "validation_confidence": 0.8,
            }
        }
        hypothesis = InfluenceHypothesis(
            candidate_id="m1",
            potential_impact="high",
            target_dimension_ids=["placement"],
            bridge_steps=[
                {
                    "claim": "The rule applies.",
                    "basis": "gap",
                    "confidence": 0.4,
                    "gap_id": "g1",
                }
            ],
            gaps=[
                {
                    "gap_id": "g1",
                    "proposition": "The rule applies.",
                    "source_type": "external_knowledge",
                    "retrieval_query": "rule applies",
                    "decision_relevance": "changes action",
                }
            ],
            possible_action_change="Avoid the unsafe option.",
            bridge_status="open",
            bridge_confidence=0.4,
        )
        evidence = AcquiredEvidence(
            evidence_id="g1:d1",
            gap_id="g1",
            document_id="d1",
            content="The page mentions a rule but not whether it applies.",
            source_type="external_knowledge",
            provenance="fixture:d1",
            relation="insufficient",
            relation_confidence=0.8,
            relation_rationale="Applicability is absent.",
        )
        validator = LLMInfluenceValidator(SequencedProvider([response]))
        with self.assertRaises(ValueError):
            asyncio.run(
                validator.validate(
                    "Should I act?",
                    CandidateMemory(memory_id="m1", content="Relevant memory."),
                    hypothesis,
                    [evidence],
                )
            )


if __name__ == "__main__":
    unittest.main()
