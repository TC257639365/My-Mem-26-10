import asyncio
import json
import unittest

from my_mem.influence.llm_components import LLMHypothesisBuilder, LLMTaskFrameBuilder
from my_mem.schemas import CandidateMemory, TaskFrame


class SequencedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return json.dumps(next(self.responses))


class LLMHypothesisBuilderTests(unittest.TestCase):
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
            baseline_response="recommend from task-only facts",
        )

    def test_reasks_when_json_has_wrong_field_types(self):
        invalid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "influence_level": "high",
                    "mechanism": "dog may damage the plant",
                    "possible_action_change": "place the plant out of reach",
                    "gaps": [
                        {
                            "gap_id": "g1",
                            "proposition": "the plant is fragile",
                            "source_type": "external_knowledge",
                            "retrieval_query": "plant fragility",
                            "decision_relevance": "changes placement",
                            "priority": "high",
                            "status": "open",
                        }
                    ],
                    "impact": "high",
                    "bridge_confidence": "medium",
                }
            ]
        }
        valid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "influence_level": "high",
                    "matched_dimension_ids": ["placement"],
                    "mechanism": ["dog may damage the plant"],
                    "bridge_steps": [
                        {
                            "claim": "A chewing dog may reach and damage the plant.",
                            "basis": "inference",
                            "confidence": 0.5,
                        }
                    ],
                    "task_supported_claims": ["The user is considering buying the plant."],
                    "memory_supported_claims": ["The dog chews things."],
                    "inferred_claims": ["The dog may chew this plant."],
                    "possible_action_change": "place the plant out of reach",
                    "gaps": [
                        {
                            "gap_id": "g1",
                            "proposition": "the plant is fragile",
                            "source_type": "external_knowledge",
                            "retrieval_query": "plant fragility",
                            "decision_relevance": "changes placement",
                            "priority": 0.8,
                            "status": "open",
                        }
                    ],
                    "impact": 0.8,
                    "bridge_confidence": 0.5,
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
            )
        )

        self.assertEqual(result[0].mechanism, ["dog may damage the plant"])
        self.assertEqual(result[0].impact, 0.8)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("Required JSON Schema", provider.calls[0][0])
        self.assertIn("high: the memory supplies a constraint", provider.calls[0][0])
        self.assertIn("failed schema validation", provider.calls[1][0])
        self.assertEqual(
            provider.calls[1][1]["call_site"], "influence.hypothesis"
        )

    def test_reasks_when_model_validator_error_contains_value_error_context(self):
        invalid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "influence_level": "none",
                    "matched_dimension_ids": ["placement"],
                    "bridge_steps": [
                        {
                            "claim": "The memory mentions a dog.",
                            "basis": "memory",
                            "confidence": 0.9,
                        }
                    ],
                    "impact": 0.05,
                    "bridge_confidence": 0.9,
                }
            ]
        }
        valid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "influence_level": "none",
                    "matched_dimension_ids": ["placement"],
                    "bridge_steps": [],
                    "impact": 0.05,
                    "bridge_confidence": 0.9,
                }
            ]
        }
        provider = SequencedProvider([invalid, valid])
        builder = LLMHypothesisBuilder(provider)

        result = asyncio.run(
            builder.build(
                "Should I buy this plant?",
                self._task_frame(),
                [CandidateMemory(memory_id="m1", content="The dog is asleep")],
            )
        )

        self.assertEqual(result[0].influence_level, "none")
        self.assertEqual(result[0].bridge_steps, [])
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("none influence cannot claim", provider.calls[1][0])

    def test_builds_statement_frame_with_prioritized_response_obligations(self):
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
                    },
                    {
                        "obligation_id": "advice",
                        "description": "offer proportionate next steps",
                        "priority": 0.6,
                        "rationale": "practical support may help",
                    },
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
                "baseline_response": "acknowledge the experience before offering advice",
            }
        }
        provider = SequencedProvider([response])
        frame = asyncio.run(LLMTaskFrameBuilder(provider).build("I barely slept."))

        self.assertEqual(frame.query_kind, "statement")
        self.assertGreater(
            frame.response_obligations[0].priority,
            frame.response_obligations[1].priority,
        )
        self.assertEqual(provider.calls[0][1]["call_site"], "influence.task_frame")
        self.assertIn("several obligations are close in priority", provider.calls[0][0])


if __name__ == "__main__":
    unittest.main()
