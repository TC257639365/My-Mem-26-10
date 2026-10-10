import asyncio
import json
import unittest

from my_mem.influence.llm_components import LLMHypothesisBuilder
from my_mem.schemas import CandidateMemory


class SequencedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return json.dumps(next(self.responses))


class LLMHypothesisBuilderTests(unittest.TestCase):
    def test_reasks_when_json_has_wrong_field_types(self):
        invalid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "status": "conditional",
                    "mechanism": "dog may damage the plant",
                    "gaps": [
                        {
                            "gap_id": "g1",
                            "proposition": "the plant is fragile",
                            "source_type": "knowledge",
                            "retrieval_query": "plant fragility",
                            "decision_relevance": "changes placement",
                            "priority": "high",
                            "status": "open",
                        }
                    ],
                    "impact": "high",
                    "uncertainty": "medium",
                }
            ]
        }
        valid = {
            "hypotheses": [
                {
                    "candidate_id": "m1",
                    "status": "conditional",
                    "mechanism": ["dog may damage the plant"],
                    "gaps": [
                        {
                            "gap_id": "g1",
                            "proposition": "the plant is fragile",
                            "source_type": "knowledge",
                            "retrieval_query": "plant fragility",
                            "decision_relevance": "changes placement",
                            "priority": 0.8,
                            "status": "open",
                        }
                    ],
                    "impact": 0.8,
                    "uncertainty": 0.5,
                }
            ]
        }
        provider = SequencedProvider([invalid, valid])
        builder = LLMHypothesisBuilder(provider)

        result = asyncio.run(
            builder.build(
                "Should I buy this plant?",
                [CandidateMemory(memory_id="m1", content="The dog chews things")],
            )
        )

        self.assertEqual(result[0].mechanism, ["dog may damage the plant"])
        self.assertEqual(result[0].impact, 0.8)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("Required JSON Schema", provider.calls[0][0])
        self.assertIn("failed schema validation", provider.calls[1][0])
        self.assertEqual(
            provider.calls[1][1]["call_site"], "influence.hypothesis"
        )


if __name__ == "__main__":
    unittest.main()
