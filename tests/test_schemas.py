import unittest
from pydantic import ValidationError

from my_mem.schemas import EvidenceGap, InfluenceHypothesis, InfluenceStatus, ValidatedInfluence


class SchemaTests(unittest.TestCase):
    def test_irrelevant_hypothesis_cannot_request_evidence(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                status=InfluenceStatus.IRRELEVANT,
                gaps=[
                    EvidenceGap(
                        gap_id="g1",
                        proposition="A testable claim",
                        source_type="knowledge",
                        retrieval_query="claim",
                        decision_relevance="could change the decision",
                    )
                ],
            )

    def test_validated_influence_requires_evidence_citation(self):
        with self.assertRaises(ValidationError):
            ValidatedInfluence(
                candidate_id="m1",
                status="validated",
                conclusion="relevant",
            )

