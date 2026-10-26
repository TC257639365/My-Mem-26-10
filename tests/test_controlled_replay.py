import asyncio
import unittest

from my_mem.evidence.controlled_replay import ControlledEvidenceRetriever
from my_mem.schemas import EvidenceDocument, EvidenceGap


class ControlledReplayTests(unittest.TestCase):
    def test_routes_by_candidate_and_channel_without_exposing_internal_metadata(self):
        gap = EvidenceGap(
            gap_id="generated-gap",
            proposition="The rule applies.",
            source_type="external_knowledge",
            retrieval_query="rule applies",
            decision_relevance="changes action",
        )
        documents = [
            EvidenceDocument(
                document_id="neutral-1",
                content="The rule applies to the current case.",
                source_type="external_knowledge",
                provenance="controlled:source:1",
                metadata={"target_candidate_id": "candidate-a", "gold_gap_id": "hidden"},
            ),
            EvidenceDocument(
                document_id="neutral-2",
                content="Evidence for another candidate.",
                source_type="external_knowledge",
                provenance="controlled:source:2",
                metadata={"target_candidate_id": "candidate-b"},
            ),
        ]
        result = asyncio.run(
            ControlledEvidenceRetriever().retrieve(
                gap, documents, limit=1, candidate_id="candidate-a"
            )
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].document_id, "neutral-1")
        self.assertEqual(result[0].relation, "unjudged")
        self.assertNotIn("hidden", result[0].model_dump_json())

    def test_wrong_acquisition_channel_returns_no_evidence(self):
        gap = EvidenceGap(
            gap_id="g1",
            proposition="The user confirms the link.",
            source_type="user",
            retrieval_query="ask user",
            decision_relevance="changes action",
        )
        document = EvidenceDocument(
            document_id="d1",
            content="A public handbook confirms the rule.",
            source_type="external_knowledge",
            provenance="controlled:d1",
            metadata={"target_candidate_id": "c1"},
        )
        result = asyncio.run(
            ControlledEvidenceRetriever().retrieve(
                gap, [document], limit=1, candidate_id="c1"
            )
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
