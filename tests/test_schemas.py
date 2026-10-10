import unittest
from pydantic import ValidationError

from my_mem.schemas import (
    BridgeStep,
    EvidenceDocument,
    EvidenceGap,
    EvidenceSourceType,
    InfluenceHypothesis,
    TaskFrame,
    ValidatedInfluence,
)


class SchemaTests(unittest.TestCase):
    def test_none_hypothesis_cannot_request_evidence(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                influence_level="none",
                impact=0.0,
                gaps=[
                    EvidenceGap(
                        gap_id="g1",
                        proposition="A testable claim",
                        source_type="external_knowledge",
                        retrieval_query="claim",
                        decision_relevance="could change the decision",
                    )
                ],
            )

    def test_impact_must_match_influence_level(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                influence_level="low",
                impact=0.8,
                possible_action_change="minor wording change",
            )

    def test_gap_cannot_repeat_supported_fact(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                influence_level="high",
                impact=0.8,
                matched_dimension_ids=["safety"],
                bridge_steps=[
                    BridgeStep(
                        claim="The allergy changes restaurant safety.",
                        basis="inference",
                        confidence=0.8,
                    )
                ],
                possible_action_change="avoid the restaurant",
                memory_supported_claims=["The user has a peanut allergy."],
                gaps=[
                    EvidenceGap(
                        gap_id="g1",
                        proposition="Whether the user has a peanut allergy.",
                        source_type="user",
                        retrieval_query="user peanut allergy",
                        decision_relevance="changes restaurant safety",
                    )
                ],
            )

    def test_source_type_accepts_only_four_acquisition_channels(self):
        document = EvidenceDocument(
            document_id="d1",
            content="public safety guidance",
            source_type="external_knowledge",
            provenance="fixture:d1",
        )
        self.assertEqual(
            document.source_type, EvidenceSourceType.EXTERNAL_KNOWLEDGE
        )
        with self.assertRaises(ValidationError):
            EvidenceDocument(
                document_id="d2",
                content="old value",
                source_type="knowledge",
                provenance="fixture:d2",
            )
        with self.assertRaises(ValidationError):
            EvidenceGap(
                gap_id="g1",
                proposition="a medical claim",
                source_type="medical_record",
                retrieval_query="medical claim",
                decision_relevance="changes safety",
            )

    def test_task_frame_requires_unique_common_dimension_ids(self):
        with self.assertRaises(ValidationError):
            TaskFrame(
                query_kind="statement",
                focal_response_goal="respond helpfully",
                response_obligations=[
                    {
                        "obligation_id": "o1",
                        "description": "acknowledge the situation",
                        "priority": 0.8,
                        "rationale": "the user made a statement",
                    }
                ],
                common_dimensions=[
                    {
                        "dimension_id": "d1",
                        "name": "safety",
                        "description": "possible safety effect",
                        "scope": "core",
                        "priority": 0.9,
                    },
                    {
                        "dimension_id": "d1",
                        "name": "comfort",
                        "description": "possible comfort effect",
                        "scope": "secondary",
                        "priority": 0.5,
                    },
                ],
                baseline_response="acknowledge first",
            )

    def test_gap_bridge_step_must_cite_an_existing_gap(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                influence_level="high",
                impact=0.8,
                matched_dimension_ids=["safety"],
                possible_action_change="avoid the unsafe option",
                bridge_steps=[
                    {
                        "claim": "the option may be unsafe",
                        "basis": "gap",
                        "confidence": 0.3,
                        "gap_id": "missing-gap",
                    }
                ],
            )

    def test_validated_influence_requires_evidence_citation(self):
        with self.assertRaises(ValidationError):
            ValidatedInfluence(
                candidate_id="m1",
                status="validated",
                influence_level="high",
                conclusion="relevant",
            )

    def test_rejected_result_must_be_none(self):
        with self.assertRaises(ValidationError):
            ValidatedInfluence(
                candidate_id="m1",
                status="rejected",
                influence_level="high",
                conclusion="not supported",
            )
