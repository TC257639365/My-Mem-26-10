import unittest

from pydantic import ValidationError

from my_mem.schemas import (
    BridgeStep,
    AcquiredEvidence,
    CandidateDimensionChallenge,
    EvidenceDocument,
    EvidenceGap,
    EvidenceSourceType,
    InfluenceHypothesis,
    TaskFrame,
    ValidatedInfluence,
)


def _gap():
    return EvidenceGap(
        gap_id="g1",
        proposition="The remembered person is the person in the task.",
        source_type="personal_memory_store",
        retrieval_query="person identity link",
        decision_relevance="determines whether the constraint applies",
    )


class SchemaTests(unittest.TestCase):
    def test_acquired_evidence_distinguishes_unjudged_from_insufficient(self):
        evidence = AcquiredEvidence(
            evidence_id="g1:d1",
            gap_id="g1",
            document_id="d1",
            content="related but inconclusive",
            source_type="external_knowledge",
            provenance="fixture:d1",
        )
        self.assertEqual(evidence.relation, "unjudged")
        with self.assertRaises(ValidationError):
            AcquiredEvidence(
                evidence_id="g1:d1",
                gap_id="g1",
                document_id="d1",
                content="related but inconclusive",
                source_type="external_knowledge",
                provenance="fixture:d1",
                relation="insufficient",
            )

    def test_high_potential_open_bridge_keeps_impact_separate_from_certainty(self):
        hypothesis = InfluenceHypothesis(
            candidate_id="m1",
            potential_impact="high",
            target_dimension_ids=["safety"],
            possible_action_change="avoid the unsafe option",
            bridge_status="open",
            bridge_confidence=0.2,
            bridge_steps=[
                BridgeStep(
                    claim="The remembered person is the person in the task.",
                    basis="gap",
                    confidence=0.2,
                    gap_id="g1",
                )
            ],
            gaps=[_gap()],
        )
        self.assertEqual(hypothesis.potential_impact, "high")
        self.assertEqual(hypothesis.bridge_status, "open")
        self.assertEqual(hypothesis.bridge_confidence, 0.2)

    def test_open_bridge_requires_gap_and_gap_step(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                potential_impact="high",
                target_dimension_ids=["safety"],
                possible_action_change="avoid the unsafe option",
                bridge_status="open",
                bridge_confidence=0.2,
                bridge_steps=[
                    BridgeStep(
                        claim="The option may be unsafe.",
                        basis="inference",
                        confidence=0.2,
                    )
                ],
            )

    def test_none_potential_requires_refuted_bridge(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                potential_impact="none",
                bridge_status="supported",
                bridge_confidence=0.9,
            )

    def test_gap_cannot_repeat_supported_fact(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                potential_impact="high",
                target_dimension_ids=["safety"],
                bridge_status="open",
                bridge_confidence=0.4,
                bridge_steps=[
                    BridgeStep(
                        claim="The user has a peanut allergy.",
                        basis="gap",
                        confidence=0.4,
                        gap_id="g1",
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
        self.assertEqual(document.source_type, EvidenceSourceType.EXTERNAL_KNOWLEDGE)
        with self.assertRaises(ValidationError):
            EvidenceDocument(
                document_id="d2",
                content="old value",
                source_type="knowledge",
                provenance="fixture:d2",
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

    def test_dimension_challenge_is_allowed_before_impact_classification(self):
        challenge = CandidateDimensionChallenge(
            candidate_id="m1",
            proposed_dimension={
                "dimension_id": "novel:m1:safety",
                "name": "latent safety constraint",
                "description": "a candidate-local safety dimension",
                "scope": "core",
                "priority": 0.9,
                "memory_grounding": "The memory records a serious reaction.",
                "query_bridge": "It may involve the same person.",
                "decision_delta": "The safe action would change if linked.",
            },
            rationale="The frame omitted the grounded safety possibility.",
        )
        self.assertEqual(challenge.proposed_dimension.scope, "core")

    def test_gap_bridge_step_must_cite_an_existing_gap(self):
        with self.assertRaises(ValidationError):
            InfluenceHypothesis(
                candidate_id="m1",
                potential_impact="high",
                target_dimension_ids=["safety"],
                possible_action_change="avoid the unsafe option",
                bridge_status="open",
                bridge_confidence=0.3,
                bridge_steps=[
                    {
                        "claim": "the option may be unsafe",
                        "basis": "gap",
                        "confidence": 0.3,
                        "gap_id": "missing-gap",
                    }
                ],
                gaps=[_gap()],
            )

    def test_validated_influence_requires_evidence_citation(self):
        with self.assertRaises(ValidationError):
            ValidatedInfluence(
                candidate_id="m1",
                status="validated",
                potential_impact="high",
                validated_impact="high",
                conclusion="relevant",
            )

    def test_unresolved_result_has_no_validated_impact(self):
        result = ValidatedInfluence(
            candidate_id="m1",
            status="unresolved",
            potential_impact="high",
            validated_impact=None,
            conclusion="identity link remains open",
            unresolved_gap_ids=["g1"],
        )
        self.assertIsNone(result.validated_impact)
        with self.assertRaises(ValidationError):
            ValidatedInfluence(
                candidate_id="m1",
                status="unresolved",
                potential_impact="high",
                validated_impact="low",
                conclusion="incorrect downgrade",
                unresolved_gap_ids=["g1"],
            )

    def test_high_potential_cannot_be_marked_irrelevant(self):
        with self.assertRaises(ValidationError):
            ValidatedInfluence(
                candidate_id="m1",
                status="irrelevant",
                potential_impact="high",
                validated_impact="none",
                conclusion="incorrect terminal state",
            )


if __name__ == "__main__":
    unittest.main()
