import asyncio
import unittest

from my_mem.evidence.fixed_corpus import FixedCorpusRetriever
from my_mem.influence.controller import ControllerConfig, InfluenceController
from my_mem.schemas import (
    CandidateDimensionChallenge,
    CandidateMemory,
    EvidenceDocument,
    EvidenceGap,
    InfluenceHypothesis,
    TaskCase,
    TaskFrame,
    ValidatedInfluence,
)


class FakeTaskFrameBuilder:
    async def build(self, task):
        return TaskFrame(
            query_kind="question",
            explicit_facts=[],
            focal_response_goal="decide whether to keep the plant",
            response_obligations=[
                {
                    "obligation_id": "answer",
                    "description": "give a recommendation",
                    "priority": 1.0,
                    "rationale": "the task asks for a decision",
                }
            ],
            common_dimensions=[
                {
                    "dimension_id": "safety",
                    "name": "household safety",
                    "description": "whether the choice is safe for the household",
                    "scope": "core",
                    "priority": 1.0,
                }
            ],
            open_dimensions=["candidate-grounded constraints not visible in the query"],
            baseline_response="give a task-grounded recommendation skeleton",
        )


class MatchedSafetyChallengeBuilder:
    async def build(self, task, task_frame, candidates):
        return [
            CandidateDimensionChallenge(
                candidate_id=candidate.memory_id,
                matched_dimension_ids=["safety"],
                rationale="The candidate may affect household safety.",
            )
            for candidate in candidates
        ]


class FakeBuilder:
    async def build(self, task, task_frame, candidates, dimension_challenges):
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                potential_impact="high",
                target_dimension_ids=["safety"],
                bridge_steps=[
                    {
                        "claim": "A dog that chews objects may ingest the plant.",
                        "basis": "inference",
                        "confidence": 0.4,
                    },
                    {
                        "claim": "The plant may be toxic to dogs.",
                        "basis": "gap",
                        "confidence": 0.4,
                        "gap_id": "g1",
                    },
                ],
                possible_action_change="Avoid a toxic plant or place it out of reach.",
                bridge_status="open",
                bridge_confidence=0.4,
                gaps=[
                    EvidenceGap(
                        gap_id="g1",
                        proposition="植物是否对犬类有毒",
                        source_type="external_knowledge",
                        retrieval_query="植物 犬类 有毒",
                        decision_relevance="changes decision",
                    )
                ],
            )
        ]


class FakeValidator:
    async def validate(self, task, candidate, hypothesis, evidence):
        return ValidatedInfluence(
            candidate_id=hypothesis.candidate_id,
            status="validated",
            potential_impact=hypothesis.potential_impact,
            validated_impact="high",
            conclusion="supported",
            evidence_ids=[evidence[0].evidence_id],
            validation_confidence=0.9,
        )


class NoneBuilder:
    async def build(self, task, task_frame, candidates, dimension_challenges):
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                potential_impact="none",
                bridge_status="refuted",
                bridge_confidence=0.05,
            )
        ]


class NovelDimensionChallengeBuilder:
    async def build(self, task, task_frame, candidates):
        return [
            CandidateDimensionChallenge(
                candidate_id=candidates[0].memory_id,
                proposed_dimension={
                    "dimension_id": "novel:m1:maintenance",
                    "name": "maintenance burden",
                    "description": "whether the memory changes expected upkeep",
                    "scope": "secondary",
                    "priority": 0.4,
                    "memory_grounding": "The user previously abandoned a high-upkeep plant.",
                    "query_bridge": "Past upkeep difficulty may affect this purchase.",
                    "decision_delta": "Mention a low-maintenance alternative.",
                },
                rationale="The query-only frame omitted maintenance history.",
            )
        ]


class NovelDimensionBuilder:
    async def build(self, task, task_frame, candidates, dimension_challenges):
        dimension_id = dimension_challenges[0].proposed_dimension.dimension_id
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                potential_impact="low",
                target_dimension_ids=[dimension_id],
                bridge_steps=[
                    {
                        "claim": "Past upkeep difficulty may recur for this purchase.",
                        "basis": "inference",
                        "confidence": 0.7,
                    }
                ],
                possible_action_change="Mention a low-maintenance alternative.",
                bridge_status="supported",
                bridge_confidence=0.7,
            )
        ]


class MemoryOnlyValidator:
    async def validate(self, task, candidate, hypothesis, evidence):
        return ValidatedInfluence(
            candidate_id=hypothesis.candidate_id,
            status="validated",
            potential_impact=hypothesis.potential_impact,
            validated_impact="low",
            conclusion="the memory supports a minor adjustment",
            evidence_ids=[f"memory:{candidate.memory_id}"],
            validation_confidence=0.7,
        )


class MustNotRun:
    async def retrieve(self, *args, **kwargs):
        raise AssertionError("this stage must not retrieve evidence")

    async def validate(self, *args, **kwargs):
        raise AssertionError("this stage must not invoke the validator")


class ValidatorMustNotRun:
    async def validate(self, *args, **kwargs):
        raise AssertionError("empty acquisition must preserve an unresolved hypothesis")


class AlwaysInsufficientJudge:
    async def judge(self, gap, evidence):
        return evidence.model_copy(
            update={
                "relation": "insufficient",
                "relation_confidence": 0.9,
                "relation_rationale": "The text discusses the topic but does not decide the gap.",
                "grounding_spans": [],
            }
        )


class ControllerTests(unittest.TestCase):
    def test_runs_one_budgeted_round(self):
        case = TaskCase(
            case_id="c1",
            task="是否养植物",
            candidates=[CandidateMemory(memory_id="m1", content="狗喜欢咬东西")],
            evidence_pool=[
                EvidenceDocument(
                    document_id="d1",
                    content="该植物对犬类有毒",
                    source_type="external_knowledge",
                    provenance="fixture:d1",
                )
            ],
        )
        controller = InfluenceController(
            FakeBuilder(),
            FixedCorpusRetriever(),
            FakeValidator(),
            ControllerConfig(max_total_gaps=1, max_evidence_per_gap=1),
            task_frame_builder=FakeTaskFrameBuilder(),
            dimension_challenge_builder=MatchedSafetyChallengeBuilder(),
        )
        result = asyncio.run(controller.run(case))
        final = result.validated_influences[0]
        self.assertEqual(final.status, "validated")
        self.assertEqual(final.potential_impact, "high")
        self.assertEqual(final.validated_impact, "high")
        self.assertEqual(result.diagnostics["difficult_bridge_candidate_ids"], ["m1"])
        self.assertEqual(result.schema_version, "stage2-v2")

    def test_open_high_impact_is_preserved_when_acquisition_returns_nothing(self):
        case = TaskCase(
            case_id="c2",
            task="是否养植物",
            candidates=[CandidateMemory(memory_id="m1", content="狗喜欢咬东西")],
        )
        controller = InfluenceController(
            FakeBuilder(),
            FixedCorpusRetriever(),
            ValidatorMustNotRun(),
            task_frame_builder=FakeTaskFrameBuilder(),
            dimension_challenge_builder=MatchedSafetyChallengeBuilder(),
        )
        result = asyncio.run(controller.run(case))
        final = result.validated_influences[0]
        self.assertEqual(final.status, "unresolved")
        self.assertEqual(final.potential_impact, "high")
        self.assertIsNone(final.validated_impact)
        self.assertEqual(final.unresolved_gap_ids, ["g1"])

    def test_none_potential_stops_before_retrieval_and_validation(self):
        case = TaskCase(
            case_id="c3",
            task="是否养植物",
            candidates=[CandidateMemory(memory_id="m1", content="去年看过花展")],
        )
        must_not_run = MustNotRun()
        controller = InfluenceController(
            NoneBuilder(),
            must_not_run,
            must_not_run,
            task_frame_builder=FakeTaskFrameBuilder(),
            dimension_challenge_builder=MatchedSafetyChallengeBuilder(),
        )
        result = asyncio.run(controller.run(case))
        final = result.validated_influences[0]
        self.assertEqual(final.status, "irrelevant")
        self.assertEqual(final.validated_impact, "none")

    def test_insufficient_evidence_preserves_unresolved_without_validator(self):
        case = TaskCase(
            case_id="c-insufficient",
            task="是否养植物",
            candidates=[CandidateMemory(memory_id="m1", content="狗喜欢咬东西")],
            evidence_pool=[
                EvidenceDocument(
                    document_id="d1",
                    content="这是一份植物养护介绍，没有犬类毒性结论。",
                    source_type="external_knowledge",
                    provenance="fixture:d1",
                )
            ],
        )
        controller = InfluenceController(
            FakeBuilder(),
            FixedCorpusRetriever(),
            ValidatorMustNotRun(),
            task_frame_builder=FakeTaskFrameBuilder(),
            dimension_challenge_builder=MatchedSafetyChallengeBuilder(),
            evidence_relation_judge=AlwaysInsufficientJudge(),
        )
        result = asyncio.run(controller.run(case))
        final = result.validated_influences[0]
        self.assertEqual(final.status, "unresolved")
        self.assertIsNone(final.validated_impact)
        self.assertEqual(result.acquired_evidence[0].relation, "insufficient")

    def test_novel_dimension_is_reviewed_before_impact_classification(self):
        case = TaskCase(
            case_id="c4",
            task="是否养植物",
            candidates=[
                CandidateMemory(
                    memory_id="m1",
                    content="The user previously abandoned a high-upkeep plant.",
                )
            ],
        )
        controller = InfluenceController(
            NovelDimensionBuilder(),
            FixedCorpusRetriever(),
            MemoryOnlyValidator(),
            task_frame_builder=FakeTaskFrameBuilder(),
            dimension_challenge_builder=NovelDimensionChallengeBuilder(),
        )

        result = asyncio.run(controller.run(case))

        self.assertEqual(result.diagnostics["novel_dimension_count"], 1)
        self.assertEqual(result.diagnostics["secondary_candidate_recall_rounds"], 0)
        self.assertEqual(
            result.dimension_challenges[0].proposed_dimension.dimension_id,
            "novel:m1:maintenance",
        )
        self.assertEqual(
            result.hypotheses[0].target_dimension_ids,
            ["novel:m1:maintenance"],
        )


if __name__ == "__main__":
    unittest.main()
