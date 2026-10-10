import asyncio
import unittest

from my_mem.evidence.fixed_corpus import FixedCorpusRetriever
from my_mem.influence.controller import ControllerConfig, InfluenceController
from my_mem.schemas import (
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
            baseline_response="assess the plant using task-only information",
        )


class FakeBuilder:
    async def build(self, task, task_frame, candidates):
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                influence_level="high",
                matched_dimension_ids=["safety"],
                bridge_steps=[
                    {
                        "claim": "A dog that chews objects may ingest the plant.",
                        "basis": "inference",
                        "confidence": 0.4,
                    }
                ],
                possible_action_change="Avoid a toxic plant or place it out of reach.",
                impact=0.8,
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
            influence_level="high",
            conclusion="supported",
            evidence_ids=[evidence[0].evidence_id],
            confidence=0.9,
        )


class NoneBuilder:
    async def build(self, task, task_frame, candidates):
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                influence_level="none",
                impact=0.05,
            )
        ]


class NovelDimensionBuilder:
    async def build(self, task, task_frame, candidates):
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                influence_level="low",
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
                bridge_steps=[
                    {
                        "claim": "Past upkeep difficulty may recur for this purchase.",
                        "basis": "inference",
                        "confidence": 0.7,
                    }
                ],
                possible_action_change="Mention a low-maintenance alternative.",
                impact=0.4,
                bridge_confidence=0.7,
            )
        ]


class MemoryOnlyValidator:
    async def validate(self, task, candidate, hypothesis, evidence):
        return ValidatedInfluence(
            candidate_id=hypothesis.candidate_id,
            status="validated",
            influence_level="low",
            conclusion="the memory supports a minor adjustment",
            evidence_ids=[f"memory:{candidate.memory_id}"],
            confidence=0.7,
        )


class MustNotRun:
    async def retrieve(self, *args, **kwargs):
        raise AssertionError("none influence must not retrieve evidence")

    async def validate(self, *args, **kwargs):
        raise AssertionError("none influence must not invoke the validator")


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
        )
        result = asyncio.run(controller.run(case))
        self.assertEqual(result.validated_influences[0].status, "validated")
        self.assertEqual(result.validated_influences[0].influence_level, "high")
        self.assertEqual(result.diagnostics["acquisition_rounds"], 1)
        self.assertEqual(result.diagnostics["bridge_reasoning_rounds"], 1)
        self.assertEqual(result.diagnostics["secondary_candidate_recall_rounds"], 0)
        self.assertEqual(result.diagnostics["difficult_bridge_candidate_ids"], ["m1"])
        self.assertEqual(result.task_frame.common_dimensions[0].dimension_id, "safety")

    def test_none_influence_stops_before_retrieval_and_validation(self):
        case = TaskCase(
            case_id="c2",
            task="是否养植物",
            candidates=[CandidateMemory(memory_id="m1", content="去年看过花展")],
        )
        must_not_run = MustNotRun()
        controller = InfluenceController(
            NoneBuilder(),
            must_not_run,
            must_not_run,
            task_frame_builder=FakeTaskFrameBuilder(),
        )
        result = asyncio.run(controller.run(case))
        final = result.validated_influences[0]
        self.assertEqual(final.status, "irrelevant")
        self.assertEqual(final.influence_level, "none")
        self.assertEqual(result.diagnostics["gap_count_proposed"], 0)

    def test_novel_dimension_is_candidate_local_and_does_not_trigger_recall(self):
        case = TaskCase(
            case_id="c3",
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
        )

        result = asyncio.run(controller.run(case))

        self.assertEqual(result.diagnostics["novel_dimension_count"], 1)
        self.assertEqual(result.diagnostics["secondary_candidate_recall_rounds"], 0)
        self.assertEqual(
            [item.dimension_id for item in result.task_frame.common_dimensions],
            ["safety"],
        )
        self.assertEqual(
            result.hypotheses[0].proposed_dimension.dimension_id,
            "novel:m1:maintenance",
        )
