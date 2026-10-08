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
    ValidatedInfluence,
)


class FakeBuilder:
    async def build(self, task, candidates):
        return [
            InfluenceHypothesis(
                candidate_id=candidates[0].memory_id,
                status="conditional",
                gaps=[
                    EvidenceGap(
                        gap_id="g1",
                        proposition="植物是否对犬类有毒",
                        source_type="knowledge",
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
            conclusion="supported",
            evidence_ids=[evidence[0].evidence_id],
            confidence=0.9,
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
                    source_type="knowledge",
                    provenance="fixture:d1",
                )
            ],
        )
        controller = InfluenceController(
            FakeBuilder(),
            FixedCorpusRetriever(),
            FakeValidator(),
            ControllerConfig(max_total_gaps=1, max_evidence_per_gap=1),
        )
        result = asyncio.run(controller.run(case))
        self.assertEqual(result.validated_influences[0].status, "validated")
        self.assertEqual(result.diagnostics["acquisition_rounds"], 1)
