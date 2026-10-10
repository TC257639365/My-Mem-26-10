import asyncio
import unittest

from my_mem.evidence.fixed_corpus import FixedCorpusRetriever
from my_mem.schemas import EvidenceDocument, EvidenceGap


class FixedCorpusTests(unittest.TestCase):
    def test_returns_provenance_and_best_overlap(self):
        gap = EvidenceGap(
            gap_id="g1",
            proposition="该植物是否对犬类有毒",
            source_type="external_knowledge",
            retrieval_query="植物 犬类 有毒",
            decision_relevance="changes household safety decision",
        )
        docs = [
            EvidenceDocument(
                document_id="d1",
                content="该植物对犬类有毒，摄入后需要联系兽医。",
                source_type="external_knowledge",
                provenance="fixture:d1",
            ),
            EvidenceDocument(
                document_id="d2",
                content="这是一份无关的旅行记录。",
                source_type="personal_memory_store",
                provenance="fixture:d2",
            ),
        ]
        result = asyncio.run(FixedCorpusRetriever().retrieve(gap, docs, limit=1))
        self.assertEqual(result[0].document_id, "d1")
        self.assertEqual(result[0].provenance, "fixture:d1")
