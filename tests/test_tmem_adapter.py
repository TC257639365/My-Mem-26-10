import json
import tempfile
import unittest
from pathlib import Path

from my_mem.adapters.tmem import load_cases, load_evidence_pool


class TMemAdapterTests(unittest.TestCase):
    def test_stage6_candidates_are_converted_to_task_case(self):
        payload = {
            "locomo_exp_user_0": [
                {
                    "query": "是否适合养植物？",
                    "candidates": [
                        {
                            "memory_id": "m1",
                            "memory_type": "item",
                            "content": "狗喜欢咬东西",
                            "retrieval_channels": ["entity_bridge_trigger"],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_results.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cases = load_cases(path)
        self.assertEqual(cases[0].case_id, "locomo_exp_user_0:0")
        self.assertEqual(cases[0].candidates[0].memory_id, "m1")

    def test_evidence_pool_accepts_wrapped_json(self):
        payload = {
            "evidence_pool": [
                {
                    "document_id": "d1",
                    "content": "该植物对犬类有毒",
                    "source_type": "knowledge",
                    "provenance": "fixture:d1",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            pool = load_evidence_pool(path)
        self.assertEqual(pool[0].provenance, "fixture:d1")
