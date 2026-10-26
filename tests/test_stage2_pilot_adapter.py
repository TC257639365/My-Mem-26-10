import unittest
from pathlib import Path

from my_mem.adapters.stage2_pilot import load_stage2_test1_cases


DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "ProcessedDataset"
    / "stage2_influence_pilot_12"
)


class Stage2PilotAdapterTests(unittest.TestCase):
    def test_loads_full_blind_input_and_support_condition(self):
        cases = load_stage2_test1_cases(DATASET_DIR, condition="Eplus")
        self.assertEqual(len(cases), 12)
        self.assertTrue(all(len(case.candidates) == 6 for case in cases))
        self.assertTrue(all(len(case.evidence_pool) == 2 for case in cases))
        serialized = "\n".join(case.model_dump_json() for case in cases)
        self.assertNotIn('"candidate_role"', serialized)
        self.assertNotIn('"construction_source"', serialized)
        self.assertNotIn('"evidence_relation"', serialized)
        evidence_json = "\n".join(
            doc.model_dump_json() for case in cases for doc in case.evidence_pool
        )
        self.assertNotIn('"Eplus"', evidence_json)

    def test_e0_has_no_evidence(self):
        cases = load_stage2_test1_cases(DATASET_DIR, condition="E0")
        self.assertTrue(all(not case.evidence_pool for case in cases))


if __name__ == "__main__":
    unittest.main()
