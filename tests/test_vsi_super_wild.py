from __future__ import annotations

import json
import unittest
from pathlib import Path

from tasks.vsi_super_wild import utils


ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "tasks" / "vsi_super_wild" / "data" / "vsi_super_wild_qa.jsonl"


class VsiSuperWildTest(unittest.TestCase):
    def test_release_task_mapping(self):
        rows = [json.loads(line) for line in QA_PATH.read_text(encoding="utf-8").splitlines()]
        vpo = [row for row in rows if row["task_type"] == "VPO"]
        vmr = [row for row in rows if row["task_type"] == "VMR"]

        self.assertEqual(len(vpo), 1302)
        self.assertEqual(len(vmr), 1215)
        self.assertTrue(all("frames captured at different moments" in row["question"] for row in vpo))
        self.assertTrue(all(len(row["frame_indices"]) == 4 for row in vpo))
        self.assertTrue(all("which direction is the camera/person moving" in row["question"] for row in vmr))
        self.assertTrue(all(len(row["frame_indices"]) == 1 for row in vmr))

    def test_frame_labels_match_release_options(self):
        self.assertEqual(utils._format_frame_label_list(4), "Frame 1, Frame 2, Frame 3, and Frame 4")

    def test_invalid_choice_response_is_not_option_a(self):
        self.assertEqual(utils.extract_letter(""), "")
        self.assertEqual(utils.extract_letter("I cannot answer."), "")
        self.assertEqual(utils.extract_letter("Unknown"), "")
        self.assertEqual(utils.extract_letter("The answer is C."), "C")

    def test_metrics_follow_paper_protocol(self):
        vmr = {"task_type": "VMR", "answer": "B", "options": ["A", "B", "C", "D"]}
        voc = {"task_type": "VOC", "answer": "10", "options": None}

        self.assertEqual(
            utils.process_results(vmr, ["B"]),
            {"vmr_accuracy": 1.0, "overall": {"task_type": "VMR", "score": 1.0}},
        )
        self.assertEqual(
            utils.process_results(voc, ["8"]),
            {"voc_mra": 0.8, "overall": {"task_type": "VOC", "score": 0.8}},
        )

    def test_overall_is_equal_weighted_across_tasks(self):
        results = [
            {"task_type": "VMR", "score": 1.0},
            {"task_type": "VMR", "score": 0.0},
            {"task_type": "VPO", "score": 1.0},
            {"task_type": "VOO", "score": 0.0},
            {"task_type": "VOC", "score": 0.5},
        ]
        self.assertEqual(utils.aggregate_overall(results), 50.0)
        self.assertEqual(utils.aggregate_overall(results[:2]), 50.0)


if __name__ == "__main__":
    unittest.main()
