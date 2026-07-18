"""Tests for optional local ChartQA slice import and external evaluation."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_external_chartqa import run_external_evaluation  # noqa: E402
from import_chartqa_slice import import_chartqa_slice  # noqa: E402


PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c6360f8ffff3f0005fe02fea7f341810000000049454e44ae426082"
)


class ChartQAImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT_DIR / "tests" / ".tmp" / "chartqa_fixture"
        self.output_dir = ROOT_DIR / "tests" / ".tmp" / "chartqa_output"
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
        (self.tmp_root / "images").mkdir(parents=True)
        for index in range(2):
            (self.tmp_root / "images" / f"chart_{index}.png").write_bytes(PNG_BYTES)
        records = [
            {"query": "What is highest?", "label": "A", "imgname": "chart_0.png"},
            {"question": "What is lowest?", "answer": "B", "image_path": "images/chart_1.png"},
        ]
        (self.tmp_root / "annotations.json").write_text(json.dumps(records), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(ROOT_DIR / "tests" / ".tmp", ignore_errors=True)

    def test_import_chartqa_slice_writes_valid_annotations(self) -> None:
        rows = import_chartqa_slice(self.tmp_root, self.output_dir, limit=2)
        self.assertEqual(len(rows), 2)
        self.assertTrue((self.output_dir / "annotations.jsonl").exists())
        self.assertEqual(rows[0]["domain"], "chart")
        self.assertEqual(rows[0]["question_type"], "external_chartqa")
        self.assertTrue((ROOT_DIR / rows[0]["image_path"]).exists())

    def test_external_chartqa_evaluation_writes_separate_outputs(self) -> None:
        import_chartqa_slice(self.tmp_root, self.output_dir, limit=2)
        eval_dir = ROOT_DIR / "tests" / ".tmp" / "chartqa_eval"
        paths = run_external_evaluation(
            annotations_path=self.output_dir / "annotations.jsonl",
            output_dir=eval_dir,
            models=["mock"],
            limit=2,
        )
        self.assertTrue(paths["raw_outputs"].exists())
        self.assertTrue(paths["claim_outputs"].exists())
        self.assertTrue(paths["scored_outputs"].exists())
        self.assertTrue(paths["metrics"].exists())
        self.assertNotEqual(paths["raw_outputs"], ROOT_DIR / "results" / "raw_outputs.jsonl")


if __name__ == "__main__":
    unittest.main()
