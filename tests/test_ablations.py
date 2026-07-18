"""Tests for descriptive ablation outputs."""

from __future__ import annotations

import csv
import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ablations import (  # noqa: E402
    answer_claim_disagreements,
    generate_ablations,
    no_claim_counts,
    perturbation_effect_sizes,
    scorer_method_breakdown,
)


class AblationTests(unittest.TestCase):
    def test_ablation_helpers_count_disagreements_and_methods(self) -> None:
        raw = [
            {"model": "m", "case_id": "a:clean", "domain": "chart", "perturbation": "clean", "answer_correct": True},
            {"model": "m", "case_id": "b:clean", "domain": "chart", "perturbation": "clean", "answer_correct": False},
            {"model": "m", "case_id": "c:clean", "domain": "chart", "perturbation": "clean", "answer_correct": False},
        ]
        scored = [
            {"model": "m", "case_id": "a:clean", "domain": "chart", "perturbation": "clean", "label": "unsupported", "scoring_method": "rule_based"},
            {"model": "m", "case_id": "b:clean", "domain": "chart", "perturbation": "clean", "label": "supported", "scoring_method": "evidence_field"},
        ]
        disagreements = answer_claim_disagreements(raw, scored)
        self.assertEqual(disagreements[0]["answer_correct_claim_unsupported"], 1)
        self.assertEqual(disagreements[0]["answer_incorrect_claim_supported"], 1)
        methods = scorer_method_breakdown(scored)
        self.assertEqual(sum(row["claim_count"] for row in methods), 2)
        no_claim = no_claim_counts(raw, scored)
        self.assertEqual(no_claim[0]["no_claim_cases"], 1)

    def test_perturbation_effect_sizes_use_clean_baseline(self) -> None:
        rows = [
            {"model": "m", "domain": "chart", "perturbation": "clean", "answer_accuracy": "0.9", "claim_faithfulness": "0.8"},
            {"model": "m", "domain": "chart", "perturbation": "blur_downscale", "answer_accuracy": "0.4", "claim_faithfulness": "0.3"},
        ]
        effects = perturbation_effect_sizes(rows)
        self.assertEqual(len(effects), 1)
        self.assertAlmostEqual(effects[0]["answer_accuracy_drop"], 0.5)
        self.assertAlmostEqual(effects[0]["claim_faithfulness_drop"], 0.5)

    def test_generate_ablations_writes_expected_csvs(self) -> None:
        tmp_dir = ROOT_DIR / "tests" / ".tmp" / "ablations"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        raw_path = tmp_dir / "raw.jsonl"
        scored_path = tmp_dir / "scored.jsonl"
        metrics_path = tmp_dir / "metrics.csv"
        raw_path.write_text('{"model":"m","case_id":"a:clean","domain":"chart","perturbation":"clean","answer_correct":true}\n', encoding="utf-8")
        scored_path.write_text('{"model":"m","case_id":"a:clean","domain":"chart","perturbation":"clean","label":"supported","scoring_method":"rule_based"}\n', encoding="utf-8")
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "domain", "perturbation", "answer_accuracy", "claim_faithfulness"])
            writer.writeheader()
            writer.writerow({"model": "m", "domain": "chart", "perturbation": "clean", "answer_accuracy": "1.0", "claim_faithfulness": "1.0"})
        paths = generate_ablations(raw_path, scored_path, metrics_path, tmp_dir / "out")
        self.assertEqual(set(paths), {"answer_claim_disagreements", "scorer_method_breakdown", "perturbation_effect_sizes", "no_claim_counts"})
        self.assertTrue(all(path.exists() for path in paths.values()))
        shutil.rmtree(ROOT_DIR / "tests" / ".tmp", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
