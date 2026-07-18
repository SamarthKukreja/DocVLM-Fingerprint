"""Focused tests for metrics aggregation edge cases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import aggregate_metrics, bootstrap_metric_cis  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_aggregate_metrics_allows_zero_claims(self) -> None:
        raw_outputs = [
            {
                "model": "qwen3_vl_8b",
                "domain": "chart",
                "perturbation": "clean",
                "answer_correct": False,
            }
        ]
        rows = aggregate_metrics(raw_outputs, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total_claims"], 0)
        self.assertEqual(rows[0]["claim_faithfulness"], 0.0)
        self.assertEqual(rows[0]["hallucination_rate"], 0.0)

    def test_bootstrap_metric_cis_are_deterministic_and_bounded(self) -> None:
        raw_outputs = [
            {"model": "m", "case_id": "a:clean", "domain": "chart", "perturbation": "clean", "answer_correct": True},
            {"model": "m", "case_id": "b:clean", "domain": "chart", "perturbation": "clean", "answer_correct": False},
            {"model": "m", "case_id": "c:clean", "domain": "chart", "perturbation": "clean", "answer_correct": True},
        ]
        scored_outputs = [
            {"model": "m", "case_id": "a:clean", "domain": "chart", "perturbation": "clean", "label": "supported"},
            {"model": "m", "case_id": "b:clean", "domain": "chart", "perturbation": "clean", "label": "unsupported"},
            {"model": "m", "case_id": "c:clean", "domain": "chart", "perturbation": "clean", "label": "supported"},
        ]
        first = bootstrap_metric_cis(raw_outputs, scored_outputs, resamples=100, seed=7)
        second = bootstrap_metric_cis(raw_outputs, scored_outputs, resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        row = first[0]
        for metric in ("answer_accuracy", "claim_faithfulness", "hallucination_rate"):
            self.assertLessEqual(row[f"{metric}_ci_low"], row[f"{metric}_mean"])
            self.assertLessEqual(row[f"{metric}_mean"], row[f"{metric}_ci_high"])

    def test_bootstrap_metric_cis_allow_zero_claims(self) -> None:
        raw_outputs = [
            {"model": "m", "case_id": "a:clean", "domain": "chart", "perturbation": "clean", "answer_correct": True},
            {"model": "m", "case_id": "b:clean", "domain": "chart", "perturbation": "clean", "answer_correct": False},
        ]
        rows = bootstrap_metric_cis(raw_outputs, [], resamples=20, seed=1)
        self.assertEqual(rows[0]["total_claims"], 0)
        self.assertEqual(rows[0]["claim_faithfulness_mean"], 0.0)
        self.assertEqual(rows[0]["hallucination_rate_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
