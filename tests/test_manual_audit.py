"""Tests for manual audit worksheet selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from manual_audit import audit_row, select_audit_records  # noqa: E402


class ManualAuditTests(unittest.TestCase):
    def test_select_audit_records_produces_thirty_pending_rows(self) -> None:
        scored = []
        for index in range(40):
            perturbation = "distractor_text" if index < 20 else "clean"
            label = "unsupported" if index < 20 else "supported"
            scored.append(
                {
                    "case_id": f"case_{index:03d}:{perturbation}",
                    "model": "model_a",
                    "domain": "chart",
                    "perturbation": perturbation,
                    "question": "Which bar is highest?",
                    "expected_answer": "A",
                    "answer": "B" if label == "unsupported" else "A",
                    "answer_correct": label == "supported",
                    "claim": f"claim {index}",
                    "label": label,
                    "evidence_used": ["The A bar is highest."],
                }
            )
        failures = [{"case_id": "case_000:distractor_text", "model": "model_a"}]
        selected = select_audit_records(scored, failures)
        self.assertEqual(len(selected), 30)
        self.assertEqual(selected[0]["case_id"], "case_000:distractor_text")
        rows = [audit_row(record) for record in selected]
        self.assertTrue(all(row["human_check"] == "pending" for row in rows))


if __name__ == "__main__":
    unittest.main()
