"""Tests for two-line answer/evidence protocol parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import answer_matches_expected, extract_claim_text, extract_final_answer, normalize_answer  # noqa: E402
from vlm_clients import _clean_generated_answer  # noqa: E402


class AnswerProtocolTests(unittest.TestCase):
    def test_extract_final_answer_from_protocol_response(self) -> None:
        response = "Answer: C\nEvidence: The C bar is tallest in the chart."
        self.assertEqual(extract_final_answer(response), "C")
        self.assertEqual(normalize_answer(extract_final_answer(response)), "c")

    def test_extract_claim_text_prefers_evidence_line(self) -> None:
        response = "Answer: C\nEvidence: The C bar is tallest in the chart."
        self.assertEqual(extract_claim_text(response), "The C bar is tallest in the chart.")


    def test_list_style_expected_answer_matches_chartqa_labels(self) -> None:
        self.assertTrue(answer_matches_expected("0.57", "['0.57']"))
        self.assertTrue(answer_matches_expected("No", "['No']"))
        self.assertFalse(answer_matches_expected("13", "['14']"))

    def test_single_line_answers_remain_supported_for_legacy_outputs(self) -> None:
        self.assertEqual(extract_final_answer("HYB"), "HYB")
        self.assertEqual(extract_claim_text("HYB"), "HYB")

    def test_client_cleanup_preserves_two_line_protocol(self) -> None:
        cleaned = _clean_generated_answer("Answer: Q4\nEvidence: The Q4 bar has the highest value.")
        self.assertIn("Answer: Q4", cleaned)
        self.assertIn("Evidence: The Q4 bar has the highest value.", cleaned)


if __name__ == "__main__":
    unittest.main()
