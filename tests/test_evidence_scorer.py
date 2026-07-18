"""Focused tests for evidence scoring edge cases."""

from __future__ import annotations

import os
import sys
import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_scorer import (  # noqa: E402
    SUPPORTED,
    UNSUPPORTED,
    CachedLLMJudge,
    ScoringCache,
    score_claim,
)


class EvidenceScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_path = ROOT_DIR / "results" / "cache" / "test_tmp" / uuid.uuid4().hex
        temp_path.mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_path
        self.scoring_cache = ScoringCache(temp_path / "scoring_cache.jsonl")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            self.llm_judge = CachedLLMJudge(temp_path / "llm_judge_cache.jsonl")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def score(self, claim: str, answer: str, evidence: list[str]) -> dict[str, object]:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            return score_claim(
                example_id="test_case",
                model="mock",
                claim=claim,
                answer=answer,
                evidence=evidence,
                scoring_cache=self.scoring_cache,
                llm_judge=self.llm_judge,
            )

    def test_one_letter_answer_does_not_match_substring(self) -> None:
        result = self.score("mock_highest", "C", ["The C bar is the tallest and is labeled 61."])
        self.assertEqual(result["label"], UNSUPPORTED)

    def test_one_letter_answer_matches_token_boundary(self) -> None:
        result = self.score("The answer is C.", "C", ["The C bar is the tallest and is labeled 61."])
        self.assertEqual(result["label"], SUPPORTED)

    def test_multi_token_answer_matches_adjacent_tokens(self) -> None:
        result = self.score("The top item is New York.", "New York", ["The label New York has the highest value."])
        self.assertEqual(result["label"], SUPPORTED)

    def test_numeric_claim_supported_when_number_is_in_evidence(self) -> None:
        result = self.score("The value is 61.", "C", ["The C bar is the tallest and is labeled 61."])
        self.assertEqual(result["label"], SUPPORTED)

    def test_numeric_claim_unsupported_when_number_is_absent(self) -> None:
        result = self.score("The value is 99.", "C", ["The C bar is the tallest and is labeled 61."])
        self.assertEqual(result["label"], UNSUPPORTED)

    def test_entity_claim_supported_by_evidence_terms(self) -> None:
        result = self.score(
            "FUSE feeds HEAD.",
            "FUSE",
            ["The diagram labels FUSE immediately before HEAD and states FUSE FEEDS HEAD."],
        )
        self.assertEqual(result["label"], SUPPORTED)

    def test_empty_claim_falls_back_to_unsupported(self) -> None:
        result = self.score("", "C", ["The C bar is the tallest."])
        self.assertEqual(result["label"], UNSUPPORTED)
        self.assertEqual(result["scoring_method"], "fallback_no_llm")

    def test_inconclusive_without_api_key_falls_back(self) -> None:
        result = self.score("Blue model likely improved.", "red", ["The blue bar is shown."])
        self.assertEqual(result["label"], UNSUPPORTED)
        self.assertEqual(result["scoring_method"], "fallback_no_llm")


if __name__ == "__main__":
    unittest.main()
