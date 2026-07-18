"""Tests for manual audit summary guardrails."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audit_summary import parse_audit_rows, summarize_audit, write_audit_summary  # noqa: E402


class AuditSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = ROOT_DIR / "tests" / ".tmp" / "audit_summary"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(ROOT_DIR / "tests" / ".tmp", ignore_errors=True)

    def test_summary_refuses_pending_rows(self) -> None:
        rows = [{"case_id": "a", "model": "m", "human_check": "pending", "notes": ""}]
        with self.assertRaises(ValueError):
            write_audit_summary(rows, self.tmp_dir / "summary.md")

    def test_summary_writes_completed_audit(self) -> None:
        rows = [
            {"case_id": "a", "model": "m", "human_check": "human_agree", "notes": ""},
            {"case_id": "b", "model": "m", "human_check": "unclear", "notes": "ambiguous label"},
        ]
        output = self.tmp_dir / "summary.md"
        write_audit_summary(rows, output)
        text = output.read_text(encoding="utf-8")
        self.assertIn("Agreement rate", text)
        self.assertIn("ambiguous label", text)
        counts = summarize_audit(rows)
        self.assertEqual(counts["reviewed"], 2)

    def test_parse_audit_rows(self) -> None:
        audit = self.tmp_dir / "audit.md"
        audit.write_text(
            "| case_id | model | human_check | notes |\n"
            "| --- | --- | --- | --- |\n"
            "| a | m | human_agree | ok |\n",
            encoding="utf-8",
        )
        rows = parse_audit_rows(audit)
        self.assertEqual(rows[0]["human_check"], "human_agree")


if __name__ == "__main__":
    unittest.main()
