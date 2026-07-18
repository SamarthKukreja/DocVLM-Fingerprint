"""Summarize a completed manual audit worksheet.

This script intentionally refuses to produce a validation summary while any rows
remain pending. That keeps README/report wording honest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import ROOT_DIR

DEFAULT_AUDIT_PATH = ROOT_DIR / "docs" / "manual_audit.md"
DEFAULT_SUMMARY_PATH = ROOT_DIR / "docs" / "manual_audit_summary.md"
VALID_HUMAN_LABELS = {"human_agree", "human_disagree", "unclear"}


def parse_audit_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "---" or set(cells[0]) == {"-"}:
            continue
        if cells[0] == "case_id":
            headers = cells
            continue
        if headers and len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def summarize_audit(rows: list[dict[str, str]]) -> dict[str, int | float]:
    counts = {"total": len(rows), "human_agree": 0, "human_disagree": 0, "unclear": 0, "pending": 0, "invalid": 0}
    for row in rows:
        label = row.get("human_check", "").strip()
        if label in VALID_HUMAN_LABELS:
            counts[label] += 1
        elif label == "pending":
            counts["pending"] += 1
        else:
            counts["invalid"] += 1
    reviewed = counts["human_agree"] + counts["human_disagree"] + counts["unclear"]
    counts["reviewed"] = reviewed
    counts["agreement_rate"] = counts["human_agree"] / reviewed if reviewed else 0.0
    return counts


def write_audit_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    counts = summarize_audit(rows)
    if counts["pending"] or counts["invalid"]:
        raise ValueError(
            "manual audit is not complete: "
            f"pending={counts['pending']} invalid={counts['invalid']}. "
            "Use only human_agree, human_disagree, or unclear before writing a summary."
        )
    disagreement_notes = [row for row in rows if row.get("human_check") in {"human_disagree", "unclear"}]
    lines = [
        "# Manual Audit Summary",
        "",
        "This summary is generated only after all audit rows have been manually reviewed.",
        "",
        f"- Total rows: {counts['total']}",
        f"- Reviewed rows: {counts['reviewed']}",
        f"- Human agree: {counts['human_agree']}",
        f"- Human disagree: {counts['human_disagree']}",
        f"- Unclear: {counts['unclear']}",
        f"- Agreement rate among reviewed rows: {counts['agreement_rate']:.3f}",
        "",
        "## Disagreement Or Unclear Notes",
        "",
    ]
    if not disagreement_notes:
        lines.append("No disagreement or unclear rows were recorded.")
    else:
        for row in disagreement_notes:
            lines.append(f"- `{row.get('case_id', '')}` / `{row.get('model', '')}`: {row.get('notes', '')}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a manual audit summary after all rows are reviewed.")
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()
    rows = parse_audit_rows(args.audit_path)
    write_audit_summary(rows, args.output)
    print(f"wrote {args.output.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
