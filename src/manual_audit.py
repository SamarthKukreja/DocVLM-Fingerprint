"""Generate a deterministic manual-audit worksheet for claim scoring labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset import ROOT_DIR


SCORED_OUTPUTS_PATH = ROOT_DIR / "results" / "scored_outputs.jsonl"
FAILURE_EXAMPLES_PATH = ROOT_DIR / "results" / "failure_examples.jsonl"
MANUAL_AUDIT_PATH = ROOT_DIR / "docs" / "manual_audit.md"
AUDIT_LIMIT = 30
AUDIT_COLUMNS = [
    "case_id",
    "model",
    "domain",
    "perturbation",
    "question",
    "expected_answer",
    "model_answer",
    "claim",
    "scorer_label",
    "evidence",
    "human_check",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("case_id", "")),
        str(record.get("model", "")),
        str(record.get("claim", "")),
        str(record.get("label", "")),
    )


def _sort_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("domain", "")),
        str(record.get("perturbation", "")),
        str(record.get("case_id", "")),
        str(record.get("model", "")),
        str(record.get("claim", "")),
    )


def _append_unique(selected: list[dict[str, Any]], seen: set[tuple[str, str, str, str]], records: list[dict[str, Any]], limit: int) -> None:
    for record in records:
        if len(selected) >= limit:
            return
        key = _record_key(record)
        if key in seen:
            continue
        selected.append(record)
        seen.add(key)


def select_audit_records(
    scored_outputs: list[dict[str, Any]], failure_examples: list[dict[str, Any]], limit: int = AUDIT_LIMIT
) -> list[dict[str, Any]]:
    """Select deterministic rows for a small human audit worksheet."""
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    failure_keys = [(str(item.get("case_id", "")), str(item.get("model", ""))) for item in failure_examples]
    for case_id, model in failure_keys:
        matches = [
            record
            for record in scored_outputs
            if str(record.get("case_id", "")) == case_id and str(record.get("model", "")) == model
        ]
        matches.sort(key=lambda record: (str(record.get("label", "")) != "unsupported", str(record.get("claim", ""))))
        _append_unique(selected, seen, matches, limit)

    unsupported_perturbed = [
        record
        for record in scored_outputs
        if str(record.get("label", "")) == "unsupported" and str(record.get("perturbation", "")) != "clean"
    ]
    _append_unique(selected, seen, sorted(unsupported_perturbed, key=_sort_key), limit)

    clean_supported = [
        record
        for record in scored_outputs
        if str(record.get("label", "")) == "supported" and str(record.get("perturbation", "")) == "clean"
    ]
    _append_unique(selected, seen, sorted(clean_supported, key=_sort_key), limit)

    disagreements = [
        record
        for record in scored_outputs
        if (bool(record.get("answer_correct")) and str(record.get("label", "")) == "unsupported")
        or ((not bool(record.get("answer_correct"))) and str(record.get("label", "")) == "supported")
    ]
    _append_unique(selected, seen, sorted(disagreements, key=_sort_key), limit)

    _append_unique(selected, seen, sorted(scored_outputs, key=_sort_key), limit)
    return selected[:limit]


def _markdown_escape(value: object) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    return text.replace("|", "\\|").strip()


def audit_row(record: dict[str, Any]) -> dict[str, str]:
    evidence = record.get("evidence_used", [])
    if isinstance(evidence, list):
        evidence_text = "; ".join(str(item) for item in evidence)
    else:
        evidence_text = str(evidence)
    return {
        "case_id": str(record.get("case_id", "")),
        "model": str(record.get("model", "")),
        "domain": str(record.get("domain", "")),
        "perturbation": str(record.get("perturbation", "")),
        "question": str(record.get("question", "")),
        "expected_answer": str(record.get("expected_answer", "")),
        "model_answer": str(record.get("answer", "")),
        "claim": str(record.get("claim", "")),
        "scorer_label": str(record.get("label", "")),
        "evidence": evidence_text,
        "human_check": "pending",
        "notes": "",
    }


def write_markdown(path: Path, selected: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [audit_row(record) for record in selected]
    lines = [
        "# Manual Claim-Scoring Audit Worksheet",
        "",
        "This worksheet is prepared for human spot-checking. All rows start as `pending`; do not describe them as reviewed by a human until the `human_check` and `notes` fields are manually reviewed.",
        "",
        "Allowed `human_check` labels after manual review: `human_agree`, `human_disagree`, `unclear`.",
        "Use `notes` for corrections, ambiguity, or evidence that the binary scorer missed. After all rows are reviewed, run `python src/audit_summary.py` to generate `docs/manual_audit_summary.md`.",
        "",
        f"Total rows: {len(rows)}",
        "",
        "| " + " | ".join(AUDIT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in AUDIT_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(row[column]) for column in AUDIT_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    scored_outputs = read_jsonl(SCORED_OUTPUTS_PATH)
    failure_examples = read_jsonl(FAILURE_EXAMPLES_PATH)
    selected = select_audit_records(scored_outputs, failure_examples)
    if len(selected) < AUDIT_LIMIT:
        raise ValueError(f"expected {AUDIT_LIMIT} audit rows, selected {len(selected)}")
    write_markdown(MANUAL_AUDIT_PATH, selected)
    print(f"wrote {len(selected)} audit rows to {MANUAL_AUDIT_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

