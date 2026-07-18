"""Descriptive ablations over existing generated-dataset outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset import ROOT_DIR

RAW_OUTPUTS_PATH = ROOT_DIR / "results" / "raw_outputs.jsonl"
SCORED_OUTPUTS_PATH = ROOT_DIR / "results" / "scored_outputs.jsonl"
METRICS_PATH = ROOT_DIR / "results" / "metrics.csv"
ABLATIONS_DIR = ROOT_DIR / "results" / "ablations"

ANSWER_CLAIM_DISAGREEMENTS_PATH = ABLATIONS_DIR / "answer_claim_disagreements.csv"
SCORER_METHOD_BREAKDOWN_PATH = ABLATIONS_DIR / "scorer_method_breakdown.csv"
PERTURBATION_EFFECT_SIZES_PATH = ABLATIONS_DIR / "perturbation_effect_sizes.csv"
NO_CLAIM_COUNTS_PATH = ABLATIONS_DIR / "no_claim_counts.csv"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _case_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("model", "")), str(record.get("case_id", ""))


def answer_claim_disagreements(raw_outputs: list[dict[str, Any]], scored_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels_by_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in scored_outputs:
        labels_by_case[_case_key(record)].append(str(record.get("label", "")))

    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "total_cases": 0,
            "answer_correct_claim_unsupported": 0,
            "answer_incorrect_claim_supported": 0,
            "answer_correct_all_claims_supported": 0,
            "answer_incorrect_no_supported_claims": 0,
        }
    )
    for record in raw_outputs:
        key = (str(record.get("model", "")), str(record.get("domain", "")), str(record.get("perturbation", "")))
        labels = labels_by_case.get(_case_key(record), [])
        answer_correct = bool(record.get("answer_correct"))
        has_supported = "supported" in labels
        has_unsupported = "unsupported" in labels
        groups[key]["total_cases"] += 1
        if answer_correct and has_unsupported:
            groups[key]["answer_correct_claim_unsupported"] += 1
        if (not answer_correct) and has_supported:
            groups[key]["answer_incorrect_claim_supported"] += 1
        if answer_correct and labels and not has_unsupported:
            groups[key]["answer_correct_all_claims_supported"] += 1
        if (not answer_correct) and not has_supported:
            groups[key]["answer_incorrect_no_supported_claims"] += 1

    rows = []
    for (model, domain, perturbation), values in sorted(groups.items()):
        rows.append({"model": model, "domain": domain, "perturbation": perturbation, **values})
    return rows


def scorer_method_breakdown(scored_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for record in scored_outputs:
        key = (
            str(record.get("model", "")),
            str(record.get("domain", "")),
            str(record.get("perturbation", "")),
            str(record.get("label", "")),
            str(record.get("scoring_method", "unknown")),
        )
        groups[key] += 1
    return [
        {
            "model": model,
            "domain": domain,
            "perturbation": perturbation,
            "label": label,
            "scoring_method": method,
            "claim_count": count,
        }
        for (model, domain, perturbation, label, method), count in sorted(groups.items())
    ]


def perturbation_effect_sizes(metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    clean_by_model_domain = {
        (row["model"], row["domain"]): row
        for row in metric_rows
        if row.get("perturbation") == "clean"
    }
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        perturbation = row.get("perturbation", "")
        if perturbation == "clean":
            continue
        clean = clean_by_model_domain.get((row["model"], row["domain"]))
        if not clean:
            continue
        clean_acc = float(clean["answer_accuracy"])
        pert_acc = float(row["answer_accuracy"])
        clean_faith = float(clean["claim_faithfulness"])
        pert_faith = float(row["claim_faithfulness"])
        rows.append(
            {
                "model": row["model"],
                "domain": row["domain"],
                "perturbation": perturbation,
                "clean_answer_accuracy": clean_acc,
                "perturbed_answer_accuracy": pert_acc,
                "answer_accuracy_drop": clean_acc - pert_acc,
                "clean_claim_faithfulness": clean_faith,
                "perturbed_claim_faithfulness": pert_faith,
                "claim_faithfulness_drop": clean_faith - pert_faith,
            }
        )
    return rows


def no_claim_counts(raw_outputs: list[dict[str, Any]], scored_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claim_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in scored_outputs:
        claim_counts[_case_key(record)] += 1
    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"total_cases": 0, "no_claim_cases": 0})
    for record in raw_outputs:
        key = (str(record.get("model", "")), str(record.get("domain", "")), str(record.get("perturbation", "")))
        groups[key]["total_cases"] += 1
        if claim_counts.get(_case_key(record), 0) == 0:
            groups[key]["no_claim_cases"] += 1
    rows = []
    for (model, domain, perturbation), values in sorted(groups.items()):
        total = values["total_cases"]
        no_claim = values["no_claim_cases"]
        rows.append(
            {
                "model": model,
                "domain": domain,
                "perturbation": perturbation,
                "total_cases": total,
                "no_claim_cases": no_claim,
                "no_claim_rate": no_claim / total if total else 0.0,
            }
        )
    return rows


def generate_ablations(
    raw_outputs_path: Path = RAW_OUTPUTS_PATH,
    scored_outputs_path: Path = SCORED_OUTPUTS_PATH,
    metrics_path: Path = METRICS_PATH,
    output_dir: Path = ABLATIONS_DIR,
) -> dict[str, Path]:
    raw_outputs = read_jsonl(raw_outputs_path)
    scored_outputs = read_jsonl(scored_outputs_path)
    metric_rows = read_csv(metrics_path)

    paths = {
        "answer_claim_disagreements": output_dir / "answer_claim_disagreements.csv",
        "scorer_method_breakdown": output_dir / "scorer_method_breakdown.csv",
        "perturbation_effect_sizes": output_dir / "perturbation_effect_sizes.csv",
        "no_claim_counts": output_dir / "no_claim_counts.csv",
    }
    write_csv(
        paths["answer_claim_disagreements"],
        answer_claim_disagreements(raw_outputs, scored_outputs),
        [
            "model",
            "domain",
            "perturbation",
            "total_cases",
            "answer_correct_claim_unsupported",
            "answer_incorrect_claim_supported",
            "answer_correct_all_claims_supported",
            "answer_incorrect_no_supported_claims",
        ],
    )
    write_csv(
        paths["scorer_method_breakdown"],
        scorer_method_breakdown(scored_outputs),
        ["model", "domain", "perturbation", "label", "scoring_method", "claim_count"],
    )
    write_csv(
        paths["perturbation_effect_sizes"],
        perturbation_effect_sizes(metric_rows),
        [
            "model",
            "domain",
            "perturbation",
            "clean_answer_accuracy",
            "perturbed_answer_accuracy",
            "answer_accuracy_drop",
            "clean_claim_faithfulness",
            "perturbed_claim_faithfulness",
            "claim_faithfulness_drop",
        ],
    )
    write_csv(
        paths["no_claim_counts"],
        no_claim_counts(raw_outputs, scored_outputs),
        ["model", "domain", "perturbation", "total_cases", "no_claim_cases", "no_claim_rate"],
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate descriptive ablations from existing outputs.")
    parser.add_argument("--output-dir", type=Path, default=ABLATIONS_DIR)
    args = parser.parse_args()
    paths = generate_ablations(output_dir=args.output_dir)
    for label, path in paths.items():
        print(f"wrote {label}: {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
