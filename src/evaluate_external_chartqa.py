"""Clean-only evaluation for an imported ChartQA-style external sanity slice.

This script keeps external-data outputs separate from the generated-dataset main
run. It expects `src/import_chartqa_slice.py` to have produced an
annotations.jsonl file under data/external/chartqa or another local output dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from claim_splitter import validate_claim_outputs
from dataset import ROOT_DIR
from evaluate import answer_matches_expected, extract_final_answer, parse_models, score_claim_outputs, split_raw_outputs, write_jsonl
from metrics import aggregate_metrics, write_metrics_csv
from schema import validate_annotations
from vlm_clients import MODELS_CONFIG_PATH, VLMClientError, get_client

DEFAULT_ANNOTATIONS_PATH = ROOT_DIR / "data" / "external" / "chartqa" / "annotations.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "external" / "chartqa"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"external annotations not found: {path}")
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
    if not records:
        raise ValueError(f"external annotations are empty: {path}")
    return records


def build_external_cases(annotations: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    selected = annotations[:limit] if limit is not None else annotations
    cases: list[dict[str, Any]] = []
    for record in selected:
        example_id = str(record["id"])
        cases.append(
            {
                "case_id": f"{example_id}:external_clean",
                "example_id": example_id,
                "domain": str(record.get("domain", "chart")),
                "image_path": str(record["image_path"]),
                "question": str(record["question"]),
                "expected_answer": str(record["answer"]),
                "evidence": [str(item) for item in record.get("evidence", [])],
                "perturbation": "external_clean",
                "is_clean": True,
            }
        )
    return cases


def evaluate_external_cases(
    cases: list[dict[str, Any]],
    models: list[str],
    config_path: Path = MODELS_CONFIG_PATH,
) -> list[dict[str, Any]]:
    raw_outputs: list[dict[str, Any]] = []
    for model_name in models:
        print(f"loading client: {model_name}", flush=True)
        client = get_client(model_name, config_path=config_path)
        print(f"running {len(cases)} external clean cases for {model_name}", flush=True)
        for index, case in enumerate(cases, start=1):
            print(f"[{model_name}] external {index}/{len(cases)} {case['case_id']}", flush=True)
            try:
                answer = client.answer(case["image_path"], case["question"], case["perturbation"])
                error = ""
            except VLMClientError as exc:
                answer = ""
                error = str(exc)
            raw_outputs.append(
                {
                    "case_id": case["case_id"],
                    "example_id": case["example_id"],
                    "domain": case["domain"],
                    "model": model_name,
                    "image_path": case["image_path"],
                    "question": case["question"],
                    "expected_answer": case["expected_answer"],
                    "answer": answer,
                    "parsed_answer": extract_final_answer(answer),
                    "answer_correct": answer_matches_expected(extract_final_answer(answer), case["expected_answer"]),
                    "perturbation": case["perturbation"],
                    "is_clean": True,
                    "error": error,
                    "external_dataset": "chartqa",
                }
            )
    return raw_outputs


def run_external_evaluation(
    annotations_path: Path = DEFAULT_ANNOTATIONS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models: list[str] | None = None,
    limit: int | None = None,
    config_path: Path = MODELS_CONFIG_PATH,
) -> dict[str, Path]:
    annotations = read_jsonl(annotations_path)
    validate_annotations(annotations, ROOT_DIR)
    model_names = models or ["mock"]
    cases = build_external_cases(annotations, limit=limit)
    annotation_index = {str(record["id"]): record for record in annotations}

    raw_outputs = evaluate_external_cases(cases, model_names, config_path=config_path)
    claim_outputs = split_raw_outputs(raw_outputs)
    validate_claim_outputs(claim_outputs)
    scored_outputs = score_claim_outputs(claim_outputs, annotation_index)
    metrics = aggregate_metrics(raw_outputs, scored_outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_outputs.jsonl"
    claim_path = output_dir / "claim_outputs.jsonl"
    scored_path = output_dir / "scored_outputs.jsonl"
    metrics_path = output_dir / "metrics.csv"
    write_jsonl(raw_path, raw_outputs)
    write_jsonl(claim_path, claim_outputs)
    write_jsonl(scored_path, scored_outputs)
    write_metrics_csv(metrics_path, metrics)
    return {
        "raw_outputs": raw_path,
        "claim_outputs": claim_path,
        "scored_outputs": scored_path,
        "metrics": metrics_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate imported ChartQA examples as a clean-only external sanity check.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default="mock", help="Comma-separated model names from the selected model config.")
    parser.add_argument("--config", type=Path, default=MODELS_CONFIG_PATH, help="Model registry config path. Defaults to configs/models.yaml.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    paths = run_external_evaluation(
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        models=parse_models(args.models, args.config),
        limit=args.limit,
        config_path=args.config,
    )
    for label, path in paths.items():
        print(f"wrote {label}: {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
