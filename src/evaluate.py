"""Run the Day 5 clean and perturbed evaluation pass."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from claim_splitter import ClaimSplitCache, split_answer, validate_claim_outputs
from dataset import ANNOTATIONS_PATH, ROOT_DIR, load_annotations
from evidence_scorer import CachedLLMJudge, ScoringCache, score_claim, validate_scored_outputs
from schema import validate_annotations
from vlm_clients import MODELS_CONFIG_PATH, VLMClientError, get_client, load_model_config


PERTURBATION_METADATA_PATH = ROOT_DIR / "data" / "perturbation_metadata.jsonl"
RAW_OUTPUTS_PATH = ROOT_DIR / "results" / "raw_outputs.jsonl"
CLAIM_OUTPUTS_PATH = ROOT_DIR / "results" / "claim_outputs.jsonl"
SCORED_OUTPUTS_PATH = ROOT_DIR / "results" / "scored_outputs.jsonl"


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def expected_answer_candidates(expected: Any) -> list[str]:
    """Return acceptable answer strings, including list-like dataset labels."""
    if isinstance(expected, (list, tuple, set)):
        return [str(item) for item in expected]
    raw = str(expected).strip()
    if raw.startswith(("[", "(")) and raw.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
    return [raw]


def answer_matches_expected(answer: str, expected: Any) -> bool:
    normalized_answer = normalize_answer(answer)
    return any(normalized_answer == normalize_answer(candidate) for candidate in expected_answer_candidates(expected))


ANSWER_LINE_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$", re.IGNORECASE)
EVIDENCE_LINE_RE = re.compile(r"^\s*(?:evidence|rationale|reason)\s*:\s*(.*)\s*$", re.IGNORECASE)


def extract_final_answer(response: str) -> str:
    """Return the short answer from a two-line response, with single-line fallback."""
    lines = [line.strip() for line in str(response).splitlines() if line.strip()]
    for line in lines:
        match = ANSWER_LINE_RE.match(line)
        if match:
            return match.group(1).strip()
    return lines[0] if lines else ""


def extract_claim_text(response: str) -> str:
    """Return the evidence/rationale text that should be split into support claims."""
    lines = [line.strip() for line in str(response).splitlines() if line.strip()]
    evidence_parts: list[str] = []
    collecting = False
    for line in lines:
        evidence_match = EVIDENCE_LINE_RE.match(line)
        if evidence_match:
            collecting = True
            if evidence_match.group(1).strip():
                evidence_parts.append(evidence_match.group(1).strip())
            continue
        if collecting:
            if ANSWER_LINE_RE.match(line):
                continue
            evidence_parts.append(line)
    if evidence_parts:
        return " ".join(evidence_parts).strip()

    non_answer_lines = [line for line in lines if not ANSWER_LINE_RE.match(line)]
    if len(non_answer_lines) > 1:
        return " ".join(non_answer_lines[1:]).strip()
    return extract_final_answer(response)


def default_models(config_path: Path = MODELS_CONFIG_PATH) -> list[str]:
    configs = load_model_config(config_path)
    mock_models = [name for name, config in configs.items() if config.get("provider") == "mock"]
    if mock_models:
        return sorted(mock_models)
    return sorted(configs)[:1]


def parse_models(raw_models: str | None, config_path: Path = MODELS_CONFIG_PATH) -> list[str]:
    if raw_models is None or not raw_models.strip():
        return default_models(config_path)
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must include at least one model name")
    return models


def load_perturbation_metadata(path: Path = PERTURBATION_METADATA_PATH) -> list[dict[str, str]]:
    metadata = []
    for record in read_jsonl(path):
        required = {"example_id", "domain", "perturbation", "output_path"}
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"{path}: perturbation metadata missing {', '.join(missing)}")
        metadata.append({key: str(value) for key, value in record.items()})
    return metadata


def build_cases(limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    annotations = load_annotations(ANNOTATIONS_PATH)
    validate_annotations(annotations, ROOT_DIR)
    selected = annotations[:limit] if limit is not None else annotations
    annotation_index = {str(record["id"]): record for record in annotations}
    selected_ids = {str(record["id"]) for record in selected}

    cases: list[dict[str, Any]] = []
    for record in selected:
        example_id = str(record["id"])
        cases.append(
            {
                "case_id": f"{example_id}:clean",
                "example_id": example_id,
                "domain": str(record["domain"]),
                "image_path": str(record["image_path"]),
                "question": str(record["question"]),
                "expected_answer": str(record["answer"]),
                "evidence": [str(item) for item in record["evidence"]],
                "perturbation": "clean",
                "is_clean": True,
            }
        )

    for item in sorted(load_perturbation_metadata(), key=lambda row: (row["example_id"], row["perturbation"])):
        example_id = item["example_id"]
        if example_id not in selected_ids:
            continue
        annotation = annotation_index[example_id]
        cases.append(
            {
                "case_id": f"{example_id}:{item['perturbation']}",
                "example_id": example_id,
                "domain": str(annotation["domain"]),
                "image_path": item["output_path"],
                "question": str(annotation["question"]),
                "expected_answer": str(annotation["answer"]),
                "evidence": [str(evidence) for evidence in annotation["evidence"]],
                "perturbation": item["perturbation"],
                "is_clean": False,
            }
        )
    return cases, annotation_index


def evaluate_cases(cases: list[dict[str, Any]], models: list[str], config_path: Path = MODELS_CONFIG_PATH) -> list[dict[str, Any]]:
    raw_outputs: list[dict[str, Any]] = []
    total_cases = len(cases)
    for model_name in models:
        print(f"loading client: {model_name}", flush=True)
        client = get_client(model_name, config_path=config_path)
        print(f"running {total_cases} cases for {model_name}", flush=True)
        for case_index, case in enumerate(cases, start=1):
            print(
                f"[{model_name}] {case_index}/{total_cases} {case['case_id']} {case['perturbation']}",
                flush=True,
            )
            try:
                answer = client.answer(case["image_path"], case["question"], case["perturbation"])
                error = ""
            except VLMClientError as exc:
                answer = ""
                error = str(exc)
            parsed_answer = extract_final_answer(answer)
            answer_correct = answer_matches_expected(parsed_answer, case["expected_answer"])
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
                    "parsed_answer": parsed_answer,
                    "answer_correct": answer_correct,
                    "perturbation": case["perturbation"],
                    "is_clean": case["is_clean"],
                    "error": error,
                }
            )
    return raw_outputs


def split_raw_outputs(raw_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache = ClaimSplitCache()
    claim_outputs: list[dict[str, Any]] = []
    for record in raw_outputs:
        answer = str(record["answer"])
        model = str(record["model"])
        cache_id = str(record["case_id"])
        claim_text = extract_claim_text(answer)
        claims = cache.get(model, cache_id, claim_text)
        if claims is None:
            claims = split_answer(claim_text)["claims"]
        cache_key = cache.set(model, cache_id, claim_text, claims)
        claim_outputs.append(
            {
                "case_id": record["case_id"],
                "example_id": record["example_id"],
                "domain": record["domain"],
                "model": model,
                "image_path": record["image_path"],
                "question": record["question"],
                "expected_answer": record["expected_answer"],
                "answer": answer,
                "parsed_answer": record.get("parsed_answer", extract_final_answer(answer)),
                "claim_source": claim_text,
                "answer_correct": record["answer_correct"],
                "perturbation": record["perturbation"],
                "is_clean": record["is_clean"],
                "claims": claims,
                "cache_key": cache_key,
            }
        )
    validate_claim_outputs(claim_outputs)
    return claim_outputs


def score_claim_outputs(claim_outputs: list[dict[str, Any]], annotation_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scoring_cache = ScoringCache()
    llm_judge = CachedLLMJudge()
    scored_outputs: list[dict[str, Any]] = []

    for record in claim_outputs:
        annotation = annotation_index[str(record["example_id"])]
        evidence = [str(item) for item in annotation["evidence"]]
        for claim_record in record["claims"]:
            claim = str(claim_record.get("claim", "")).strip()
            if not claim:
                continue
            scored = score_claim(
                example_id=str(record["example_id"]),
                model=str(record["model"]),
                claim=claim,
                answer=str(annotation["answer"]),
                evidence=evidence,
                scoring_cache=scoring_cache,
                llm_judge=llm_judge,
            )
            scored.update(
                {
                    "case_id": record["case_id"],
                    "domain": record["domain"],
                    "image_path": record["image_path"],
                    "question": record["question"],
                    "expected_answer": record["expected_answer"],
                    "answer": record["answer"],
                    "parsed_answer": record.get("parsed_answer", extract_final_answer(str(record["answer"]))),
                    "claim_source": record.get("claim_source", ""),
                    "answer_correct": record["answer_correct"],
                    "perturbation": record["perturbation"],
                    "is_clean": record["is_clean"],
                    "claim_type": claim_record.get("type", "factual"),
                }
            )
            scored_outputs.append(scored)

    validate_scored_outputs(scored_outputs)
    return scored_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Day 5 model evaluation over clean and perturbed examples.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of source annotations before perturbations.")
    parser.add_argument("--models", default=None, help="Comma-separated model names from the selected model config.")
    parser.add_argument("--config", type=Path, default=MODELS_CONFIG_PATH, help="Model registry config path. Defaults to configs/models.yaml.")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    models = parse_models(args.models, args.config)
    cases, annotation_index = build_cases(limit=args.limit)
    raw_outputs = evaluate_cases(cases, models, config_path=args.config)
    claim_outputs = split_raw_outputs(raw_outputs)
    scored_outputs = score_claim_outputs(claim_outputs, annotation_index)

    write_jsonl(RAW_OUTPUTS_PATH, raw_outputs)
    write_jsonl(CLAIM_OUTPUTS_PATH, claim_outputs)
    write_jsonl(SCORED_OUTPUTS_PATH, scored_outputs)

    clean_count = sum(1 for case in cases if case["is_clean"])
    perturbed_count = len(cases) - clean_count
    print(f"models: {', '.join(models)}")
    print(f"cases: {len(cases)} ({clean_count} clean, {perturbed_count} perturbed)")
    print(f"wrote {len(raw_outputs)} raw outputs to {RAW_OUTPUTS_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote {len(claim_outputs)} claim outputs to {CLAIM_OUTPUTS_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote {len(scored_outputs)} scored outputs to {SCORED_OUTPUTS_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

