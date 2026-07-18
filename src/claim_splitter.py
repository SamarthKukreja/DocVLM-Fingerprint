"""Split model answers into atomic claim records.

The Day 4 splitter is intentionally lightweight and deterministic by default.
It accepts raw model-output JSONL records when available and falls back to the
annotation answers so the module can be validated before Day 5 experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dataset import ANNOTATIONS_PATH, load_annotations


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_OUTPUTS_PATH = ROOT_DIR / "results" / "raw_outputs.jsonl"
CLAIM_OUTPUTS_PATH = ROOT_DIR / "results" / "claim_outputs.jsonl"
CLAIM_CACHE_PATH = ROOT_DIR / "results" / "cache" / "claim_splitter_cache.jsonl"

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_CLAIM_SPLITTER_MODEL = "gpt-4o-mini"
CLAIM_SPLITTER_MODEL_ENV = "VLM_FINGERPRINT_CLAIM_SPLITTER_MODEL"
LLM_TIMEOUT_SECONDS = 30

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+|;\s*")
CLAIM_TYPES = {"factual", "inference"}


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _splitter_backend() -> str:
    if os.getenv("OPENAI_API_KEY"):
        model_name = os.getenv(CLAIM_SPLITTER_MODEL_ENV, DEFAULT_CLAIM_SPLITTER_MODEL)
        return f"llm_openai:{model_name}"
    return "deterministic"


def _cache_key(model: str, example_id: str, answer: str) -> str:
    payload = {
        "model": model,
        "example_id": example_id,
        "answer": answer,
        "splitter_backend": _splitter_backend(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_claim_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return None

    claims = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim = _normalize_claim(item)
        if claim["claim"]:
            claims.append(claim)
    return {"claims": claims}


def _llm_split_answer(answer: str) -> dict[str, list[dict[str, str]]] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    request_payload = {
        "model": os.getenv(CLAIM_SPLITTER_MODEL_ENV, DEFAULT_CLAIM_SPLITTER_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Split the model answer into atomic claims. Return strict JSON only: "
                    "{\"claims\":[{\"claim\":\"...\",\"type\":\"factual\"}]}. "
                    "Allowed claim types are factual and inference. Do not score support."
                ),
            },
            {"role": "user", "content": f"Answer:\n{answer}"},
        ],
    }
    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None

    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    payload = _extract_json_object(content)
    return _normalize_claim_payload(payload) if payload is not None else None


class ClaimSplitCache:
    """JSONL-backed cache for split answer claims."""

    def __init__(self, path: Path = CLAIM_CACHE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        for record in _jsonl_records(path):
            key = record.get("cache_key")
            if isinstance(key, str):
                self._records[key] = record

    def get(self, model: str, example_id: str, answer: str) -> list[dict[str, str]] | None:
        record = self._records.get(_cache_key(model, example_id, answer))
        if record is None:
            return None
        claims = record.get("claims")
        if not isinstance(claims, list):
            return None
        return [_normalize_claim(item) for item in claims if isinstance(item, dict)]

    def set(self, model: str, example_id: str, answer: str, claims: list[dict[str, str]]) -> str:
        key = _cache_key(model, example_id, answer)
        if key in self._records:
            return key
        record = {
            "cache_key": key,
            "model": model,
            "example_id": example_id,
            "answer": answer,
            "claims": claims,
        }
        self._records[key] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return key


def _normalize_claim(record: dict[str, Any]) -> dict[str, str]:
    claim = str(record.get("claim", "")).strip()
    claim_type = str(record.get("type", "factual")).strip().lower()
    if claim_type not in CLAIM_TYPES:
        claim_type = "factual"
    return {"claim": claim, "type": claim_type}


def _claim_type(text: str) -> str:
    lowered = text.lower()
    inference_markers = ("likely", "probably", "suggests", "implies", "appears")
    return "inference" if any(marker in lowered for marker in inference_markers) else "factual"


def split_answer(answer: str) -> dict[str, list[dict[str, str]]]:
    """Return strict structured claims for a model answer."""
    answer = answer.strip()
    if not answer:
        return {"claims": []}

    llm_claims = _llm_split_answer(answer)
    if llm_claims is not None:
        return llm_claims

    pieces = [piece.strip(" \t\r\n\"'") for piece in SENTENCE_BOUNDARY_RE.split(answer)]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        pieces = [answer]

    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for piece in pieces:
        normalized = re.sub(r"\s+", " ", piece).strip()
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append({"claim": normalized, "type": _claim_type(normalized)})
    return {"claims": claims}


def _raw_answer_records() -> list[dict[str, str]]:
    raw_records = _jsonl_records(RAW_OUTPUTS_PATH)
    if raw_records:
        return [
            {
                "example_id": str(record.get("example_id") or record.get("id") or ""),
                "model": str(record.get("model") or "unknown"),
                "answer": str(record.get("answer") or ""),
                "source": "raw_model_output",
            }
            for record in raw_records
        ]

    return [
        {
            "example_id": str(record["id"]),
            "model": "reference",
            "answer": str(record["answer"]),
            "source": "annotation_answer",
        }
        for record in load_annotations(ANNOTATIONS_PATH)
    ]


def build_claim_outputs(cache: ClaimSplitCache | None = None) -> list[dict[str, Any]]:
    """Split available answers and return JSONL-ready output records."""
    cache = cache or ClaimSplitCache()
    outputs: list[dict[str, Any]] = []

    for record in _raw_answer_records():
        answer = record["answer"]
        model = record["model"]
        example_id = record["example_id"]
        claims = cache.get(model, example_id, answer)
        if claims is None:
            claims = split_answer(answer)["claims"]
            cache_key = cache.set(model, example_id, answer, claims)
        else:
            cache_key = _cache_key(model, example_id, answer)
        outputs.append(
            {
                "example_id": example_id,
                "model": model,
                "answer": answer,
                "claims": claims,
                "source": record["source"],
                "cache_key": cache_key,
            }
        )
    return outputs


def validate_claim_outputs(records: list[dict[str, Any]]) -> None:
    """Validate the strict Day 4 claim-output shape."""
    for index, record in enumerate(records, start=1):
        for field in ("example_id", "model", "answer", "claims"):
            if field not in record:
                raise ValueError(f"claim output {index}: missing {field}")
        claims = record["claims"]
        if not isinstance(claims, list):
            raise ValueError(f"claim output {index}: claims must be a list")
        for claim_index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                raise ValueError(f"claim output {index}.{claim_index}: claim must be an object")
            normalized = _normalize_claim(claim)
            if not normalized["claim"]:
                raise ValueError(f"claim output {index}.{claim_index}: claim text is empty")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split model answers into atomic claims.")
    parser.add_argument("--output", type=Path, default=CLAIM_OUTPUTS_PATH)
    args = parser.parse_args()

    outputs = build_claim_outputs()
    validate_claim_outputs(outputs)
    _write_jsonl(args.output, outputs)
    print(f"wrote {len(outputs)} claim output records to {args.output.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


