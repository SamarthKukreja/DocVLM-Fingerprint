"""Score atomic claims as supported or unsupported by annotation evidence."""

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

from claim_splitter import CLAIM_OUTPUTS_PATH, build_claim_outputs
from dataset import ANNOTATIONS_PATH, load_annotations


ROOT_DIR = Path(__file__).resolve().parents[1]
SCORED_OUTPUTS_PATH = ROOT_DIR / "results" / "scored_outputs.jsonl"
SCORING_CACHE_PATH = ROOT_DIR / "results" / "cache" / "claim_scoring_cache.jsonl"
LLM_JUDGE_CACHE_PATH = ROOT_DIR / "results" / "cache" / "llm_judge_cache.jsonl"

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_LLM_JUDGE_MODEL = "gpt-4o-mini"
LLM_JUDGE_MODEL_ENV = "VLM_FINGERPRINT_LLM_JUDGE_MODEL"
LLM_TIMEOUT_SECONDS = 30

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
ALLOWED_LABELS = {SUPPORTED, UNSUPPORTED}
SCORING_VERSION = "2026-07-02-token-boundary-v1"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "bar",
    "bars",
    "by",
    "category",
    "chart",
    "document",
    "for",
    "has",
    "highest",
    "in",
    "is",
    "it",
    "labeled",
    "line",
    "of",
    "on",
    "page",
    "shows",
    "the",
    "to",
    "total",
    "value",
    "with",
}


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


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _token_sequence(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _contains_token_phrase(text: str, phrase: str) -> bool:
    """Return true when phrase appears as complete adjacent tokens in text."""
    phrase_tokens = _token_sequence(phrase)
    if not phrase_tokens:
        return False
    text_tokens = _token_sequence(text)
    if len(phrase_tokens) > len(text_tokens):
        return False
    return any(
        text_tokens[index : index + len(phrase_tokens)] == phrase_tokens
        for index in range(len(text_tokens) - len(phrase_tokens) + 1)
    )


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in STOPWORDS}


def _numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def _cache_key(model: str, example_id: str, claim: str, evidence: list[str], answer: str) -> str:
    payload = {
        "scoring_version": SCORING_VERSION,
        "model": model,
        "example_id": example_id,
        "claim": claim,
        "evidence": evidence,
        "answer": answer,
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


class ScoringCache:
    """JSONL cache for claim scoring judgments."""

    def __init__(self, path: Path = SCORING_CACHE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        for record in _jsonl_records(path):
            key = record.get("cache_key")
            label = record.get("label")
            if isinstance(key, str) and label in ALLOWED_LABELS:
                self._records[key] = record

    def get(self, key: str) -> dict[str, Any] | None:
        record = self._records.get(key)
        return dict(record) if record is not None else None

    def set(self, key: str, record: dict[str, Any]) -> None:
        if key in self._records:
            return
        if record.get("label") not in ALLOWED_LABELS:
            raise ValueError(f"invalid scoring label: {record.get('label')!r}")
        cached_record = dict(record)
        cached_record["cache_key"] = key
        self._records[key] = cached_record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cached_record, sort_keys=True) + "\n")


class CachedLLMJudge:
    """Strict-JSON LLM fallback used only after simpler checks are inconclusive."""

    def __init__(self, path: Path = LLM_JUDGE_CACHE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv(LLM_JUDGE_MODEL_ENV, DEFAULT_LLM_JUDGE_MODEL)
        self._records: dict[str, dict[str, Any]] = {}
        for record in _jsonl_records(path):
            key = record.get("cache_key")
            label = record.get("label")
            if isinstance(key, str) and label in ALLOWED_LABELS:
                self._records[key] = record

    def judge(self, key: str, claim: str, answer: str, evidence: list[str]) -> dict[str, Any] | None:
        record = self._records.get(key)
        if record is not None:
            return {
                "label": record["label"],
                "scoring_method": "llm_judge_cache",
                "evidence_used": record.get("evidence_used", evidence),
            }
        if not self.api_key:
            return None

        judgment = self._request_judgment(claim=claim, answer=answer, evidence=evidence)
        if judgment is None:
            return None

        cache_record = {
            "cache_key": key,
            "claim": claim,
            "answer": answer,
            "evidence_used": evidence,
            "label": judgment["label"],
            "scoring_method": "llm_judge",
            "model": self.model_name,
        }
        self._records[key] = cache_record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cache_record, sort_keys=True) + "\n")
        return {
            "label": judgment["label"],
            "scoring_method": "llm_judge",
            "evidence_used": evidence,
        }

    def _request_judgment(self, *, claim: str, answer: str, evidence: list[str]) -> dict[str, str] | None:
        request_payload = {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Judge whether the claim is supported by the evidence. Return strict JSON only: "
                        "{\"label\":\"supported\"} or {\"label\":\"unsupported\"}. "
                        "Allowed labels are supported and unsupported. Do not use any other label."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Annotated answer: {answer}\n"
                        f"Evidence: {json.dumps(evidence)}\n"
                        f"Claim: {claim}"
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            OPENAI_CHAT_COMPLETIONS_URL,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
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
        if payload is None:
            return None
        label = payload.get("label")
        if label not in ALLOWED_LABELS:
            return None
        return {"label": label}


def rule_based_score(claim: str, answer: str, evidence: list[str]) -> dict[str, Any] | None:
    """Apply exact, numeric, and label/entity checks before any judge fallback."""
    claim_norm = _normalize(claim)
    answer_norm = _normalize(answer)
    evidence_text = " ".join(evidence)
    evidence_norm = _normalize(evidence_text)

    answer_in_claim = claim_norm == answer_norm or _contains_token_phrase(claim, answer)
    answer_in_evidence = _contains_token_phrase(evidence_text, answer)
    if answer_norm and answer_in_claim and answer_in_evidence:
        return {"label": SUPPORTED, "scoring_method": "rule_based", "evidence_used": evidence}

    claim_numbers = _numbers(claim)
    if claim_numbers:
        evidence_numbers = _numbers(evidence_text)
        if claim_numbers <= evidence_numbers:
            return {"label": SUPPORTED, "scoring_method": "rule_based", "evidence_used": evidence}
        return {"label": UNSUPPORTED, "scoring_method": "rule_based", "evidence_used": evidence}

    claim_entities = _content_tokens(claim)
    evidence_entities = _content_tokens(evidence_text) | _content_tokens(answer)
    if claim_entities:
        absent = claim_entities - evidence_entities
        if not absent:
            return {"label": SUPPORTED, "scoring_method": "rule_based", "evidence_used": evidence}
        if len(absent) == len(claim_entities):
            return {"label": UNSUPPORTED, "scoring_method": "rule_based", "evidence_used": evidence}

    return None


def evidence_field_score(claim: str, evidence: list[str]) -> dict[str, Any] | None:
    """Compare remaining claim content against the human-written evidence field."""
    claim_terms = _content_tokens(claim)
    if not claim_terms:
        return None

    evidence_text = " ".join(evidence)
    evidence_terms = _content_tokens(evidence_text)
    if not evidence_terms:
        return None

    overlap = claim_terms & evidence_terms
    overlap_ratio = len(overlap) / len(claim_terms)
    if overlap_ratio >= 0.6:
        return {"label": SUPPORTED, "scoring_method": "evidence_field", "evidence_used": evidence}
    if overlap_ratio == 0:
        return {"label": UNSUPPORTED, "scoring_method": "evidence_field", "evidence_used": evidence}
    return None


def score_claim(
    *,
    example_id: str,
    model: str,
    claim: str,
    answer: str,
    evidence: list[str],
    scoring_cache: ScoringCache | None = None,
    llm_judge: CachedLLMJudge | None = None,
) -> dict[str, Any]:
    """Score one claim using the Day 4 hierarchy."""
    scoring_cache = scoring_cache or ScoringCache()
    llm_judge = llm_judge or CachedLLMJudge()
    key = _cache_key(model, example_id, claim, evidence, answer)

    cached = scoring_cache.get(key)
    if cached is not None:
        cached["cached"] = True
        return cached

    decision = rule_based_score(claim, answer, evidence)
    if decision is None:
        decision = evidence_field_score(claim, evidence)
    if decision is None:
        decision = llm_judge.judge(key, claim=claim, answer=answer, evidence=evidence)
    if decision is None:
        decision = {"label": UNSUPPORTED, "scoring_method": "fallback_no_llm", "evidence_used": evidence}

    record = {
        "example_id": example_id,
        "model": model,
        "claim": claim,
        "label": decision["label"],
        "scoring_method": decision["scoring_method"],
        "evidence_used": decision.get("evidence_used", evidence),
        "cached": False,
    }
    scoring_cache.set(key, record)
    record["cache_key"] = key
    return record


def _annotation_index() -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in load_annotations(ANNOTATIONS_PATH)}


def _claim_outputs() -> list[dict[str, Any]]:
    records = _jsonl_records(CLAIM_OUTPUTS_PATH)
    return records if records else build_claim_outputs()


def build_scored_outputs() -> list[dict[str, Any]]:
    annotations = _annotation_index()
    scoring_cache = ScoringCache()
    llm_judge = CachedLLMJudge()
    scored: list[dict[str, Any]] = []

    for record in _claim_outputs():
        example_id = str(record.get("example_id", ""))
        annotation = annotations.get(example_id)
        if annotation is None:
            continue
        evidence = [str(item) for item in annotation["evidence"]]
        answer = str(annotation["answer"])
        model = str(record.get("model", "unknown"))
        claims = record.get("claims", [])
        if not isinstance(claims, list):
            continue
        for claim_record in claims:
            if not isinstance(claim_record, dict):
                continue
            claim = str(claim_record.get("claim", "")).strip()
            if not claim:
                continue
            scored.append(
                score_claim(
                    example_id=example_id,
                    model=model,
                    claim=claim,
                    answer=answer,
                    evidence=evidence,
                    scoring_cache=scoring_cache,
                    llm_judge=llm_judge,
                )
            )
    return scored


def validate_scored_outputs(records: list[dict[str, Any]]) -> None:
    required = {"example_id", "model", "claim", "label", "scoring_method"}
    for index, record in enumerate(records, start=1):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"scored output {index}: missing {', '.join(missing)}")
        if record["label"] not in ALLOWED_LABELS:
            raise ValueError(f"scored output {index}: invalid label {record['label']!r}")

        if record["scoring_method"] in {"llm_judge", "llm_judge_cache"}:
            cache_key = record.get("cache_key")
            judge_cache = _jsonl_records(LLM_JUDGE_CACHE_PATH)
            if not cache_key or not any(judgment.get("cache_key") == cache_key for judgment in judge_cache):
                raise ValueError("LLM judge output must have a matching cached JSONL record")


def run_self_test() -> None:
    scoring_cache = ScoringCache(ROOT_DIR / "results" / "cache" / "self_test_scoring_cache.jsonl")
    llm_judge = CachedLLMJudge(ROOT_DIR / "results" / "cache" / "self_test_llm_judge_cache.jsonl")
    cases = [
        ("case_exact", "mock", "C", "C", ["The C bar is the tallest and is labeled 61."], SUPPORTED),
        (
            "case_substring_false_positive",
            "mock",
            "mock_highest",
            "C",
            ["The C bar is the tallest and is labeled 61."],
            UNSUPPORTED,
        ),
        ("case_numeric", "mock", "The value is 61.", "C", ["The C bar is the tallest and is labeled 61."], SUPPORTED),
        ("case_absent", "mock", "ResNet has the highest value.", "C", ["The C bar is the tallest."], UNSUPPORTED),
    ]
    records = [
        score_claim(
            example_id=example_id,
            model=model,
            claim=claim,
            answer=answer,
            evidence=evidence,
            scoring_cache=scoring_cache,
            llm_judge=llm_judge,
        )
        for example_id, model, claim, answer, evidence, _expected in cases
    ]
    validate_scored_outputs(records)
    for record, case in zip(records, cases):
        expected = case[-1]
        if record["label"] != expected:
            raise AssertionError(f"{case[0]} expected {expected}, got {record['label']}")

    empty_claims: list[dict[str, Any]] = []
    validate_scored_outputs(empty_claims)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score atomic claims against annotation evidence.")
    parser.add_argument("--output", type=Path, default=SCORED_OUTPUTS_PATH)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("self-test passed")
        return 0

    outputs = build_scored_outputs()
    validate_scored_outputs(outputs)
    _write_jsonl(args.output, outputs)
    print(f"wrote {len(outputs)} scored claim records to {args.output.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
