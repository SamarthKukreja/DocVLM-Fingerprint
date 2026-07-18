"""Raw model-answer cache for DocVLM-Fingerprint clients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """JSONL-backed cache keyed by model, image, question, and perturbation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = record.get("cache_key")
                    if isinstance(key, str):
                        self._records[key] = record

    @staticmethod
    def make_key(model: str, image_path: str, question: str, perturbation: str | None = None) -> str:
        payload = {
            "model": model,
            "image_path": image_path,
            "question": question,
            "perturbation": perturbation or "clean",
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, model: str, image_path: str, question: str, perturbation: str | None = None) -> str | None:
        key = self.make_key(model, image_path, question, perturbation)
        record = self._records.get(key)
        if record is None:
            return None
        answer = record.get("answer")
        return answer if isinstance(answer, str) else None

    def set(
        self,
        model: str,
        image_path: str,
        question: str,
        answer: str,
        perturbation: str | None = None,
        example_id: str | None = None,
    ) -> None:
        key = self.make_key(model, image_path, question, perturbation)
        if key in self._records:
            return
        record = {
            "cache_key": key,
            "model": model,
            "example_id": example_id or "",
            "image_path": image_path,
            "question": question,
            "perturbation": perturbation or "clean",
            "answer": answer,
            "cached": True,
        }
        self._records[key] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
