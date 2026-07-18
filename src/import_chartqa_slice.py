"""Import a small local ChartQA-style slice as an optional external sanity check."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from dataset import ROOT_DIR
from schema import validate_annotations


QUESTION_KEYS = ("question", "query")
ANSWER_KEYS = ("answer", "label")
IMAGE_KEYS = ("image", "image_path", "imgname", "filename", "file_name")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
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

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "annotations", "examples", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        records: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, dict))
        if records:
            return records
    return []


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return None


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
            try:
                index.setdefault(str(path.relative_to(root)).replace("\\", "/"), path)
            except ValueError:
                pass
    return index


def resolve_image(root: Path, image_value: str, image_index: dict[str, Path]) -> Path | None:
    candidate = Path(image_value)
    candidates = [root / candidate, root / candidate.name]
    for item in candidates:
        if item.exists() and item.is_file():
            return item
    normalized = image_value.replace("\\", "/")
    return image_index.get(normalized) or image_index.get(candidate.name) or image_index.get(candidate.stem)


def iter_candidate_records(chartqa_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(chartqa_root.rglob("*.json")) + sorted(chartqa_root.rglob("*.jsonl")):
        records.extend(read_records(path))
    return records


def import_chartqa_slice(chartqa_root: Path, output_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    chartqa_root = chartqa_root.resolve()
    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_index = build_image_index(chartqa_root)

    selected: list[dict[str, Any]] = []
    for record in iter_candidate_records(chartqa_root):
        if len(selected) >= limit:
            break
        question = first_text(record, QUESTION_KEYS)
        answer = first_text(record, ANSWER_KEYS)
        image_value = first_text(record, IMAGE_KEYS)
        if not question or not answer or not image_value:
            continue
        image_path = resolve_image(chartqa_root, image_value, image_index)
        if image_path is None:
            continue
        output_name = f"chartqa_{len(selected) + 1:03d}{image_path.suffix.lower()}"
        output_image = image_dir / output_name
        shutil.copy2(image_path, output_image)
        selected.append(
            {
                "id": f"chartqa_{len(selected) + 1:03d}",
                "domain": "chart",
                "image_path": str(output_image.relative_to(ROOT_DIR)).replace("\\", "/"),
                "question": question,
                "answer": answer,
                "evidence": [f"ChartQA answer: {answer}"],
                "question_type": "external_chartqa",
            }
        )

    if len(selected) < limit:
        raise ValueError(f"only imported {len(selected)} usable ChartQA records out of requested {limit}")
    validate_annotations(selected, ROOT_DIR)
    annotations_path = output_dir / "annotations.jsonl"
    with annotations_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a small local ChartQA-style slice without downloading data.")
    parser.add_argument("--chartqa-root", required=True, type=Path, help="Local ChartQA dataset directory.")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to import.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data" / "external" / "chartqa")
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    records = import_chartqa_slice(args.chartqa_root, args.output_dir, args.limit)
    print(f"wrote {len(records)} ChartQA records to {(args.output_dir / 'annotations.jsonl').relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
