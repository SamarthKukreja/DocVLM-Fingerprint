"""Regenerate the local CharXiv supplement from a local CharXiv image zip.

This is an optional data-preparation helper. It is intentionally kept outside
the main evaluation pipeline because it requires Pillow and third-party images.

Expected input:
  A CharXiv validation image zip whose entries are named {figure_id}.jpg.

Output:
  data/external/charxiv/samples/{id}.png
  data/external/charxiv/annotations.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[3]
CHARXIV_DIR = ROOT_DIR / "data" / "external" / "charxiv"
SELECTION_PATH = CHARXIV_DIR / "charxiv_selection.json"
SAMPLES_DIR = CHARXIV_DIR / "samples"
ANNOTATIONS_PATH = CHARXIV_DIR / "annotations.jsonl"
CANVAS_WIDTH = 640
FIT_BOX = (640, 600)
MIN_HEIGHT = 360


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="
") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "
")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-zip", type=Path, required=True, help="CharXiv val images.zip")
    args = parser.parse_args()

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with zipfile.ZipFile(args.images_zip) as archive:
        for item in selection:
            image = Image.open(io.BytesIO(archive.read(f"{item['figure_id']}.jpg"))).convert("RGB")
            image.thumbnail(FIT_BOX)
            canvas = Image.new("RGB", (CANVAS_WIDTH, max(image.height, MIN_HEIGHT)), (255, 255, 255))
            canvas.paste(image, ((CANVAS_WIDTH - image.width) // 2, 0))

            image_rel = f"data/external/charxiv/samples/{item['id']}.png"
            canvas.save(ROOT_DIR / image_rel)
            records.append(
                {
                    "id": item["id"],
                    "domain": "chart_real",
                    "image_path": image_rel,
                    "question": item["question"],
                    "answer": item["answer"],
                    "evidence": item["evidence"],
                    "question_type": item["question_type"],
                    "source": item["source"],
                    "license": item["license"],
                    "external_dataset": "CharXiv",
                }
            )

    write_jsonl(ANNOTATIONS_PATH, records)
    print(f"wrote {len(records)} CharXiv records to {ANNOTATIONS_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
