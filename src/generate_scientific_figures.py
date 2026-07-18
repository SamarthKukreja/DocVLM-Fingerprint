"""Generate the Day 6 deterministic scientific figure/page subset."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dataset import ANNOTATIONS_PATH, ROOT_DIR, load_annotations
from generate_charts import BLUE, GREEN, GRID, INK, MUTED, ORANGE, RED, Canvas
from schema import validate_annotations


SAMPLES_DIR = ROOT_DIR / "data" / "samples"
SEED = 20260606


def annotation(
    example_id: str,
    question: str,
    answer: str,
    evidence: str,
    question_type: str,
) -> dict[str, object]:
    return {
        "id": example_id,
        "domain": "scientific_figure",
        "image_path": f"data/samples/{example_id}.png",
        "question": question,
        "answer": answer,
        "evidence": [evidence],
        "question_type": question_type,
    }


def draw_ablation_table(path: Path, title: str, rows: list[tuple[str, int, int]]) -> None:
    canvas = Canvas()
    canvas.text(48, 36, title, INK, 3)
    x0, y0 = 78, 96
    col_widths = [170, 130, 130]
    row_height = 48
    headers = ["METHOD", "ACC", "PARAM"]
    table_width = sum(col_widths)
    canvas.rect(x0, y0, table_width, row_height * (len(rows) + 1), INK, fill=False)
    canvas.rect(x0 + 1, y0 + 1, table_width - 2, row_height - 1, (237, 244, 251))
    x = x0
    for width in col_widths[:-1]:
        x += width
        canvas.line(x, y0, x, y0 + row_height * (len(rows) + 1), INK)
    for idx in range(len(rows) + 2):
        y = y0 + idx * row_height
        canvas.line(x0, y, x0 + table_width, y, INK)
    cursor = x0 + 16
    for header, width in zip(headers, col_widths):
        canvas.text(cursor, y0 + 16, header, INK, 2)
        cursor += width
    for row_idx, (name, acc, param) in enumerate(rows):
        y = y0 + row_height * (row_idx + 1) + 16
        canvas.text(x0 + 16, y, name, INK, 2)
        canvas.text(x0 + col_widths[0] + 18, y, str(acc), BLUE if acc == max(item[1] for item in rows) else MUTED, 2)
        canvas.text(x0 + col_widths[0] + col_widths[1] + 18, y, str(param), MUTED, 2)
    canvas.save_png(path)


def draw_architecture(path: Path, title: str, blocks: list[str], answer: str) -> None:
    canvas = Canvas()
    canvas.text(36, 34, title, INK, 3)
    y = 170
    start_x = 56
    width = 110
    gap = 40
    for idx, block in enumerate(blocks):
        x = start_x + idx * (width + gap)
        color = GREEN if block == answer else (242, 247, 252)
        canvas.rect(x, y, width, 72, color)
        canvas.rect(x, y, width, 72, INK, fill=False)
        canvas.text(x + 14, y + 28, block, INK, 2)
        if idx < len(blocks) - 1:
            arrow_y = y + 36
            canvas.line(x + width, arrow_y, x + width + gap - 10, arrow_y, ORANGE)
            canvas.line(x + width + gap - 10, arrow_y, x + width + gap - 22, arrow_y - 8, ORANGE)
            canvas.line(x + width + gap - 10, arrow_y, x + width + gap - 22, arrow_y + 8, ORANGE)
    canvas.text(76, 304, f"{answer} FEEDS HEAD", MUTED, 2)
    canvas.save_png(path)


def draw_method_bars(path: Path, title: str, methods: list[str], values: list[int]) -> None:
    canvas = Canvas()
    canvas.text(42, 32, title, INK, 3)
    canvas.line(82, 330, 590, 330, INK)
    canvas.line(82, 84, 82, 330, INK)
    for y in [130, 180, 230, 280]:
        canvas.line(82, y, 590, y, GRID)
    max_value = max(values)
    for idx, (method, value) in enumerate(zip(methods, values)):
        x = 118 + idx * 120
        height = int((value / max_value) * 210)
        y = 330 - height
        color = [BLUE, GREEN, ORANGE, RED][idx % 4]
        canvas.rect(x, y, 70, height, color)
        canvas.text(x + 2, y - 26, str(value), INK, 2)
        canvas.text(x - 8, 348, method, MUTED, 2)
    canvas.save_png(path)


def draw_method_page(path: Path, title: str, rows: list[str], highlight: str) -> None:
    canvas = Canvas()
    canvas.rect(70, 34, 500, 350, (255, 255, 252))
    canvas.rect(70, 34, 500, 350, INK, fill=False)
    canvas.text(100, 62, title, INK, 3)
    y = 122
    for row in rows:
        color = GREEN if highlight in row else INK
        canvas.text(102, y, row, color, 2)
        y += 38
    canvas.save_png(path)


def generate() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    methods = ["BASE", "AUG", "FUSE", "FULL"]
    for idx in range(10):
        example_id = f"sci_{idx + 1:03d}"
        acc_values = [rng.randint(58, 88) for _ in methods]
        best_idx = acc_values.index(max(acc_values))
        rows = [(method, acc, rng.choice([12, 18, 24, 30])) for method, acc in zip(methods, acc_values)]
        draw_ablation_table(SAMPLES_DIR / f"{example_id}.png", f"ABLATION {idx + 1:02d}", rows)
        records.append(
            annotation(
                example_id,
                "Which method has the highest accuracy?",
                methods[best_idx],
                f"The ablation table shows {methods[best_idx]} with the highest ACC value, {acc_values[best_idx]}.",
                "scientific_table_lookup",
            )
        )

    block_sets = [
        ["INPUT", "ENC", "FUSE", "HEAD"],
        ["IMAGE", "PATCH", "FUSE", "HEAD"],
        ["TEXT", "ALIGN", "FUSE", "HEAD"],
        ["DATA", "STEM", "MIX", "HEAD"],
    ]
    for idx in range(10):
        example_id = f"sci_{idx + 11:03d}"
        blocks = block_sets[idx % len(block_sets)]
        answer = blocks[-2]
        draw_architecture(SAMPLES_DIR / f"{example_id}.png", f"MODEL FLOW {idx + 1:02d}", blocks, answer)
        records.append(
            annotation(
                example_id,
                "Which block feeds the head?",
                answer,
                f"The diagram labels {answer} immediately before HEAD and states {answer} FEEDS HEAD.",
                "diagram_relation_lookup",
            )
        )

    method_pool = ["CNN", "VLM", "OCR", "HYB"]
    for idx in range(10):
        example_id = f"sci_{idx + 21:03d}"
        values = [rng.randint(40, 96) for _ in method_pool]
        best_idx = values.index(max(values))
        draw_method_bars(SAMPLES_DIR / f"{example_id}.png", f"METHOD SCORE {idx + 1:02d}", method_pool, values)
        records.append(
            annotation(
                example_id,
                "Which method has the top score?",
                method_pool[best_idx],
                f"The {method_pool[best_idx]} bar is the tallest and is labeled {values[best_idx]}.",
                "scientific_plot_lookup",
            )
        )

    losses = ["CE", "MSE", "L1", "KL"]
    for idx in range(10):
        example_id = f"sci_{idx + 31:03d}"
        loss = rng.choice(losses)
        steps = rng.choice(["3", "4", "5", "6"])
        rows = [
            f"METHOD ID: S-{900 + idx}",
            f"LOSS TERM: {loss}",
            f"TRAIN STEPS: {steps}",
            "DATA SPLIT: VAL",
            "OUTPUT TYPE: SCORE",
        ]
        draw_method_page(SAMPLES_DIR / f"{example_id}.png", f"METHOD NOTE {idx + 1:02d}", rows, loss)
        records.append(
            annotation(
                example_id,
                "What loss term is listed?",
                loss,
                f"The method note states LOSS TERM: {loss}.",
                "scientific_page_lookup",
            )
        )

    existing_records = [
        record for record in load_annotations(ANNOTATIONS_PATH) if record.get("domain") != "scientific_figure"
    ]
    combined = existing_records + records
    validate_annotations(combined, ROOT_DIR)
    with ANNOTATIONS_PATH.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def main() -> int:
    records = generate()
    print(f"wrote {len(records)} scientific figure annotations to {ANNOTATIONS_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote scientific figure PNGs to {SAMPLES_DIR.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
