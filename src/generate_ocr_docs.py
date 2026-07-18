"""Generate the Day 3 deterministic OCR-heavy document subset."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dataset import ANNOTATIONS_PATH, ROOT_DIR, load_annotations
from generate_charts import BLUE, GREEN, GRID, INK, MUTED, ORANGE, RED, Canvas
from schema import validate_annotations


SAMPLES_DIR = ROOT_DIR / "data" / "samples"
SEED = 20260303


def write_rows(canvas: Canvas, x: int, y: int, rows: list[str], scale: int = 2, gap: int = 24) -> None:
    for idx, row in enumerate(rows):
        canvas.text(x, y + idx * gap, row, INK, scale)


def draw_receipt(path: Path, title: str, rows: list[tuple[str, int]], total: int, receipt_id: str) -> None:
    canvas = Canvas()
    canvas.rect(120, 32, 400, 356, (250, 250, 246))
    canvas.rect(120, 32, 400, 356, INK, fill=False)
    canvas.text(176, 58, title, INK, 3)
    canvas.text(154, 100, f"ID: {receipt_id}", MUTED, 2)
    y = 142
    for label, value in rows:
        canvas.text(154, y, label, INK, 2)
        canvas.text(402, y, str(value), INK, 2)
        y += 34
    canvas.line(150, y, 490, y, GRID)
    canvas.text(154, y + 24, "TOTAL", INK, 3)
    canvas.text(398, y + 24, str(total), BLUE, 3)
    canvas.save_png(path)


def draw_form(path: Path, title: str, fields: list[tuple[str, str]], target_color: tuple[int, int, int]) -> None:
    canvas = Canvas()
    canvas.rect(58, 42, 524, 330, (248, 251, 255))
    canvas.rect(58, 42, 524, 330, INK, fill=False)
    canvas.text(84, 70, title, INK, 3)
    y = 122
    for label, value in fields:
        canvas.text(92, y, label, MUTED, 2)
        canvas.rect(260, y - 8, 244, 30, (255, 255, 255))
        canvas.rect(260, y - 8, 244, 30, GRID, fill=False)
        canvas.text(274, y, value, target_color, 2)
        y += 46
    canvas.save_png(path)


def draw_invoice(path: Path, title: str, rows: list[list[str]], answer_col: int) -> None:
    canvas = Canvas()
    canvas.text(58, 40, title, INK, 3)
    x0, y0 = 50, 94
    col_widths = [150, 120, 120, 120]
    row_height = 42
    headers = ["ITEM", "QTY", "RATE", "DUE"]
    table_width = sum(col_widths)
    canvas.rect(x0, y0, table_width, row_height * (len(rows) + 1), INK, fill=False)
    canvas.rect(x0 + 1, y0 + 1, table_width - 2, row_height - 1, (235, 242, 250))
    x = x0
    for width in col_widths[:-1]:
        x += width
        canvas.line(x, y0, x, y0 + row_height * (len(rows) + 1), INK)
    for idx in range(len(rows) + 2):
        y = y0 + idx * row_height
        canvas.line(x0, y, x0 + table_width, y, INK)
    cursor = x0 + 12
    for header, width in zip(headers, col_widths):
        canvas.text(cursor, y0 + 14, header, INK, 2)
        cursor += width
    for row_idx, row in enumerate(rows):
        cursor = x0 + 12
        y = y0 + row_height * (row_idx + 1) + 14
        for cell_idx, (cell, width) in enumerate(zip(row, col_widths)):
            color = ORANGE if cell_idx == answer_col else MUTED
            canvas.text(cursor, y, cell, color, 2)
            cursor += width
    canvas.save_png(path)


def draw_policy(path: Path, title: str, lines: list[str], highlight: str) -> None:
    canvas = Canvas()
    canvas.rect(70, 36, 500, 348, (255, 255, 252))
    canvas.rect(70, 36, 500, 348, INK, fill=False)
    canvas.text(96, 64, title, INK, 3)
    y = 116
    for line in lines:
        color = GREEN if highlight in line else INK
        canvas.text(100, y, line, color, 2)
        y += 34
    canvas.save_png(path)


def annotation(
    example_id: str,
    question: str,
    answer: str,
    evidence: str,
    question_type: str,
) -> dict[str, object]:
    return {
        "id": example_id,
        "domain": "ocr_doc",
        "image_path": f"data/samples/{example_id}.png",
        "question": question,
        "answer": answer,
        "evidence": [evidence],
        "question_type": question_type,
    }


def generate() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for idx in range(10):
        example_id = f"ocr_{idx + 1:03d}"
        total = rng.randint(42, 98)
        rows = [("BREAD", rng.randint(4, 9)), ("MILK", rng.randint(3, 8)), ("FRUIT", rng.randint(8, 18))]
        draw_receipt(SAMPLES_DIR / f"{example_id}.png", "STORE RECEIPT", rows, total, f"R-{310 + idx}")
        records.append(
            annotation(
                example_id,
                "What is the receipt total?",
                str(total),
                f"The receipt line labeled TOTAL shows {total}.",
                "text_numeric_lookup",
            )
        )

    for idx in range(10):
        example_id = f"ocr_{idx + 11:03d}"
        case_id = f"C-{520 + idx}"
        status = rng.choice(["OPEN", "PAID", "HOLD", "DONE"])
        fields = [("CASE", case_id), ("STATUS", status), ("OWNER", rng.choice(["ALPHA", "BETA", "GAMMA"]))]
        draw_form(SAMPLES_DIR / f"{example_id}.png", "SERVICE FORM", fields, BLUE)
        records.append(
            annotation(
                example_id,
                "What is the status?",
                status,
                f"The STATUS field is filled with {status}.",
                "key_value_lookup",
            )
        )

    for idx in range(10):
        example_id = f"ocr_{idx + 21:03d}"
        rows = []
        due_values = [rng.randint(20, 90) for _ in range(4)]
        names = ["ALPHA", "BETA", "DELTA", "OMEGA"]
        for name, due in zip(names, due_values):
            rows.append([name, str(rng.randint(1, 5)), str(rng.randint(7, 18)), str(due)])
        max_row = max(rows, key=lambda row: int(row[3]))
        draw_invoice(SAMPLES_DIR / f"{example_id}.png", "INVOICE TABLE", rows, 3)
        records.append(
            annotation(
                example_id,
                "Which item has the highest due value?",
                max_row[0],
                f"The row for {max_row[0]} has the highest DUE value, {max_row[3]}.",
                "table_text_lookup",
            )
        )

    for idx in range(10):
        example_id = f"ocr_{idx + 31:03d}"
        days = rng.choice(["15", "30", "45", "60"])
        code = f"P-{700 + idx}"
        lines = [
            f"POLICY CODE: {code}",
            "SECTION A: ACCESS",
            f"REVIEW DAYS: {days}",
            "NOTICE: INTERNAL",
            "OWNER: TEAM DATA",
        ]
        draw_policy(SAMPLES_DIR / f"{example_id}.png", "POLICY PAGE", lines, days)
        records.append(
            annotation(
                example_id,
                "How many review days are listed?",
                days,
                f"The policy page states REVIEW DAYS: {days}.",
                "text_numeric_lookup",
            )
        )

    existing_records = [record for record in load_annotations(ANNOTATIONS_PATH) if record.get("domain") != "ocr_doc"]
    combined = existing_records + records
    validate_annotations(combined, ROOT_DIR)
    with ANNOTATIONS_PATH.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def main() -> int:
    records = generate()
    print(f"wrote {len(records)} OCR document annotations to {ANNOTATIONS_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote OCR document PNGs to {SAMPLES_DIR.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
