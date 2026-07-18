"""Generate the Day 1 deterministic chart/table subset."""

from __future__ import annotations

import json
import random
import struct
import zlib
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
ANNOTATIONS_PATH = ROOT_DIR / "data" / "annotations.jsonl"
WIDTH = 640
HEIGHT = 420
SEED = 20260101

Color = tuple[int, int, int]

WHITE: Color = (255, 255, 255)
INK: Color = (33, 37, 41)
MUTED: Color = (102, 112, 128)
GRID: Color = (220, 226, 235)
BLUE: Color = (65, 105, 225)
GREEN: Color = (46, 160, 67)
ORANGE: Color = (219, 123, 43)
RED: Color = (205, 72, 72)
PURPLE: Color = (126, 87, 194)
PALETTE = [BLUE, GREEN, ORANGE, RED, PURPLE]


FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ".": ["0", "0", "0", "0", "0", "0", "1"],
    "%": ["10001", "00010", "00100", "01000", "10000", "00000", "10001"],
    ":": ["0", "1", "0", "0", "0", "1", "0"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "010", "010"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["011", "100", "100", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["011", "100", "100", "101", "101", "101", "011"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "010"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["1001", "1101", "1011", "1001", "1001", "1001", "1001"],
    "O": ["010", "101", "101", "101", "101", "101", "010"],
    "P": ["110", "101", "101", "110", "100", "100", "100"],
    "Q": ["010", "101", "101", "101", "101", "010", "001"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["011", "100", "100", "010", "001", "001", "110"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}


class Canvas:
    """Tiny RGB canvas with PNG export."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, background: Color = WHITE) -> None:
        self.width = width
        self.height = height
        self.rows = [bytearray(background * width) for _ in range(height)]

    def pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = x * 3
            self.rows[y][offset : offset + 3] = bytes(color)

    def rect(self, x: int, y: int, w: int, h: int, color: Color, fill: bool = True) -> None:
        if fill:
            for yy in range(max(0, y), min(self.height, y + h)):
                for xx in range(max(0, x), min(self.width, x + w)):
                    self.pixel(xx, yy, color)
            return
        self.line(x, y, x + w, y, color)
        self.line(x, y + h, x + w, y + h, color)
        self.line(x, y, x, y + h, color)
        self.line(x + w, y, x + w, y + h, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def text(self, x: int, y: int, text: str, color: Color = INK, scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT.get(char, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cursor + gx * scale, y + gy * scale, scale, scale, color)
            cursor += (max(len(row) for row in glyph) + 1) * scale

    def save_png(self, path: Path) -> None:
        def chunk(kind: bytes, data: bytes) -> bytes:
            payload = kind + data
            return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        data = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)),
                chunk(b"IDAT", zlib.compress(raw, 9)),
                chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(data)


def draw_frame(canvas: Canvas, title: str) -> None:
    canvas.text(28, 24, title, INK, 2)
    for y in [100, 160, 220, 280, 340]:
        canvas.line(80, y, 590, y, GRID)
    canvas.line(80, 80, 80, 350, INK)
    canvas.line(80, 350, 590, 350, INK)


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[int]) -> None:
    canvas = Canvas()
    draw_frame(canvas, title)
    max_value = max(values)
    bar_width = 54
    gap = 38
    start_x = 115
    for idx, (label, value) in enumerate(zip(labels, values)):
        height = int((value / max_value) * 220)
        x = start_x + idx * (bar_width + gap)
        y = 350 - height
        canvas.rect(x, y, bar_width, height, PALETTE[idx % len(PALETTE)])
        canvas.text(x - 4, y - 24, str(value), INK, 2)
        canvas.text(x - 6, 364, label, MUTED, 2)
    canvas.save_png(path)


def draw_line_chart(path: Path, title: str, labels: list[str], values: list[int]) -> None:
    canvas = Canvas()
    draw_frame(canvas, title)
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1)
    points: list[tuple[int, int]] = []
    for idx, value in enumerate(values):
        x = 95 + idx * 96
        y = 340 - int(((value - min_value) / span) * 220)
        points.append((x, y))
        canvas.rect(x - 5, y - 5, 10, 10, BLUE)
        canvas.text(x - 12, y - 28, str(value), INK, 2)
        canvas.text(x - 10, 364, labels[idx], MUTED, 2)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        for thickness in range(-2, 3):
            canvas.line(x0, y0 + thickness, x1, y1 + thickness, GREEN)
    canvas.save_png(path)


def draw_table(path: Path, title: str, headers: list[str], rows: list[list[str]]) -> None:
    canvas = Canvas()
    canvas.text(28, 24, title, INK, 2)
    x0, y0 = 54, 86
    col_widths = [160, 130, 130, 130]
    row_height = 46
    table_width = sum(col_widths)
    table_height = row_height * (len(rows) + 1)
    canvas.rect(x0, y0, table_width, table_height, INK, fill=False)
    x = x0
    for width in col_widths[:-1]:
        x += width
        canvas.line(x, y0, x, y0 + table_height, INK)
    for idx in range(len(rows) + 1):
        y = y0 + idx * row_height
        canvas.line(x0, y, x0 + table_width, y, INK)
    canvas.rect(x0 + 1, y0 + 1, table_width - 2, row_height - 1, (236, 241, 248))
    cursor = x0 + 12
    for header, width in zip(headers, col_widths):
        canvas.text(cursor, y0 + 15, header, INK, 2)
        cursor += width
    for row_idx, row in enumerate(rows):
        cursor = x0 + 12
        y = y0 + row_height * (row_idx + 1) + 15
        for cell, width in zip(row, col_widths):
            canvas.text(cursor, y, cell, MUTED, 2)
            cursor += width
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
        "domain": "chart",
        "image_path": f"data/samples/{example_id}.png",
        "question": question,
        "answer": answer,
        "evidence": [evidence],
        "question_type": question_type,
    }


def bar_specs(rng: random.Random) -> Iterable[tuple[list[str], list[int], str]]:
    labels_pool = [
        ["A", "B", "C", "D"],
        ["BASE", "OCR", "VLM", "HYB"],
        ["Q1", "Q2", "Q3", "Q4"],
        ["RED", "BLUE", "GREEN", "GOLD"],
    ]
    for idx in range(16):
        labels = labels_pool[idx % len(labels_pool)]
        values = [rng.randint(42, 96) for _ in labels]
        yield labels, values, f"BAR SET {idx + 1:02d}"


def line_specs(rng: random.Random) -> Iterable[tuple[list[str], list[int], str]]:
    labels = ["M1", "M2", "M3", "M4", "M5"]
    for idx in range(12):
        start = rng.randint(35, 72)
        steps = [rng.randint(-7, 12) for _ in labels]
        values: list[int] = []
        current = start
        for step in steps:
            current = max(20, min(99, current + step))
            values.append(current)
        yield labels, values, f"TREND SET {idx + 1:02d}"


def table_specs(rng: random.Random) -> Iterable[tuple[list[str], list[list[str]], str]]:
    headers = ["ITEM", "ACC", "ERR", "N"]
    names = ["ALPHA", "BETA", "GAMMA", "DELTA", "SIGMA", "OMEGA"]
    for idx in range(12):
        chosen = rng.sample(names, 4)
        rows = []
        for name in chosen:
            acc = rng.randint(61, 94)
            err = 100 - acc
            n = rng.choice([50, 75, 100, 125])
            rows.append([name, str(acc), str(err), str(n)])
        yield headers, rows, f"TABLE SET {idx + 1:02d}"


def generate() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for offset, (labels, values, title) in enumerate(bar_specs(rng), start=1):
        example_id = f"chart_{offset:03d}"
        path = SAMPLES_DIR / f"{example_id}.png"
        draw_bar_chart(path, title, labels, values)
        best_idx = values.index(max(values))
        best_label = labels[best_idx]
        best_value = values[best_idx]
        records.append(
            annotation(
                example_id,
                "Which category has the highest value?",
                best_label,
                f"The {best_label} bar is the tallest and is labeled {best_value}.",
                "max_lookup",
            )
        )

    for offset, (labels, values, title) in enumerate(line_specs(rng), start=17):
        example_id = f"chart_{offset:03d}"
        path = SAMPLES_DIR / f"{example_id}.png"
        draw_line_chart(path, title, labels, values)
        final_label = labels[-1]
        final_value = values[-1]
        records.append(
            annotation(
                example_id,
                "What is the final plotted value?",
                str(final_value),
                f"The final point at {final_label} is labeled {final_value}.",
                "numeric_lookup",
            )
        )

    for offset, (headers, rows, title) in enumerate(table_specs(rng), start=29):
        example_id = f"chart_{offset:03d}"
        path = SAMPLES_DIR / f"{example_id}.png"
        draw_table(path, title, headers, rows)
        best_row = max(rows, key=lambda row: int(row[1]))
        records.append(
            annotation(
                example_id,
                "Which item has the highest accuracy?",
                best_row[0],
                f"The table row for {best_row[0]} has the highest ACC value, {best_row[1]}.",
                "table_lookup",
            )
        )

    with ANNOTATIONS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return records


def main() -> int:
    records = generate()
    print(f"wrote {len(records)} chart annotations to {ANNOTATIONS_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote chart PNGs to {SAMPLES_DIR.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
