"""Aggregate Day 5 outputs and generate the core MVP plots."""

from __future__ import annotations

import csv
import json
import math
import random
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset import ROOT_DIR
from report import FAILURE_EXAMPLES_PATH, update_paper_results, write_failure_examples


RAW_OUTPUTS_PATH = ROOT_DIR / "results" / "raw_outputs.jsonl"
SCORED_OUTPUTS_PATH = ROOT_DIR / "results" / "scored_outputs.jsonl"
METRICS_PATH = ROOT_DIR / "results" / "metrics.csv"
METRIC_CIS_PATH = ROOT_DIR / "results" / "metric_cis_by_perturbation.csv"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
ACCURACY_HEATMAP_PATH = FIGURES_DIR / "accuracy_heatmap.png"
FAITHFULNESS_HEATMAP_PATH = FIGURES_DIR / "faithfulness_heatmap.png"
SCATTER_PATH = FIGURES_DIR / "accuracy_vs_faithfulness.png"
PERTURBATION_IMPACT_PATH = FIGURES_DIR / "perturbation_impact.png"

BOOTSTRAP_SEED = 20260703
BOOTSTRAP_RESAMPLES = 2000
CI_CONFIDENCE = 0.95

PERTURBATION_ORDER = {
    "clean": 0,
    "blur_downscale": 1,
    "crop_removal": 2,
    "jpeg_compression": 3,
    "distractor_text": 4,
}

FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    ".": ["0", "0", "0", "0", "0", "0", "1"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    "_": ["000", "000", "000", "000", "000", "000", "111"],
    ":": ["0", "1", "0", "0", "0", "1", "0"],
    "/": ["001", "001", "010", "010", "010", "100", "100"],
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
    "Q": ["010", "101", "101", "101", "101", "110", "011"],
    "R": ["110", "101", "101", "110", "101", "101", "101"],
    "S": ["011", "100", "100", "010", "001", "001", "110"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}

Color = tuple[int, int, int]


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"required input is missing: {path.relative_to(ROOT_DIR)}"
        )
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
    if not records and not allow_empty:
        raise ValueError(f"required input is empty: {path.relative_to(ROOT_DIR)}")
    return records


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def perturbation_sort_key(name: str) -> tuple[int, str]:
    return PERTURBATION_ORDER.get(name, 100), name


def aggregate_metrics(
    raw_outputs: list[dict[str, Any]], scored_outputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_stats: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    claim_stats: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "supported": 0, "unsupported": 0}
    )

    for record in raw_outputs:
        key = (str(record["model"]), str(record["domain"]), str(record["perturbation"]))
        raw_stats[key]["total"] += 1
        raw_stats[key]["correct"] += 1 if bool(record.get("answer_correct")) else 0

    for record in scored_outputs:
        label = str(record["label"])
        if label not in {"supported", "unsupported"}:
            raise ValueError(f"unexpected claim label: {label}")
        key = (str(record["model"]), str(record["domain"]), str(record["perturbation"]))
        claim_stats[key]["total"] += 1
        claim_stats[key][label] += 1

    clean_accuracy = {
        (model, domain): safe_ratio(values["correct"], values["total"])
        for (model, domain, perturbation), values in raw_stats.items()
        if perturbation == "clean"
    }

    rows: list[dict[str, Any]] = []
    for model, domain, perturbation in sorted(
        raw_stats, key=lambda item: (item[0], item[1], perturbation_sort_key(item[2]))
    ):
        raw = raw_stats[(model, domain, perturbation)]
        claims = claim_stats[(model, domain, perturbation)]
        answer_accuracy = safe_ratio(raw["correct"], raw["total"])
        claim_faithfulness = safe_ratio(claims["supported"], claims["total"])
        hallucination_rate = safe_ratio(claims["unsupported"], claims["total"])
        baseline = clean_accuracy.get((model, domain), answer_accuracy)
        rows.append(
            {
                "model": model,
                "domain": domain,
                "perturbation": perturbation,
                "total_answers": raw["total"],
                "correct_answers": raw["correct"],
                "answer_accuracy": answer_accuracy,
                "total_claims": claims["total"],
                "supported_claims": claims["supported"],
                "unsupported_claims": claims["unsupported"],
                "claim_faithfulness": claim_faithfulness,
                "hallucination_rate": hallucination_rate,
                "perturbation_drop": 0.0
                if perturbation == "clean"
                else baseline - answer_accuracy,
            }
        )
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "domain",
        "perturbation",
        "total_answers",
        "correct_answers",
        "answer_accuracy",
        "total_claims",
        "supported_claims",
        "unsupported_claims",
        "claim_faithfulness",
        "hallucination_rate",
        "perturbation_drop",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in (
                "answer_accuracy",
                "claim_faithfulness",
                "hallucination_rate",
                "perturbation_drop",
            ):
                output[key] = f"{float(output[key]):.4f}"
            writer.writerow(output)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = percentile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _case_units_by_perturbation(
    raw_outputs: list[dict[str, Any]], scored_outputs: list[dict[str, Any]]
) -> dict[str, list[dict[str, int]]]:
    claim_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total_claims": 0, "supported_claims": 0, "unsupported_claims": 0}
    )
    for record in scored_outputs:
        label = str(record.get("label", ""))
        if label not in {"supported", "unsupported"}:
            raise ValueError(f"unexpected claim label: {label}")
        key = (str(record["model"]), str(record["case_id"]))
        claim_stats[key]["total_claims"] += 1
        claim_stats[key][f"{label}_claims"] += 1

    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    for record in raw_outputs:
        key = (str(record["model"]), str(record["case_id"]))
        claims = claim_stats[key]
        grouped[str(record["perturbation"])].append(
            {
                "total_answers": 1,
                "correct_answers": 1 if bool(record.get("answer_correct")) else 0,
                "total_claims": claims["total_claims"],
                "supported_claims": claims["supported_claims"],
                "unsupported_claims": claims["unsupported_claims"],
            }
        )
    return grouped


def _summarize_units(units: list[dict[str, int]]) -> dict[str, float | int]:
    total_answers = sum(unit["total_answers"] for unit in units)
    correct_answers = sum(unit["correct_answers"] for unit in units)
    total_claims = sum(unit["total_claims"] for unit in units)
    supported_claims = sum(unit["supported_claims"] for unit in units)
    unsupported_claims = sum(unit["unsupported_claims"] for unit in units)
    return {
        "total_cases": total_answers,
        "total_claims": total_claims,
        "answer_accuracy": safe_ratio(correct_answers, total_answers),
        "claim_faithfulness": safe_ratio(supported_claims, total_claims),
        "hallucination_rate": safe_ratio(unsupported_claims, total_claims),
    }


def bootstrap_metric_cis(
    raw_outputs: list[dict[str, Any]],
    scored_outputs: list[dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Compute deterministic case-level bootstrap CIs by perturbation."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    grouped = _case_units_by_perturbation(raw_outputs, scored_outputs)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    lower_percentile = (1.0 - CI_CONFIDENCE) / 2.0
    upper_percentile = 1.0 - lower_percentile

    for perturbation in sorted(grouped, key=perturbation_sort_key):
        units = grouped[perturbation]
        if not units:
            continue
        observed = _summarize_units(units)
        draws = {
            "answer_accuracy": [],
            "claim_faithfulness": [],
            "hallucination_rate": [],
        }
        for _ in range(resamples):
            sample = [units[rng.randrange(len(units))] for _ in range(len(units))]
            summary = _summarize_units(sample)
            for metric in draws:
                draws[metric].append(float(summary[metric]))
        row: dict[str, Any] = {
            "perturbation": perturbation,
            "total_cases": observed["total_cases"],
            "total_claims": observed["total_claims"],
            "resamples": resamples,
            "seed": seed,
        }
        for metric, values in draws.items():
            values.sort()
            row[f"{metric}_mean"] = float(observed[metric])
            row[f"{metric}_ci_low"] = _percentile(values, lower_percentile)
            row[f"{metric}_ci_high"] = _percentile(values, upper_percentile)
        rows.append(row)
    return rows


def write_metric_cis_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "perturbation",
        "total_cases",
        "total_claims",
        "resamples",
        "seed",
        "answer_accuracy_mean",
        "answer_accuracy_ci_low",
        "answer_accuracy_ci_high",
        "claim_faithfulness_mean",
        "claim_faithfulness_ci_low",
        "claim_faithfulness_ci_high",
        "hallucination_rate_mean",
        "hallucination_rate_ci_low",
        "hallucination_rate_ci_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in fieldnames:
                if key.endswith(("_mean", "_low", "_high")):
                    output[key] = f"{float(output[key]):.4f}"
            writer.writerow(output)


class Canvas:
    def __init__(
        self, width: int, height: int, background: Color = (255, 255, 255)
    ) -> None:
        self.width = width
        self.height = height
        self.rows = [bytearray(background * width) for _ in range(height)]

    def pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = x * 3
            self.rows[y][offset : offset + 3] = bytes(color)

    def rect(self, x: int, y: int, width: int, height: int, color: Color) -> None:
        for yy in range(max(0, y), min(self.height, y + height)):
            row = self.rows[yy]
            for xx in range(max(0, x), min(self.width, x + width)):
                offset = xx * 3
                row[offset : offset + 3] = bytes(color)

    def line(self, x1: int, y1: int, x2: int, y2: int, color: Color) -> None:
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for step in range(steps + 1):
            t = step / steps
            x = round(x1 + (x2 - x1) * t)
            y = round(y1 + (y2 - y1) * t)
            self.pixel(x, y, color)

    def text(
        self, x: int, y: int, text: str, color: Color = (30, 34, 39), scale: int = 2
    ) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT.get(char, FONT[" "])
            for gy, line in enumerate(glyph):
                for gx, bit in enumerate(line):
                    if bit == "1":
                        self.rect(
                            cursor + gx * scale, y + gy * scale, scale, scale, color
                        )
            cursor += (max(len(line) for line in glyph) + 1) * scale

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def chunk(kind: bytes, chunk_data: bytes) -> bytes:
            payload = kind + chunk_data
            return (
                struct.pack(">I", len(chunk_data))
                + payload
                + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
            )

        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        data = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
                ),
                chunk(b"IDAT", zlib.compress(raw, 9)),
                chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(data)


def metric_color(value: float | None) -> Color:
    if value is None:
        return (230, 232, 235)
    value = max(0.0, min(1.0, value))
    red = round(215 - 135 * value)
    green = round(77 + 105 * value)
    blue = round(85 + 30 * value)
    return red, green, blue


def draw_heatmap(
    rows: list[dict[str, Any]], metric_name: str, title: str, path: Path
) -> None:
    row_labels = sorted({f"{row['model']}:{row['domain']}" for row in rows})
    perturbations = sorted(
        {str(row["perturbation"]) for row in rows}, key=perturbation_sort_key
    )
    values = {
        (f"{row['model']}:{row['domain']}", str(row["perturbation"])): float(
            row[metric_name]
        )
        for row in rows
    }

    cell_width = 132
    cell_height = 54
    left = 170
    top = 88
    width = left + cell_width * max(1, len(perturbations)) + 34
    height = top + cell_height * max(1, len(row_labels)) + 54
    canvas = Canvas(width, height)
    canvas.text(24, 22, title, scale=3)

    for col, perturbation in enumerate(perturbations):
        canvas.text(left + col * cell_width + 6, 62, perturbation[:16], scale=1)
    for row_index, row_label in enumerate(row_labels):
        y = top + row_index * cell_height
        canvas.text(24, y + 18, row_label[:22], scale=1)
        for col, perturbation in enumerate(perturbations):
            x = left + col * cell_width
            value = values.get((row_label, perturbation))
            canvas.rect(x, y, cell_width - 4, cell_height - 4, metric_color(value))
            canvas.text(
                x + 42, y + 18, "NA" if value is None else f"{value:.2f}", scale=2
            )
    canvas.save(path)


def point_color(perturbation: str) -> Color:
    palette = {
        "clean": (44, 123, 182),
        "blur_downscale": (253, 174, 97),
        "crop_removal": (215, 48, 39),
        "jpeg_compression": (116, 173, 209),
        "distractor_text": (26, 152, 80),
    }
    return palette.get(perturbation, (80, 80, 80))


def _pillow_font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_centered(
    draw: Any, box: tuple[int, int, int, int], text: str, font: Any, fill: Color
) -> None:
    x0, y0, x1, y1 = box
    width, height = _text_size(draw, text, font)
    draw.text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def _draw_multiline_centered(
    draw: Any,
    box: tuple[int, int, int, int],
    text: str,
    font: Any,
    fill: Color,
    spacing: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    sizes = [_text_size(draw, line, font) for line in lines]
    total_height = sum(height for _, height in sizes) + spacing * max(0, len(lines) - 1)
    y = y0 + (y1 - y0 - total_height) / 2
    for line, (width, height) in zip(lines, sizes):
        draw.text((x0 + (x1 - x0 - width) / 2, y), line, font=font, fill=fill)
        y += height + spacing


def _readable_value_color(color: Color) -> Color:
    red, green, blue = color
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (255, 255, 255) if luminance < 125 else (24, 30, 36)


def _pretty_label(value: str) -> str:
    return value.replace("_", " ")


def _save_pillow(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))


def _draw_heatmap_pillow(
    rows: list[dict[str, Any]], metric_name: str, title: str, path: Path
) -> None:
    from PIL import Image, ImageDraw

    row_pairs = sorted({(str(row["model"]), str(row["domain"])) for row in rows})
    perturbations = sorted(
        {str(row["perturbation"]) for row in rows}, key=perturbation_sort_key
    )
    values = {
        (str(row["model"]), str(row["domain"]), str(row["perturbation"])): float(
            row[metric_name]
        )
        for row in rows
    }

    cell_width = 250
    cell_height = 92
    left = 440
    top = 210
    width = left + cell_width * max(1, len(perturbations)) + 100
    height = top + cell_height * max(1, len(row_pairs)) + 105
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = _pillow_font(46, bold=True)
    label_font = _pillow_font(24, bold=True)
    small_font = _pillow_font(22)
    value_font = _pillow_font(30, bold=True)
    axis_color = (45, 52, 60)
    grid_color = (220, 224, 229)

    draw.text((44, 36), title.title(), font=title_font, fill=axis_color)
    draw.text(
        (44, 105),
        "Rows are model/domain pairs; cells show aggregated metric values.",
        font=small_font,
        fill=(95, 103, 112),
    )

    for col, perturbation in enumerate(perturbations):
        x0 = left + col * cell_width
        label = _pretty_label(perturbation).title().replace(" ", "\n")
        _draw_multiline_centered(
            draw,
            (x0 + 4, 135, x0 + cell_width - 8, top - 18),
            label,
            small_font,
            axis_color,
        )

    for row_index, (model, domain) in enumerate(row_pairs):
        y0 = top + row_index * cell_height
        y1 = y0 + cell_height - 8
        row_label = f"{model}\n{domain}"
        _draw_multiline_centered(
            draw, (34, y0, left - 34, y1), row_label, small_font, axis_color
        )
        for col, perturbation in enumerate(perturbations):
            x0 = left + col * cell_width
            x1 = x0 + cell_width - 10
            value = values.get((model, domain, perturbation))
            color = metric_color(value)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=color)
            draw.rounded_rectangle(
                (x0, y0, x1, y1), radius=12, outline=(255, 255, 255), width=3
            )
            _draw_centered(
                draw,
                (x0, y0, x1, y1),
                "NA" if value is None else f"{value:.2f}",
                value_font,
                _readable_value_color(color),
            )

    draw.line(
        (left, top - 4, left + cell_width * len(perturbations), top - 4),
        fill=grid_color,
        width=2,
    )
    _save_pillow(image, path)


def _draw_scatter_pillow(rows: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1800, 1280
    left, right = 205, 1320
    top, bottom = 190, 1030
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(46, bold=True)
    label_font = _pillow_font(28, bold=True)
    small_font = _pillow_font(23)
    tick_font = _pillow_font(22)
    axis_color = (45, 52, 60)
    grid_color = (222, 226, 231)

    draw.text(
        (70, 40), "Answer Accuracy vs. Claim Support", font=title_font, fill=axis_color
    )
    draw.text(
        (70, 105),
        "Each point is a model/domain/perturbation aggregate.",
        font=small_font,
        fill=(95, 103, 112),
    )

    for tick in range(6):
        value = tick / 5
        x = left + round((right - left) * value)
        y = bottom - round((bottom - top) * value)
        draw.line((x, top, x, bottom), fill=grid_color, width=2)
        draw.line((left, y, right, y), fill=grid_color, width=2)
        _draw_centered(
            draw,
            (x - 42, bottom + 18, x + 42, bottom + 54),
            f"{value:.1f}",
            tick_font,
            axis_color,
        )
        _draw_centered(
            draw,
            (left - 86, y - 18, left - 18, y + 18),
            f"{value:.1f}",
            tick_font,
            axis_color,
        )

    draw.line((left, bottom, right, bottom), fill=axis_color, width=4)
    draw.line((left, bottom, left, top), fill=axis_color, width=4)
    _draw_centered(
        draw, (left, 1090, right, 1132), "Answer Accuracy", label_font, axis_color
    )
    draw.text((40, 130), "Claim Support", font=label_font, fill=axis_color)

    for row in rows:
        x_value = max(0.0, min(1.0, float(row["answer_accuracy"])))
        y_value = max(0.0, min(1.0, float(row["claim_faithfulness"])))
        x = left + round((right - left) * x_value)
        y = bottom - round((bottom - top) * y_value)
        color = point_color(str(row["perturbation"]))
        draw.ellipse(
            (x - 13, y - 13, x + 13, y + 13),
            fill=color,
            outline=(255, 255, 255),
            width=3,
        )

    legend_x, legend_y = 1410, 210
    draw.text(
        (legend_x, legend_y - 65), "Perturbation", font=label_font, fill=axis_color
    )
    for index, perturbation in enumerate(
        sorted({str(row["perturbation"]) for row in rows}, key=perturbation_sort_key)
    ):
        y = legend_y + index * 58
        color = point_color(perturbation)
        draw.ellipse((legend_x, y, legend_x + 26, y + 26), fill=color)
        draw.text(
            (legend_x + 42, y - 3),
            _pretty_label(perturbation).title(),
            font=small_font,
            fill=axis_color,
        )

    _save_pillow(image, path)


def _draw_perturbation_impact_pillow(rows: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    models = sorted({str(row["model"]) for row in rows})
    perturbations = [
        item
        for item in sorted(
            {str(row["perturbation"]) for row in rows}, key=perturbation_sort_key
        )
        if item != "clean"
    ]
    drops_by_model_perturbation: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        perturbation = str(row["perturbation"])
        if perturbation == "clean":
            continue
        drops_by_model_perturbation[(str(row["model"]), perturbation)].append(
            float(row["perturbation_drop"])
        )
    average_drop = {
        key: sum(values) / len(values)
        for key, values in drops_by_model_perturbation.items()
        if values
    }
    max_drop = max([0.1, *(max(0.0, value) for value in average_drop.values())])
    y_max = min(1.0, max(0.25, math.ceil(max_drop * 10) / 10))

    width, height = 1900, 1260
    left, right = 170, 1415
    top, bottom = 210, 975
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(46, bold=True)
    label_font = _pillow_font(28, bold=True)
    small_font = _pillow_font(23)
    tick_font = _pillow_font(21)
    axis_color = (45, 52, 60)
    grid_color = (222, 226, 231)
    palette = [(44, 123, 182), (26, 152, 80), (215, 48, 39), (116, 173, 209)]

    draw.text((70, 40), "Perturbation Impact", font=title_font, fill=axis_color)
    draw.text(
        (70, 105),
        "Average answer-accuracy drop from each model's clean setting.",
        font=small_font,
        fill=(95, 103, 112),
    )

    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - round((bottom - top) * value / y_max)
        draw.line((left, y, right, y), fill=grid_color, width=2)
        _draw_centered(
            draw, (48, y - 18, left - 18, y + 18), f"{value:.1f}", tick_font, axis_color
        )

    draw.line((left, bottom, right, bottom), fill=axis_color, width=4)
    draw.line((left, bottom, left, top), fill=axis_color, width=4)
    draw.text((42, 164), "Accuracy Drop", font=label_font, fill=axis_color)

    group_width = (right - left) / max(1, len(perturbations))
    bar_width = min(70, max(42, int((group_width - 72) / max(1, len(models)))))
    for group_index, perturbation in enumerate(perturbations):
        group_x = left + group_index * group_width
        _draw_multiline_centered(
            draw,
            (
                round(group_x),
                bottom + 26,
                round(group_x + group_width - 16),
                bottom + 112,
            ),
            _pretty_label(perturbation).title().replace(" ", "\n"),
            small_font,
            axis_color,
        )
        for model_index, model in enumerate(models):
            value = max(0.0, average_drop.get((model, perturbation), 0.0))
            bar_height = round((bottom - top) * value / y_max) if y_max else 0
            x0 = round(group_x + 36 + model_index * (bar_width + 16))
            y0 = bottom - bar_height
            color = palette[model_index % len(palette)]
            draw.rounded_rectangle(
                (x0, y0, x0 + bar_width, bottom), radius=8, fill=color
            )
            _draw_centered(
                draw,
                (x0 - 18, y0 - 42, x0 + bar_width + 18, y0 - 8),
                f"{value:.2f}",
                tick_font,
                axis_color,
            )

    legend_x, legend_y = 1490, 230
    draw.text((legend_x, legend_y - 70), "Model", font=label_font, fill=axis_color)
    for index, model in enumerate(models):
        y = legend_y + index * 62
        color = palette[index % len(palette)]
        draw.rounded_rectangle(
            (legend_x, y, legend_x + 34, y + 24), radius=5, fill=color
        )
        draw.text((legend_x + 50, y - 2), model, font=small_font, fill=axis_color)

    _save_pillow(image, path)


def draw_pillow_plots(rows: list[dict[str, Any]]) -> bool:
    try:
        import PIL  # noqa: F401
    except Exception:
        return False
    _draw_heatmap_pillow(
        rows, "answer_accuracy", "Answer Accuracy", ACCURACY_HEATMAP_PATH
    )
    _draw_heatmap_pillow(
        rows, "claim_faithfulness", "Claim Support", FAITHFULNESS_HEATMAP_PATH
    )
    _draw_scatter_pillow(rows, SCATTER_PATH)
    _draw_perturbation_impact_pillow(rows, PERTURBATION_IMPACT_PATH)
    return True


def draw_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    width = 860
    height = 620
    left = 90
    right = 790
    top = 80
    bottom = 520
    canvas = Canvas(width, height)
    canvas.text(40, 24, "ACCURACY VS FAITHFULNESS", scale=3)
    canvas.line(left, bottom, right, bottom, (40, 40, 40))
    canvas.line(left, bottom, left, top, (40, 40, 40))
    canvas.text(left - 8, bottom + 18, "0.0", scale=2)
    canvas.text(right - 32, bottom + 18, "1.0", scale=2)
    canvas.text(24, top - 10, "1.0", scale=2)
    canvas.text(24, bottom - 12, "0.0", scale=2)
    canvas.text(310, 560, "ANSWER ACCURACY", scale=2)
    canvas.text(12, 48, "CLAIM FAITHFULNESS", scale=1)

    for tick in range(6):
        x = left + round((right - left) * tick / 5)
        y = bottom - round((bottom - top) * tick / 5)
        canvas.line(x, bottom - 4, x, bottom + 4, (90, 90, 90))
        canvas.line(left - 4, y, left + 4, y, (90, 90, 90))

    for row in rows:
        x_value = max(0.0, min(1.0, float(row["answer_accuracy"])))
        y_value = max(0.0, min(1.0, float(row["claim_faithfulness"])))
        x = left + round((right - left) * x_value)
        y = bottom - round((bottom - top) * y_value)
        color = point_color(str(row["perturbation"]))
        canvas.rect(x - 5, y - 5, 11, 11, color)

    legend_x = 610
    legend_y = 94
    for index, perturbation in enumerate(
        sorted({str(row["perturbation"]) for row in rows}, key=perturbation_sort_key)
    ):
        y = legend_y + index * 24
        canvas.rect(legend_x, y, 12, 12, point_color(perturbation))
        canvas.text(legend_x + 20, y - 1, perturbation[:18], scale=1)
    canvas.save(path)


def draw_perturbation_impact(rows: list[dict[str, Any]], path: Path) -> None:
    models = sorted({str(row["model"]) for row in rows})
    perturbations = [
        item
        for item in sorted(
            {str(row["perturbation"]) for row in rows}, key=perturbation_sort_key
        )
        if item != "clean"
    ]
    drops_by_model_perturbation: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        perturbation = str(row["perturbation"])
        if perturbation == "clean":
            continue
        drops_by_model_perturbation[(str(row["model"]), perturbation)].append(
            float(row["perturbation_drop"])
        )

    average_drop = {
        key: sum(values) / len(values)
        for key, values in drops_by_model_perturbation.items()
        if values
    }
    max_drop = max([0.1, *(max(0.0, value) for value in average_drop.values())])
    y_max = min(1.0, max(0.25, math.ceil(max_drop * 10) / 10))

    width = 980
    height = 620
    left = 96
    right = 920
    top = 92
    bottom = 500
    canvas = Canvas(width, height)
    canvas.text(36, 24, "PERTURBATION IMPACT", scale=3)
    canvas.text(36, 58, "AVG ACCURACY DROP FROM CLEAN", scale=2)
    canvas.line(left, bottom, right, bottom, (40, 40, 40))
    canvas.line(left, bottom, left, top, (40, 40, 40))

    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - round((bottom - top) * value / y_max)
        canvas.line(left - 4, y, right, y, (220, 224, 228) if tick else (40, 40, 40))
        canvas.text(34, y - 6, f"{value:.1f}", scale=1)

    palette = [(44, 123, 182), (26, 152, 80), (215, 48, 39), (116, 173, 209)]
    group_width = (right - left) // max(1, len(perturbations))
    bar_width = max(14, min(34, (group_width - 34) // max(1, len(models))))
    for group_index, perturbation in enumerate(perturbations):
        group_x = left + group_index * group_width
        label = perturbation.replace("_", " ").upper()[:18]
        canvas.text(group_x + 6, bottom + 20, label, scale=1)
        for model_index, model in enumerate(models):
            value = max(0.0, average_drop.get((model, perturbation), 0.0))
            bar_height = round((bottom - top) * value / y_max) if y_max else 0
            x = group_x + 18 + model_index * (bar_width + 8)
            y = bottom - bar_height
            canvas.rect(
                x, y, bar_width, bar_height, palette[model_index % len(palette)]
            )
            canvas.text(x - 2, max(top, y - 18), f"{value:.2f}", scale=1)

    legend_x = 610
    legend_y = 28
    for index, model in enumerate(models):
        y = legend_y + index * 22
        canvas.rect(legend_x, y, 12, 12, palette[index % len(palette)])
        canvas.text(legend_x + 20, y - 1, model[:24], scale=1)

    canvas.save(path)


def generate_plots(rows: list[dict[str, Any]]) -> None:
    if draw_pillow_plots(rows):
        return
    draw_heatmap(rows, "answer_accuracy", "ANSWER ACCURACY", ACCURACY_HEATMAP_PATH)
    draw_heatmap(
        rows, "claim_faithfulness", "CLAIM FAITHFULNESS", FAITHFULNESS_HEATMAP_PATH
    )
    draw_scatter(rows, SCATTER_PATH)
    draw_perturbation_impact(rows, PERTURBATION_IMPACT_PATH)


def main() -> int:
    raw_outputs = read_jsonl(RAW_OUTPUTS_PATH)
    scored_outputs = read_jsonl(SCORED_OUTPUTS_PATH, allow_empty=True)
    rows = aggregate_metrics(raw_outputs, scored_outputs)
    if not rows:
        raise ValueError("no metric rows were produced")

    ci_rows = bootstrap_metric_cis(raw_outputs, scored_outputs)
    write_metrics_csv(METRICS_PATH, rows)
    write_metric_cis_csv(METRIC_CIS_PATH, ci_rows)
    generate_plots(rows)
    failure_examples = write_failure_examples(scored_outputs)
    update_paper_results(rows, failure_examples, ci_rows)
    print(f"wrote {len(rows)} metric rows to {METRICS_PATH.relative_to(ROOT_DIR)}")
    print(
        f"wrote {len(ci_rows)} confidence-interval rows to {METRIC_CIS_PATH.relative_to(ROOT_DIR)}"
    )
    print(f"wrote {ACCURACY_HEATMAP_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote {FAITHFULNESS_HEATMAP_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote {SCATTER_PATH.relative_to(ROOT_DIR)}")
    print(f"wrote {PERTURBATION_IMPACT_PATH.relative_to(ROOT_DIR)}")
    print(
        f"wrote {len(failure_examples)} failure examples to {FAILURE_EXAMPLES_PATH.relative_to(ROOT_DIR)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
