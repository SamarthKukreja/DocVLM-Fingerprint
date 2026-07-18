"""Generate deterministic MVP visual perturbations for all annotated examples."""

from __future__ import annotations

import json
import random
import struct
import zlib
from pathlib import Path
from typing import Callable

from dataset import ANNOTATIONS_PATH, ROOT_DIR, load_annotations
from schema import validate_annotations


PERTURBED_DIR = ROOT_DIR / "data" / "perturbed"
METADATA_PATH = ROOT_DIR / "data" / "perturbation_metadata.jsonl"
SEED = 20260202
PERTURBATIONS = ("blur_downscale", "crop_removal", "jpeg_compression", "distractor_text")
SEVERITY = "medium"

Color = tuple[int, int, int]
ImageRows = list[bytearray]

WHITE: Color = (255, 255, 255)
BLACK: Color = (30, 34, 39)
RED: Color = (198, 55, 55)

FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
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
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "N": ["1001", "1101", "1011", "1001", "1001", "1001", "1001"],
    "O": ["010", "101", "101", "101", "101", "101", "010"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
}


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def read_png(path: Path) -> tuple[int, int, ImageRows]:
    """Read an 8-bit RGB/RGBA PNG into RGB rows."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")

    offset = 8
    width = height = bit_depth = color_type = None
    compressed_parts: list[bytes] = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if bit_depth != 8 or color_type not in {2, 6} or compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError(f"unsupported PNG format: {path}")
        elif kind == b"IDAT":
            compressed_parts.append(chunk_data)
        elif kind == b"IEND":
            break

    if width is None or height is None or color_type is None:
        raise ValueError(f"missing PNG header: {path}")

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(b"".join(compressed_parts))
    rows: ImageRows = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        current = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for idx in range(stride):
            left = current[idx - channels] if idx >= channels else 0
            above = previous[idx]
            upper_left = previous[idx - channels] if idx >= channels else 0
            if filter_type == 1:
                current[idx] = (current[idx] + left) & 0xFF
            elif filter_type == 2:
                current[idx] = (current[idx] + above) & 0xFF
            elif filter_type == 3:
                current[idx] = (current[idx] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                current[idx] = (current[idx] + paeth_predictor(left, above, upper_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        if channels == 4:
            rgb = bytearray()
            for idx in range(0, len(current), 4):
                rgb.extend(current[idx : idx + 3])
            rows.append(rgb)
        else:
            rows.append(current)
        previous = current
    return width, height, rows


def write_png(path: Path, width: int, height: int, rows: ImageRows) -> None:
    """Write RGB rows as an 8-bit PNG."""

    def chunk(kind: bytes, chunk_data: bytes) -> bytes:
        payload = kind + chunk_data
        return struct.pack(">I", len(chunk_data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, 9)),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def clone_rows(rows: ImageRows) -> ImageRows:
    return [bytearray(row) for row in rows]


def get_pixel(rows: ImageRows, width: int, x: int, y: int) -> Color:
    offset = x * 3
    row = rows[y]
    return row[offset], row[offset + 1], row[offset + 2]


def set_pixel(rows: ImageRows, width: int, x: int, y: int, color: Color) -> None:
    if 0 <= y < len(rows) and 0 <= x < width:
        offset = x * 3
        rows[y][offset : offset + 3] = bytes(color)


def fill_rect(rows: ImageRows, width: int, x: int, y: int, rect_width: int, rect_height: int, color: Color) -> None:
    for yy in range(max(0, y), min(len(rows), y + rect_height)):
        for xx in range(max(0, x), min(width, x + rect_width)):
            set_pixel(rows, width, xx, yy, color)


def draw_text(rows: ImageRows, width: int, x: int, y: int, text: str, color: Color, scale: int = 3) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT[" "])
        for gy, line in enumerate(glyph):
            for gx, bit in enumerate(line):
                if bit == "1":
                    fill_rect(rows, width, cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += (max(len(line) for line in glyph) + 1) * scale


def perturb_blur_downscale(width: int, height: int, rows: ImageRows, rng: random.Random) -> ImageRows:
    """Downscale by block averaging, then restore original dimensions."""
    block = 4
    small_width = max(1, width // block)
    small_height = max(1, height // block)
    small: list[list[Color]] = []
    for sy in range(small_height):
        small_row: list[Color] = []
        for sx in range(small_width):
            totals = [0, 0, 0]
            count = 0
            for y in range(sy * block, min(height, (sy + 1) * block)):
                for x in range(sx * block, min(width, (sx + 1) * block)):
                    pixel = get_pixel(rows, width, x, y)
                    totals[0] += pixel[0]
                    totals[1] += pixel[1]
                    totals[2] += pixel[2]
                    count += 1
            small_row.append((totals[0] // count, totals[1] // count, totals[2] // count))
        small.append(small_row)

    output = clone_rows(rows)
    for y in range(height):
        for x in range(width):
            set_pixel(output, width, x, y, small[min(small_height - 1, y // block)][min(small_width - 1, x // block)])
    return output


def perturb_crop_removal(width: int, height: int, rows: ImageRows, rng: random.Random) -> ImageRows:
    """Remove a deterministic rectangular region while preserving canvas size."""
    output = clone_rows(rows)
    rect_width = width // 3
    rect_height = height // 4
    x = rng.randint(width // 5, width - rect_width - width // 8)
    y = rng.randint(height // 5, height - rect_height - height // 8)
    fill_rect(output, width, x, y, rect_width, rect_height, WHITE)
    fill_rect(output, width, x, y, rect_width, 3, BLACK)
    fill_rect(output, width, x, y + rect_height - 3, rect_width, 3, BLACK)
    fill_rect(output, width, x, y, 3, rect_height, BLACK)
    fill_rect(output, width, x + rect_width - 3, y, 3, rect_height, BLACK)
    draw_text(output, width, x + 14, y + 14, "REMOVED", BLACK, 2)
    return output


def perturb_jpeg_compression(width: int, height: int, rows: ImageRows, rng: random.Random) -> ImageRows:
    """Approximate medium JPEG compression artifacts with block averaging and color quantization."""
    output = clone_rows(rows)
    block = 8
    quant = 32
    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            totals = [0, 0, 0]
            count = 0
            for y in range(y0, min(height, y0 + block)):
                for x in range(x0, min(width, x0 + block)):
                    pixel = get_pixel(rows, width, x, y)
                    totals[0] += pixel[0]
                    totals[1] += pixel[1]
                    totals[2] += pixel[2]
                    count += 1
            color = tuple(max(0, min(255, round((channel // count) / quant) * quant)) for channel in totals)
            for y in range(y0, min(height, y0 + block)):
                for x in range(x0, min(width, x0 + block)):
                    set_pixel(output, width, x, y, color)  # type: ignore[arg-type]
    return output


def perturb_distractor_text(width: int, height: int, rows: ImageRows, rng: random.Random) -> ImageRows:
    """Overlay deterministic irrelevant text without changing image dimensions."""
    output = clone_rows(rows)
    label = f"NOTE {rng.randint(10, 99)}"
    x = rng.randint(width // 2, width - 170)
    y = rng.randint(70, height - 90)
    fill_rect(output, width, x - 10, y - 10, 150, 46, (255, 245, 245))
    draw_text(output, width, x, y, label, RED, 3)
    return output


PERTURBATION_FUNCS: dict[str, Callable[[int, int, ImageRows, random.Random], ImageRows]] = {
    "blur_downscale": perturb_blur_downscale,
    "crop_removal": perturb_crop_removal,
    "jpeg_compression": perturb_jpeg_compression,
    "distractor_text": perturb_distractor_text,
}


def generate_perturbations() -> list[dict[str, str]]:
    records = load_annotations(ANNOTATIONS_PATH)
    validate_annotations(records, ROOT_DIR)
    PERTURBED_DIR.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, str]] = []
    output_paths: set[str] = set()
    for record in records:
        input_path = ROOT_DIR / record["image_path"]
        width, height, rows = read_png(input_path)
        for perturbation in PERTURBATIONS:
            rng = random.Random(f"{SEED}:{record['id']}:{perturbation}")
            output_relative = f"data/perturbed/{record['id']}_{perturbation}.png"
            if output_relative in output_paths:
                raise ValueError(f"duplicate perturbation output path: {output_relative}")
            output_paths.add(output_relative)
            output_path = ROOT_DIR / output_relative
            output_rows = PERTURBATION_FUNCS[perturbation](width, height, rows, rng)
            write_png(output_path, width, height, output_rows)
            if not output_path.exists():
                raise ValueError(f"perturbation output was not written: {output_relative}")
            metadata.append(
                {
                    "example_id": record["id"],
                    "domain": record["domain"],
                    "perturbation": perturbation,
                    "severity": SEVERITY,
                    "input_path": record["image_path"],
                    "output_path": output_relative,
                }
            )

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    generated_files = sorted(PERTURBED_DIR.glob("*.png"))
    if len(metadata) != len(generated_files):
        raise ValueError(f"metadata count {len(metadata)} does not match generated file count {len(generated_files)}")
    if len(metadata) != len(records) * len(PERTURBATIONS):
        raise ValueError("unexpected perturbation metadata count")
    return metadata


def main() -> int:
    metadata = generate_perturbations()
    print(f"wrote {len(metadata)} perturbed images to {PERTURBED_DIR.relative_to(ROOT_DIR)}")
    print(f"wrote metadata to {METADATA_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

