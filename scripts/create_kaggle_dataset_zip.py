"""Create the local Kaggle dataset zip for DocVLM-Fingerprint.

The generated zip is intentionally ignored by git. Upload it to Kaggle as a
private dataset when rerunning the open-VLM notebook.
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kaggle" / "docvlm-fingerprint-kaggle-dataset.zip"
TOP_LEVEL_DIR = "docvlm-fingerprint"

EXCLUDED_DIRS = {
    ".git",
    ".agents",
    ".cache",
    ".streamlit",
    ".pytest_cache",
    "__pycache__",
    "tmp",
    "htmlcov",
    "venv",
    ".venv",
    "env",
}

EXCLUDED_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.log",
    "*.out",
    "*.synctex.gz",
    "*.toc",
    ".env",
    ".env.*",
    "kaggle/*.zip",
    "results/*.jsonl",
    "results/*.csv",
    "results/cache/*",
    "results/figures/*",
    "results/per_model/*",
    "results/ablations/*",
    "results/external/*",
    "data/external/chartqa/*",
}

ALLOWED_ENV_FILES = {".env.example"}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
        return True
    if path.name in ALLOWED_ENV_FILES:
        return False
    return any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDED_PATTERNS)


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not should_skip(path):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def create_zip(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    files = iter_files()
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{TOP_LEVEL_DIR}/{rel}")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = create_zip(args.output.resolve())
    print(f"wrote {args.output} with {count} files under {TOP_LEVEL_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
