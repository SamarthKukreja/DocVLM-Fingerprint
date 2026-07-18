"""Minimal Streamlit inspection dashboard for DocVLM-Fingerprint outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_OUTPUTS_PATH = ROOT_DIR / "results" / "raw_outputs.jsonl"
SCORED_OUTPUTS_PATH = ROOT_DIR / "results" / "scored_outputs.jsonl"
METRICS_PATH = ROOT_DIR / "results" / "metrics.csv"
FIGURE_PATHS = [
    ROOT_DIR / "results" / "figures" / "accuracy_heatmap.png",
    ROOT_DIR / "results" / "figures" / "faithfulness_heatmap.png",
    ROOT_DIR / "results" / "figures" / "accuracy_vs_faithfulness.png",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    raw_outputs = read_jsonl(RAW_OUTPUTS_PATH)
    scored_outputs = read_jsonl(SCORED_OUTPUTS_PATH)
    metrics = read_csv(METRICS_PATH)
    claims_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored_outputs:
        claims_by_case[str(record.get("case_id", ""))].append(record)
    return raw_outputs, dict(claims_by_case), metrics


def case_label(record: dict[str, Any]) -> str:
    return f"{record.get('case_id')} | {record.get('domain')} | {record.get('model')}"


def main() -> None:
    st.set_page_config(page_title="DocVLM-Fingerprint", layout="wide")
    st.title("DocVLM-Fingerprint")
    st.caption("Document-centric VLM faithfulness stress-testing workflow")

    raw_outputs, claims_by_case, metrics = load_outputs()
    if not raw_outputs:
        st.error("No raw outputs found. Run `python src/evaluate.py` and `python src/metrics.py` first.")
        return

    with st.sidebar:
        st.header("Filters")
        domains = sorted({str(record.get("domain", "")) for record in raw_outputs})
        perturbations = sorted({str(record.get("perturbation", "")) for record in raw_outputs})
        domain = st.selectbox("Domain", ["all", *domains])
        perturbation = st.selectbox("Perturbation", ["all", *perturbations])

    filtered = [
        record
        for record in raw_outputs
        if (domain == "all" or record.get("domain") == domain)
        and (perturbation == "all" or record.get("perturbation") == perturbation)
    ]
    if not filtered:
        st.warning("No examples match the selected filters.")
        return

    selected = st.selectbox("Example", filtered, format_func=case_label)
    case_id = str(selected.get("case_id", ""))
    claim_records = claims_by_case.get(case_id, [])

    image_path = ROOT_DIR / str(selected.get("image_path", ""))
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Example")
        if image_path.exists():
            st.image(str(image_path), caption=str(selected.get("image_path", "")), use_container_width=True)
        else:
            st.warning(f"Image path not found: {image_path}")

    with right:
        st.subheader("Answer")
        st.write(f"**Domain:** {selected.get('domain')}")
        st.write(f"**Perturbation:** {selected.get('perturbation')}")
        st.write(f"**Question:** {selected.get('question')}")
        st.write(f"**Ground truth:** {selected.get('expected_answer')}")
        st.write(f"**Model answer:** {selected.get('answer')}")
        st.write(f"**Answer correct:** {selected.get('answer_correct')}")

    st.subheader("Claims And Labels")
    if claim_records:
        st.table(
            [
                {
                    "claim": record.get("claim"),
                    "label": record.get("label"),
                    "method": record.get("scoring_method"),
                    "evidence": "; ".join(str(item) for item in record.get("evidence_used", [])),
                }
                for record in claim_records
            ]
        )
    else:
        st.info("No scored claims found for this example.")

    st.subheader("Metrics")
    if metrics:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    else:
        st.info("No metrics CSV found.")

    st.subheader("Core Plots")
    plot_cols = st.columns(3)
    for column, path in zip(plot_cols, FIGURE_PATHS):
        with column:
            if path.exists():
                st.image(str(path), caption=path.name, use_container_width=True)
            else:
                st.warning(f"Missing {path.name}")


if __name__ == "__main__":
    main()
