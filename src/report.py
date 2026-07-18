"""Helpers for selecting readable Day 6 failure examples and updating the report."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset import ROOT_DIR


FAILURE_EXAMPLES_PATH = ROOT_DIR / "results" / "failure_examples.jsonl"
PAPER_PATH = ROOT_DIR / "report" / "paper.tex"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
REPORT_FIGURES_DIR = ROOT_DIR / "report" / "figures"
REPORT_FIGURE_FILES = (
    "accuracy_heatmap.png",
    "faithfulness_heatmap.png",
    "accuracy_vs_faithfulness.png",
    "perturbation_impact.png",
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def sync_report_figures() -> None:
    """Copy generated result plots into report/figures for LaTeX builds."""
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for filename in REPORT_FIGURE_FILES:
        source = RESULTS_FIGURES_DIR / filename
        if source.exists():
            shutil.copy2(source, REPORT_FIGURES_DIR / filename)


def _priority(record: dict[str, Any]) -> tuple[int, int, str]:
    domain_rank = {"chart": 0, "ocr_doc": 1, "scientific_figure": 2}.get(str(record.get("domain")), 9)
    perturbation = str(record.get("perturbation", "clean"))
    perturb_rank = 0 if perturbation == "distractor_text" else 1 if perturbation != "clean" else 2
    return domain_rank, perturb_rank, str(record.get("case_id", ""))


def _explanation(labels: list[str], answer_correct: bool, perturbation: str) -> str:
    supported = labels.count("supported")
    unsupported = labels.count("unsupported")
    if answer_correct and unsupported:
        return "Answer marked correct but at least one extracted claim is unsupported by the evidence."
    if not answer_correct and supported:
        return "Answer marked incorrect while at least one claim still matches the evidence."
    if perturbation != "clean" and unsupported:
        return "Perturbed input produced unsupported claim text under the binary scorer."
    return "Model answer or claim text is not supported by the annotation evidence."


def select_failure_examples(scored_outputs: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Select a compact, domain-diverse set of unsupported-claim examples."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored_outputs:
        if record.get("label") == "unsupported":
            grouped[str(record.get("case_id", ""))].append(record)

    candidates: list[dict[str, Any]] = []
    for case_id, records in grouped.items():
        first = records[0]
        labels = [str(record.get("label")) for record in records]
        candidates.append(
            {
                "case_id": case_id,
                "example_id": first.get("example_id", ""),
                "domain": first.get("domain", ""),
                "perturbation": first.get("perturbation", "clean"),
                "model": first.get("model", ""),
                "image_path": first.get("image_path", ""),
                "question": first.get("question", ""),
                "ground_truth_answer": first.get("expected_answer", ""),
                "model_answer": first.get("answer", ""),
                "claims": [record.get("claim", "") for record in records],
                "support_labels": labels,
                "explanation": _explanation(labels, bool(first.get("answer_correct")), str(first.get("perturbation"))),
            }
        )

    candidates.sort(key=_priority)
    selected: list[dict[str, Any]] = []
    for domain in ["chart", "ocr_doc", "scientific_figure"]:
        match = next((item for item in candidates if item["domain"] == domain and item not in selected), None)
        if match is not None:
            selected.append(match)

    distractor = next(
        (item for item in candidates if item["perturbation"] == "distractor_text" and item not in selected),
        None,
    )
    if distractor is not None:
        selected.append(distractor)

    for item in candidates:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected[:limit]


def write_failure_examples(scored_outputs: list[dict[str, Any]], path: Path = FAILURE_EXAMPLES_PATH) -> list[dict[str, Any]]:
    examples = select_failure_examples(scored_outputs)
    write_jsonl(path, examples)
    return examples


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }.items():
        text = text.replace(old, new)
    return text


def domain_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in read_jsonl(ROOT_DIR / "data" / "annotations.jsonl"):
        counts[str(record.get("domain", "unknown"))] += 1
    return dict(counts)


def perturbation_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in read_jsonl(ROOT_DIR / "data" / "perturbation_metadata.jsonl"):
        counts[str(record.get("domain", "unknown"))] += 1
    return dict(counts)


def metric_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llllrrr}",
        r"\hline",
        r"\textbf{Model} & \textbf{Domain} & \textbf{Perturbation} & \textbf{N} & \textbf{Acc.} & \textbf{Faith.} & \textbf{Halluc.} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{Model} & \textbf{Domain} & \textbf{Perturbation} & \textbf{N} & \textbf{Acc.} & \textbf{Faith.} & \textbf{Halluc.} \\",
        r"\hline",
        r"\endhead",
    ]
    for row in rows:
        line = "{} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f}".format(
            latex_escape(row["model"]),
            latex_escape(row["domain"]),
            latex_escape(row["perturbation"]),
            int(row["total_answers"]),
            float(row["answer_accuracy"]),
            float(row["claim_faithfulness"]),
            float(row["hallucination_rate"]),
        )
        lines.append(line + r" \\")
    lines.extend([r"\hline", r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def ci_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No bootstrap confidence interval rows were generated."
    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"\textbf{Perturbation} & \textbf{Acc. 95\% CI} & \textbf{Faith. 95\% CI} & \textbf{Halluc. 95\% CI} \\",
        r"\hline",
    ]
    for row in rows:
        line = "{} & {:.2f} to {:.2f} & {:.2f} to {:.2f} & {:.2f} to {:.2f}".format(
            latex_escape(row["perturbation"]),
            float(row["answer_accuracy_ci_low"]),
            float(row["answer_accuracy_ci_high"]),
            float(row["claim_faithfulness_ci_low"]),
            float(row["claim_faithfulness_ci_high"]),
            float(row["hallucination_rate_ci_low"]),
            float(row["hallucination_rate_ci_high"]),
        )
        lines.append(line + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)




def _average(rows: list[dict[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _model_perturbation_rows(metric_rows: list[dict[str, Any]], model: str, perturbation: str) -> list[dict[str, Any]]:
    return [row for row in metric_rows if str(row["model"]) == model and str(row["perturbation"]) == perturbation]


def _perturbation_rows(metric_rows: list[dict[str, Any]], perturbation: str) -> list[dict[str, Any]]:
    return [row for row in metric_rows if str(row["perturbation"]) == perturbation]


def prose_label(value: object) -> str:
    return str(value).replace("_", " ")


def key_findings_text(metric_rows: list[dict[str, Any]]) -> str:
    models = sorted({str(row["model"]) for row in metric_rows})
    clean_parts = []
    for model in models:
        clean_rows = _model_perturbation_rows(metric_rows, model, "clean")
        clean_parts.append(
            "{}: {:.3f} answer accuracy / {:.3f} claim support".format(
                latex_escape(prose_label(model)),
                _average(clean_rows, "answer_accuracy"),
                _average(clean_rows, "claim_faithfulness"),
            )
        )
    perturbation_summaries = []
    for perturbation in ["jpeg_compression", "blur_downscale", "crop_removal", "distractor_text"]:
        rows = _perturbation_rows(metric_rows, perturbation)
        if rows:
            perturbation_summaries.append(
                (
                    _average(rows, "answer_accuracy"),
                    perturbation,
                    _average(rows, "claim_faithfulness"),
                    _average(rows, "hallucination_rate"),
                )
            )
    perturbation_summaries.sort(key=lambda item: item[0])
    worst = perturbation_summaries[:2]
    moderate = perturbation_summaries[2:]
    lines = [
        "Clean generated examples are mostly solved by the evaluated open VLMs. Average clean performance across the three domains is: {}.".format(
            "; ".join(clean_parts)
        )
    ]
    if worst:
        worst_text = "; ".join(
            "{} with {:.3f} answer accuracy, {:.3f} claim support, and {:.3f} hallucination rate".format(
                latex_escape(prose_label(perturbation)), accuracy, faithfulness, hallucination
            )
            for accuracy, perturbation, faithfulness, hallucination in worst
        )
        lines.append(
            "The strongest perturbation failures are {}. These values describe this run only; they should not be read as population guarantees.".format(
                worst_text
            )
        )
    if moderate:
        moderate_text = "; ".join(
            "{} averages {:.3f} accuracy and {:.3f} claim support".format(
                latex_escape(prose_label(perturbation)), accuracy, faithfulness
            )
            for accuracy, perturbation, faithfulness, _hallucination in moderate
        )
        lines.append("The other perturbations are milder in this run: {}.".format(moderate_text))
    return "\n\n".join(lines)


def scale_family_text(metric_rows: list[dict[str, Any]]) -> str:
    models = {str(row["model"]) for row in metric_rows}
    if {"qwen3_vl_4b", "qwen3_vl_8b"}.issubset(models):
        qwen4_clean = _model_perturbation_rows(metric_rows, "qwen3_vl_4b", "clean")
        qwen8_clean = _model_perturbation_rows(metric_rows, "qwen3_vl_8b", "clean")
        return (
            "The run includes a Qwen3-VL size comparison. "
            "Average clean answer accuracy is {:.3f} for qwen3 vl 4b and {:.3f} for qwen3 vl 8b; "
            "average clean claim support is {:.3f} and {:.3f}, respectively. "
            "This is useful for checking size sensitivity within one family, although it is not a third independent model family."
        ).format(
            _average(qwen4_clean, "answer_accuracy"),
            _average(qwen8_clean, "answer_accuracy"),
            _average(qwen4_clean, "claim_faithfulness"),
            _average(qwen8_clean, "claim_faithfulness"),
        )
    return (
        "The Kaggle notebook is prepared for a balanced third model extension with qwen3 vl 4b, "
        "which enables a Qwen 4B/8B size comparison after the new run is imported. Until those results are present in "
        "results/metrics.csv, the paper should not describe a completed three model result."
    )


def model_count_phrase(metric_rows: list[dict[str, Any]]) -> str:
    count = len({str(row["model"]) for row in metric_rows})
    return "one open model" if count == 1 else f"{count} open model entries"

def dataset_table() -> str:
    counts = domain_counts()
    perturbed = perturbation_counts()
    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"\textbf{Domain} & \textbf{Clean examples} & \textbf{Perturbed variants} \\",
        r"\hline",
    ]
    for domain in ["chart", "ocr_doc", "scientific_figure"]:
        line = "{} & {} & {}".format(latex_escape(domain), counts.get(domain, 0), perturbed.get(domain, 0))
        lines.append(line + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)


def failure_text(examples: list[dict[str, Any]], limit: int = 3) -> str:
    if not examples:
        return "No failure examples were selected by the helper."
    lines = [r"\begin{enumerate}"]
    for example in examples[:limit]:
        claims = "; ".join(str(claim) for claim in example.get("claims", [])[:2])
        lines.append(
            "\\item {} / {} / {}: question: {}; answer: {}; claims: {}. {}".format(
                latex_escape(example.get("example_id", "")),
                latex_escape(example.get("domain", "")),
                latex_escape(example.get("perturbation", "")),
                latex_escape(example.get("question", "")),
                latex_escape(example.get("model_answer", "")),
                latex_escape(claims),
                latex_escape(example.get("explanation", "")),
            )
        )
    lines.append(r"\end{enumerate}")
    return "\n".join(lines)


def update_paper_results(
    metric_rows: list[dict[str, Any]],
    failure_examples: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]] | None = None,
    paper_path: Path = PAPER_PATH,
) -> None:
    """Refresh paper Results/Failure Analysis sections with generated Day 6 outputs."""
    sync_report_figures()
    text = paper_path.read_text(encoding="utf-8")
    double_start = text.find(r"\\section{Results}")
    start = double_start if double_start != -1 else text.find(r"\section{Results}")
    end = text.find(r"\section{Conclusion}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("paper does not contain replaceable Results and Conclusion sections")

    models = sorted({str(row["model"]) for row in metric_rows})
    total_examples = sum(domain_counts().values())
    replacement = f"""\\section{{Results}}

The generated evaluation set contains {total_examples} clean examples across chart, OCR document, and scientific figure domains. The Kaggle run evaluates three model entries: {latex_escape(', '.join(prose_label(model) for model in models))}. Each model is run on the clean examples and all four perturbation variants. The parser recovered final answers for all 1800 calls; 1764 calls included explicit Evidence lines, with most noncompliant generations occurring on heavily degraded chart inputs.

Answer accuracy can hide unsupported visual claims, so the analysis also reports claim support.

{dataset_table()}

\\subsection{{Main Metrics}}

{metric_table(metric_rows)}

Bootstrap intervals use case level resampling within each perturbation and a fixed seed. The full table is written to results/metric\\_cis\\_by\\_perturbation.csv.

{ci_table(ci_rows or [])}

\\subsection{{Key Findings}}

{key_findings_text(metric_rows)}

\\subsection{{Clean Baseline}}

The clean rows provide the baseline across charts, OCR documents, and scientific figures. The models handle these controlled inputs well. That makes the perturbation drops easier to interpret: the large failures come from specific visual degradations, not from a task format that was already unsolved.

\\subsection{{Perturbation Robustness}}

The perturbation rows compare each domain with its clean setting under the same four transformations. JPEG compression and blur/downscale produce the largest drops, often reducing strong clean performance to very low accuracy and claim support. Crop/removal causes moderate damage. Distractor text is less disruptive in this generated set. I report these as observed differences, not as causal estimates about model internals.

\\subsection{{Accuracy vs. Claim Support}}

The scatter plot checks whether final answer correctness and claim support separate. The two-line answer and evidence protocol scores the final answer separately from the supporting visual claim. In this run, the strongest separation is not correct answers with unsupported evidence; it is 113 claim rows where the final answer is incorrect while the extracted evidence claim is still supported. This makes visible partial visual grounding that answer accuracy alone would hide.

\\subsection{{Qwen Size Comparison}}

{scale_family_text(metric_rows)}

\\subsection{{External Real Data Checks}}

The external data checks are kept separate from the generated set. A 30 example HuggingFaceM4/ChartQA slice evaluates clean chart question answering for all three model entries, giving 90 additional VLM calls under results/external/chartqa. Exact-match answer accuracies are 0.600 for qwen3 vl 8b, 0.200 for internvl35 8b hf, and 0.500 for qwen3 vl 4b. A 30 example HuggingFaceM4/DocumentVQA validation slice evaluates clean OCR document question answering for the same three model entries, giving 90 additional VLM calls under results/external/documentvqa. Answer accuracies are 0.667 for qwen3 vl 8b, 0.633 for internvl35 8b hf, and 0.633 for qwen3 vl 4b. A separate CharXiv supplement adds 20 real arXiv chart examples with the same four perturbations, evaluated for qwen3 vl 4b and qwen3 vl 8b only under results/external/charxiv. Clean CharXiv accuracy is 0.400 and 0.450, respectively; jpeg compression drops both to 0.100, and blur downscale reaches 0.150 and 0.250. CharXiv is the most direct external perturbation transfer check, while ChartQA and DocumentVQA are clean only. These external numbers are small checks; they are not merged with the generated perturbation study.

\\subsection{{Descriptive Ablations}}

The ablation helper writes descriptive tables under results/ablations without additional VLM calls. The tables summarize answer and claim disagreement counts, scorer method usage by label, perturbation effect sizes, and no claim cases. They help interpret the run; they are not proposed as a new metric.

\\subsection{{Failure Examples}}

{failure_text(failure_examples)}

The generated plots are stored in results/figures and copied into report/figures for paper builds. Figure~\\ref{{fig:accuracy-heatmap}} shows answer accuracy, Figure~\\ref{{fig:faithfulness-heatmap}} shows claim support, Figure~\\ref{{fig:accuracy-faithfulness}} compares the two metrics, and Figure~\\ref{{fig:perturbation-impact}} summarizes average accuracy drop from clean inputs.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\linewidth]{{accuracy_heatmap.png}}
\\caption{{Answer accuracy by model, domain, and perturbation.}}
\\label{{fig:accuracy-heatmap}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\linewidth]{{faithfulness_heatmap.png}}
\\caption{{Claim support by model, domain, and perturbation.}}
\\label{{fig:faithfulness-heatmap}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.8\\linewidth]{{accuracy_vs_faithfulness.png}}
\\caption{{Relationship between answer accuracy and claim support.}}
\\label{{fig:accuracy-faithfulness}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\linewidth]{{perturbation_impact.png}}
\\caption{{Average answer accuracy drop from the clean setting, grouped by perturbation and model entry.}}
\\label{{fig:perturbation-impact}}
\\end{{figure}}

\\clearpage

\\section{{Failure Analysis}}

The selected failure examples focus on unsupported claims found by the binary scorer. The manual audit worksheet is prepared for post-run review and should be summarized only after the rows are manually checked. The examples make these records easy to inspect: a chart distractor case where the model follows a distractor NYB label instead of HYB, an OCR receipt case where it reads 52 instead of 50, and a scientific figure case where it selects BASE instead of FULL. They are debugging examples, not a taxonomy of VLM failures.

\\section{{Limitations}}

The study is intentionally small. The main run covers all three generated domains for {model_count_phrase(metric_rows)}, but only two independent model families are represented because Qwen 4B and 8B form a size comparison. The generated data improves control and reproducibility, but it limits what can be concluded about real documents. Most questions are single-hop short-answer lookups, so answer accuracy and claim support move together more often than they would for long rationales. Some generations also failed to follow the requested Evidence line format, which is reported as a protocol-compliance limitation rather than hidden. The binary scorer is transparent and testable, although it can miss paraphrases or ambiguous visual evidence. The unsupported label means a claim is not entailed by the annotated evidence string, so hallucination rate is an upper bound on fabrication under this scorer. The ChartQA, DocumentVQA, and CharXiv slices add real data evidence, but they are small and separate from the generated perturbation study; CharXiv is also Qwen-only. The simulated JPEG perturbation can remove small text entirely, so near-zero scores should be read as a missing-information floor condition rather than calibrated compression robustness. Future work should add independent multi annotator review and more visual changes, including rotations, layout shifts, lighting variation, watermarks, and adversarial text.
"""
    paper_path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")




