# DocVLM-Fingerprint Project Page

HTML version: `docs/index.html` for GitHub Pages.

## One-Line Summary

DocVLM-Fingerprint is a reproducible evaluation workflow for testing how document-centric vision-language models behave under visual perturbations and claim-level faithfulness scoring.

## Why It Matters

Vision-language models can look reliable on clean document images but fail sharply when images are compressed, blurred, cropped, or visually distracted. This project makes those failures inspectable by pairing answer accuracy with claim support labels, plots, and concrete failure examples.

## What I Built

- A deterministic generated dataset with 120 chart, OCR-document, and scientific-figure examples.
- Four perturbation families for every example: blur/downscale, crop/removal, JPEG compression, and distractor text.
- A plug-in model interface for local or API-backed VLM clients, with a custom-model config example for rerunning the pipeline on another model.
- Claim extraction and binary evidence scoring with cacheable, auditable outputs.
- Metrics, bootstrap confidence intervals, an audit-ready worksheet, descriptive ablation tables, heatmaps, failure examples, a Streamlit inspection dashboard, and a pdf report.
- A Kaggle notebook for running open VLMs on GPU and exporting reproducible result artifacts.
- A small CharXiv real-chart perturbation supplement with local-only images that stays separate from the main generated-dataset metrics.

## Demonstration Run

The checked-in full run evaluates `qwen3_vl_8b`, `internvl35_8b_hf`, and `qwen3_vl_4b` over 120 generated examples and four perturbations per example. This gives 600 evaluated cases per model and 1800 VLM calls total. The run contains two independent model families plus a Qwen 4B/8B scale-family comparison. Separate real-data checks add 90 clean ChartQA calls, 90 clean DocumentVQA calls, and a 20-example CharXiv real-chart perturbation supplement for the Qwen scale comparison without changing the main generated-dataset metrics.

The main finding is intentionally narrow: the evaluated models perform strongly on clean inputs, but JPEG compression and blur/downscale cause severe degradation. The run is a controlled systems/evaluation case study, not a general-purpose benchmark or ranking table.

## Reproducible Artifacts

- `results/raw_outputs.jsonl`: model answers
- `results/claim_outputs.jsonl`: extracted answer claims
- `results/scored_outputs.jsonl`: claim support labels
- `results/metrics.csv`: aggregated metrics
- `results/metric_cis_by_perturbation.csv`: perturbation-level bootstrap confidence intervals
- `docs/manual_audit.md`: 30-row manual-audit worksheet; `docs/manual_audit_summary.md` is generated only after review
- `results/figures/`: generated plots, including the perturbation-impact bar chart
- `results/failure_examples.jsonl`: selected unsupported-claim examples
- `results/ablations/`: descriptive ablation tables from existing outputs
- `results/external/chartqa/`: completed clean-only ChartQA sanity-check outputs
- `results/external/documentvqa/`: completed clean-only DocumentVQA sanity-check outputs
- `data/external/charxiv/` and `results/external/charxiv/`: curated real-chart perturbation supplement and Qwen-only outputs
- `report/docvlm_fingerprint_paper.pdf`: concise paper-style report
- `dashboard/app.py`: local inspection UI

## Engineering Signal

The project emphasizes maintainable ML evaluation infrastructure: deterministic data generation, stable schemas, model-client abstraction, custom config based model selection, cache-aware execution, standard-library tests, reproducible Kaggle execution, bootstrap uncertainty summaries, audit guardrails, external-data import/evaluation, descriptive ablations, explicit limitations, and artifacts that can be inspected without rerunning expensive VLM inference.

## Limitations

The main dataset is synthetic and small, the checked-in full run includes three model entries but only two independent model families, and the binary scorer is intentionally simple. These choices keep the project reproducible and auditable, but they limit broad claims. Bootstrap intervals are included as descriptive summaries. The ChartQA, DocumentVQA, and CharXiv runs are external checks rather than benchmark results, and CharXiv is Qwen-only. The 30-row manual-audit worksheet is prepared for review after a fresh run. Strong future extensions would add independent multi-annotator review, larger public real-world document slices, and more independent open-model families.
