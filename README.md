# DocVLM-Fingerprint

DocVLM-Fingerprint is a compact evaluation framework for diagnosing when vision-language models produce document answers and whether the supporting visual evidence is grounded.

It packages task curation, deterministic visual perturbations, plug-in model evaluation, answer accuracy, claim-level faithfulness scoring, metrics, plots, and failure analysis into a small document-centric workflow. The included model comparisons are demonstration case studies. The primary artifact is a reusable evaluation framework for testing additional VLMs through the same data, perturbation, scoring, and metrics pipeline.

## At A Glance

- **What it is:** a reproducible document-VQA evaluation workflow for charts, OCR-heavy documents, and scientific figures.
- **Main run:** 3 open model entries, 120 generated examples, 4 perturbations per example, 1800 VLM calls, and checked-in JSONL/CSV/PNG outputs.
- **Main finding:** clean performance is strong, but JPEG compression and blur/downscale cause severe drops; the two-line `Answer:`/`Evidence:` protocol also reveals 113 claim rows where the final answer is wrong while the evidence claim is still grounded.
- **External checks:** small clean-only ChartQA and DocumentVQA slices plus a Qwen-only CharXiv real-chart perturbation supplement, all kept separate from the main generated-dataset metrics.
- **Reproduce:** run local validation with `python -B -m unittest discover -s tests`, then use `kaggle/docvlm_fingerprint_open_vlm_case_study.ipynb` for GPU model evaluation.
- **Scope:** this is a controlled systems/evaluation case study, not a broad benchmark, not a leaderboard, and not a new faithfulness metric.

## Reviewer Summary

In one minute: this project demonstrates reproducible ML evaluation systems engineering for document-centric VLM reliability. It builds a controlled dataset, applies deterministic visual stress tests, evaluates open VLMs through a shared client interface, scores answer correctness and grounded claim support, and produces auditable JSONL/CSV artifacts, plots, failure examples, a dashboard, and a paper-style report.

Resume framing: built a reproducible document-centric VLM robustness and faithfulness evaluation workflow with deterministic perturbations, plug-in model clients, claim-level scoring, Kaggle GPU execution, bootstrap CIs, audit-ready failure analysis, and ablation artifacts over 1800 checked-in open-VLM calls.

## What This Is

DocVLM-Fingerprint is not a new benchmark, not a new faithfulness metric, and not a ranking of proprietary VLMs. It is a reproducible systems/evaluation project focused on document-centric hallucination robustness.

Answer accuracy is insufficient because a model can produce the right short answer while adding unsupported visual reasoning, or it can give a wrong final answer while some extracted claims remain grounded. DocVLM-Fingerprint reports answer accuracy alongside claim faithfulness and hallucination rate so these behaviors are visible together.

The name `Fingerprint` refers to failure-mode and faithfulness fingerprints: compact profiles of model weakness under perturbation and claim scoring. It does not mean model-ownership or identity fingerprinting.

## Quick Links For Reviewers

- Project page: `docs/index.html` for GitHub Pages; text summary: `docs/project_page.md`
- Paper : `report/docvlm_fingerprint_paper.pdf`
- Paper build notes: `report/README.md`
- Metrics: `results/metrics.csv`
- Bootstrap CIs: `results/metric_cis_by_perturbation.csv`
- Manual audit worksheet: `docs/manual_audit.md`; summary helper writes `docs/manual_audit_summary.md` after review
- Manual audit summary helper: `src/audit_summary.py`
- External data importer/evaluator: `src/import_chartqa_slice.py`, `src/evaluate_external_chartqa.py`
- Custom model config example: `configs/custom_model.example.yaml`
- CharXiv real-chart supplement: `data/external/charxiv/`, `results/external/charxiv/`
- Failure examples: `results/failure_examples.jsonl`
- Ablations: `results/ablations/`
- Kaggle GPU notebook: `kaggle/docvlm_fingerprint_open_vlm_case_study.ipynb`
- Dashboard: `dashboard/app.py`

## Current Artifact

- 120 generated examples across 3 domains: charts/tables, OCR-heavy documents, and scientific figures/pages.
- 4 deterministic perturbations: blur/downscale, crop/removal, JPEG-style compression artifacts, and distractor text.
- Checked-in real open-VLM results for `qwen3_vl_8b`, `internvl35_8b_hf`, and `qwen3_vl_4b` across all 120 examples and 4 perturbations each; this is two independent model families plus a Qwen 4B/8B scale-family comparison.
- Binary claim labels only: `supported` and `unsupported`.
- Outputs: raw answers, extracted claims, scored claims, metrics CSV, bootstrap confidence intervals, manual audit worksheet, ablation tables, three plots, failure examples, dashboard, LaTeX report, and separate external-data checks.

## Pipeline

1. Generate deterministic chart, OCR document, and scientific figure/page examples.
2. Validate annotation schema and image paths.
3. Generate four perturbations for every example.
4. Evaluate configured models through a shared `answer(image_path, question)` interface.
5. Split model answers into atomic claims.
6. Score claims against annotation evidence as `supported` or `unsupported`.
7. Aggregate answer accuracy, claim faithfulness, hallucination rate, and perturbation drop.
8. Inspect examples, claims, labels, and plots in the dashboard.

## Results Snapshot

The checked-in run is a real open-VLM evaluation from Kaggle. It evaluates three model entries, `qwen3_vl_8b`, `internvl35_8b_hf`, and `qwen3_vl_4b`, over the full generated dataset: 120 clean examples plus four perturbations per example, for 600 cases per model and 1800 VLM calls total. This gives two independent model families plus a Qwen 4B/8B scale-family comparison, supporting a controlled systems/evaluation case study rather than a general-purpose VLM benchmark. The full run uses a two-line `Answer:`/`Evidence:` protocol, shows high clean performance, and shows severe degradation under JPEG compression and blur/downscale. The parser recovered final answers for all 1800 calls; 1764 calls included explicit `Evidence:` lines.

| Model            | Domain            | Perturbation     | Answer accuracy | Claim faithfulness | Hallucination rate |
| ---------------- | ----------------- | ---------------- | --------------: | -----------------: | -----------------: |
| qwen3_vl_8b      | chart             | clean            |          1.0000 |             1.0000 |             0.0000 |
| qwen3_vl_8b      | chart             | jpeg_compression |          0.0000 |             0.0500 |             0.9500 |
| qwen3_vl_8b      | ocr_doc           | clean            |          0.9750 |             1.0000 |             0.0000 |
| qwen3_vl_8b      | scientific_figure | clean            |          0.9750 |             0.9750 |             0.0250 |
| internvl35_8b_hf | chart             | clean            |          0.7000 |             1.0000 |             0.0000 |
| internvl35_8b_hf | chart             | jpeg_compression |          0.0250 |             0.1000 |             0.9000 |
| internvl35_8b_hf | ocr_doc           | clean            |          0.9500 |             1.0000 |             0.0000 |
| internvl35_8b_hf | scientific_figure | clean            |          0.8750 |             1.0000 |             0.0000 |
| qwen3_vl_4b      | chart             | clean            |          0.9500 |             0.9750 |             0.0250 |
| qwen3_vl_4b      | chart             | jpeg_compression |          0.0000 |             0.0000 |             1.0000 |

Full real-VLM metrics are in `results/metrics.csv`; perturbation-level bootstrap confidence intervals are in `results/metric_cis_by_perturbation.csv`. The ablation table records 113 claim rows where the final answer is incorrect but the evidence claim is supported, showing why answer accuracy and evidence support are worth tracking separately.

Core plots:

- `results/figures/accuracy_heatmap.png`
- `results/figures/faithfulness_heatmap.png`
- `results/figures/accuracy_vs_faithfulness.png`
- `results/figures/perturbation_impact.png`

Failure examples are stored in `results/failure_examples.jsonl`. A 30-row audit worksheet in `docs/manual_audit.md` is prepared for post-run review; regenerate the summary only after the rows are manually checked.

## Open VLM Case Study On Kaggle

The real open-model evaluation is packaged in `kaggle/` so it can run on a GPU notebook without changing local validation artifacts. The checked-in full run uses `qwen3_vl_8b`, `internvl35_8b_hf`, and `qwen3_vl_4b`: two independent model families plus a Qwen 4B/8B scale-family comparison. Fresh reruns use a two-line `Answer:` and `Evidence:` protocol so answer accuracy and grounded evidence support can be inspected separately. This should be read as a controlled case study, not a leaderboard. `glm41v_9b_thinking` remains configured for optional appendix experiments only.

Create the local Kaggle dataset zip first, then upload it as a Kaggle dataset:

```bash
python scripts/create_kaggle_dataset_zip.py
```

Upload `kaggle/docvlm-fingerprint-kaggle-dataset.zip`, then import `kaggle/docvlm_fingerprint_open_vlm_case_study.ipynb` and run it with GPU + Internet enabled. The notebook exports `docvlm_fingerprint_open_vlm_results.zip` with regenerated JSONL, CSV, plots, and per-model outputs. The zip files are local runtime artifacts, ignored by git, and should not be committed.

## External Real-Data Sanity Checks

The main reported run uses the generated dataset only. Three separate real-data checks are included under `results/external/` and remain separate from `results/metrics.csv`. ChartQA uses a 30-example `HuggingFaceM4/ChartQA` slice with three model entries and 90 clean chart-QA calls: `0.6000` exact-match accuracy for `qwen3_vl_8b`, `0.2000` for `internvl35_8b_hf`, and `0.5000` for `qwen3_vl_4b`. DocumentVQA uses a 30-example `HuggingFaceM4/DocumentVQA` validation slice with the same three model entries and 90 clean OCR-document calls: `0.6667` accuracy for `qwen3_vl_8b`, `0.6333` for `internvl35_8b_hf`, and `0.6333` for `qwen3_vl_4b`.

CharXiv adds the most direct real-data perturbation check: `data/external/charxiv/` contains curated metadata for 20 real arXiv chart examples with the same four perturbations, and `results/external/charxiv/` contains Qwen-only outputs for `qwen3_vl_4b` and `qwen3_vl_8b`. The CharXiv image folders are local-only and ignored by git. Clean answer accuracy is `0.4000` for `qwen3_vl_4b` and `0.4500` for `qwen3_vl_8b`; JPEG compression drops both to `0.1000`, while blur/downscale reaches `0.1500` and `0.2500`. Treat ChartQA, DocumentVQA, and CharXiv as small external checks, not benchmark results.

Reproduce or rerun the clean external paths with the import/evaluation helpers in `src/import_chartqa_slice.py` and `src/evaluate_external_chartqa.py`. The CharXiv supplement includes source attribution fields in `data/external/charxiv/charxiv_selection.json`; regenerated CharXiv images should remain local unless redistribution permissions are reviewed.

## Public Project Page

A lightweight GitHub Pages version is available at `docs/index.html`. To publish it, open the repository settings on GitHub, enable Pages from the `main` branch and `/docs` folder, then use:

```text
https://samarthkukreja.github.io/DocVLM-Fingerprint/
```

The page uses only files under `docs/`, including copied result figures and first-party generated sample images. External dataset images are not included.

## Dashboard

Run from the repository root after generating outputs:

```bash
streamlit run dashboard/app.py
```

The dashboard reads existing JSONL/CSV/PNG outputs and shows:

- selected image
- domain and perturbation
- question and ground-truth answer
- model answer
- extracted claims
- support labels and scoring methods
- metrics table
- the three core plots

## Run Commands

From `docvlm-fingerprint/`:

```bash
python src/generate_charts.py
python src/generate_ocr_docs.py
python src/generate_scientific_figures.py
python src/dataset.py
python src/perturbations.py
python src/evaluate.py
python src/metrics.py
python src/manual_audit.py
python src/ablations.py
```

Useful checks:

```bash
python src/dataset.py
python src/evidence_scorer.py --self-test
python src/ablations.py
python -m unittest discover -s tests
python -c "from pathlib import Path; assert Path('results/metrics.csv').exists(); assert Path('results/metric_cis_by_perturbation.csv').exists(); assert Path('docs/manual_audit.md').exists()"
```

## Add Your Own Model

All model integrations use the shared `answer(image_path, question)` client interface. For most Hugging Face chat-template VLMs, start with the example config instead of editing source code:

```bash
copy configs\custom_model.example.yaml configs\custom_model.yaml
python src/evaluate.py --config configs/custom_model.yaml --models custom_mock --limit 2
python src/metrics.py
```

To evaluate your own Hugging Face model, edit `configs/custom_model.yaml`, set `name`, `provider`, `model_id`, and model-loading fields, then run:

```bash
python src/evaluate.py --config configs/custom_model.yaml --models my_hf_vlm --limit 5
python src/metrics.py
```

For a new provider API or model family:

1. Implement a subclass of `BaseVLMClient` in `src/vlm_clients.py`.
2. Put the API call or local inference code inside `_answer_uncached(image_path, question)`.
3. Register the provider in `CLIENT_REGISTRY`.
4. Add a model entry to `configs/models.yaml` or pass a separate config with `--config`.
5. Document required environment variables in `.env.example` without committing secrets.

External clean-data checks also accept the same model config path:

```bash
python src/evaluate_external_chartqa.py --config configs/custom_model.yaml --models my_hf_vlm --annotations data/external/chartqa/annotations.jsonl --output-dir results/external/chartqa/custom
```

## Related Work And Differentiation

Existing work already covers broad multimodal evaluation, hallucination benchmarks, robustness testing, document/chart/OCR tasks, and claim-level faithfulness. Relevant overlaps include VLMEvalKit, lmms-eval, HallusionBench, POPE, FaithScore, VLM-RobustBench, OCR-Robust, CharXiv, ChartQA, PlotQA, DocVQA, and ScienceQA-style datasets.

DocVLM-Fingerprint differs by integrating these ideas into a compact reusable workflow for document-centric stress testing: small generated data, standardized perturbations, plug-in model clients, answer accuracy, grounded faithfulness scoring, and readable failure examples.

Contribution summary: reproducible document-centric VLM stress-test workflow; deterministic perturbation suite with open-VLM case study; answer/claim faithfulness reporting with bootstrap CIs; audit-ready failure analysis; completed external ChartQA and DocumentVQA sanity checks; a Qwen-only CharXiv real-chart perturbation supplement; and descriptive ablation tables for answer/claim disagreement and scorer behavior.

See:

- `docs/project_page.md`
- `docs/related_work_and_differentiation.md`
- `docs/literature_review_matrix.md`

## Limitations

These limitations are intentional scope boundaries rather than hidden claims; they are part of the project framing for reviewers.

- The main dataset has 120 generated examples. It is useful for controlled, reproducible inspection, but it is not representative of all real documents.
- The checked-in full run includes three model entries, but only two independent model families because `qwen3_vl_4b` and `qwen3_vl_8b` are a scale-family comparison. The project should still be framed as a controlled workflow case study, not a final model ranking.
- Most questions are single-hop lookups. The two-line answer/evidence protocol makes claim support measurable, but longer rationale-style tasks would stress claim scoring more strongly.
- The binary scorer is transparent and testable, but simple. Nuanced paraphrases, visual entailment, and ambiguous evidence may require manual review or a stronger verifier. `unsupported` means not entailed by the annotated evidence string, so hallucination rate is an upper bound on fabrication under this scorer.
- The perturbation set is focused on four common visual stresses. Rotations, layout shifts, lighting changes, watermarks, and adversarial text remain future work.
- Bootstrap confidence intervals are included as descriptive uncertainty summaries, not population-level guarantees. The manual audit worksheet should be reviewed after each fresh full run; independent multi-annotator review is still needed before claiming independent human validation. The simulated JPEG perturbation can remove small text entirely, so near-zero scores should be read as a missing-information floor condition rather than calibrated compression robustness.
- No model training or fine-tuning is performed; the contribution is evaluation infrastructure and analysis.

Recommended next improvements: polish the final PDF/project page, keep external-result claims clearly separated from the main generated run, and consider a larger independently reviewed audit only if time remains.
