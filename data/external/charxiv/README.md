# CharXiv External Perturbation Supplement

This folder contains a small real-chart supplement for DocVLM-Fingerprint. It is separate from the main generated-dataset run.

## Contents

- `annotations.jsonl`: 20 curated CharXiv chart examples.
- `perturbation_metadata.jsonl`: four perturbations per example, 80 variants total.
- `samples/`: normalized real chart images.
- `perturbed/`: deterministic perturbations for the normalized images.
- `charxiv_selection.json`: exact figure IDs, questions, answers, evidence strings, source, and license fields.
- `verification_sheet.md`: worksheet used to inspect the curated evidence strings.
- `ingest_charxiv.py`: optional regeneration helper for a local CharXiv image zip.

## Reporting

The corresponding result artifacts live under `results/external/charxiv/`. They cover `qwen3_vl_4b` and `qwen3_vl_8b` only, so they should be reported as a Qwen-family external supplement, not as part of the main three-entry generated-dataset run.

CharXiv is useful here because it applies the same four perturbations to real arXiv chart images. ChartQA and DocumentVQA remain clean-only sanity checks.

## Attribution

The selected figures come from the CharXiv validation split and are recorded with per-example `source` and `license` fields. The selection file records `CC BY-SA 4.0` for the images. Keep this attribution with any redistributed copy of the supplement.
