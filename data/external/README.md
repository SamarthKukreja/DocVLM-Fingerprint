# Optional External Dataset Slices

This directory is reserved for optional real-data sanity checks. The main checked-in DocVLM-Fingerprint results use the generated dataset in `data/annotations.jsonl`; external slices are not part of the reported generated-dataset run unless explicitly documented.

## ChartQA

Use the importer with a local ChartQA-style dataset copy:

```bash
python src/import_chartqa_slice.py --chartqa-root <path> --limit 30 --output-dir data/external/chartqa
python src/evaluate_external_chartqa.py --annotations data/external/chartqa/annotations.jsonl --models mock --output-dir results/external/chartqa
```

The importer copies a small local slice into `data/external/chartqa/` and writes `annotations.jsonl` in the same annotation shape used by the project. It does not download ChartQA and does not make license or redistribution assumptions. Do not commit imported third-party images unless their license has been checked separately.

Treat ChartQA as a clean-only external sanity check. Report its outputs separately from the generated-dataset metrics, and treat answer accuracy as the primary external metric because imported ChartQA records do not provide full visual evidence rationales for strong claim-faithfulness claims.

## CharXiv

`data/external/charxiv/` contains a curated 20-example real-chart supplement from CharXiv, with four deterministic perturbations per example. Its outputs live under `results/external/charxiv/` and cover the Qwen 4B/8B scale-family comparison only.

Report CharXiv separately from the main generated-dataset run. It is useful because it applies the same perturbation families to real arXiv chart images, while ChartQA and DocumentVQA are clean-only sanity checks. Keep the included source and `CC BY-SA 4.0` license fields with any redistributed copy of the supplement.
