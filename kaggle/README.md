# Kaggle Artifacts for DocVLM-Fingerprint

Create the dataset zip from the repository root with `python scripts/create_kaggle_dataset_zip.py`, upload `docvlm-fingerprint-kaggle-dataset.zip` as a Kaggle dataset, then import or upload `docvlm_fingerprint_open_vlm_case_study.ipynb` as a Kaggle Notebook. The generated zip files are local artifacts and are ignored by git.

Recommended Kaggle settings:

- Accelerator: GPU
- Internet: On

The checked-in full paper run uses three model entries: two independent model families plus a Qwen 4B/8B scale-family comparison:

1. `qwen3_vl_8b`: Qwen3-VL-8B-Instruct, a recent 8B-class open VLM with strong document/OCR/chart capability.
2. `internvl35_8b_hf`: InternVL3.5-8B-HF, a separate 8B-class open VLM family for comparison through the same pipeline.
3. `qwen3_vl_4b`: Qwen3-VL-4B-Instruct, used as a scale-family comparison against Qwen3-VL-8B.

`glm41v_9b_thinking` remains configured for optional appendix experiments, but it is not part of the checked-in full-run metrics. The project should be described as a controlled open-VLM workflow case study, not as a ranking exercise.

Fallback models remain configured if Kaggle memory or runtime becomes tight:

- `qwen25_vl_3b`
- `internvl3_2b`
- `smolvlm2_2b`
- `llava15_7b` for optional legacy appendix comparison

The notebook will:

1. Load the project into `/kaggle/working/docvlm-fingerprint` from either the uploaded zip or Kaggle's auto-expanded dataset folder.
2. Install optional Hugging Face VLM dependencies while keeping Kaggle's managed Torch/protobuf/Pillow stack intact.
3. Run dataset and scorer checks.
4. Smoke-test the configured open VLMs.
5. Run the full evaluation for the selected paper models one at a time.
6. Merge outputs and regenerate metrics/plots.
7. Export `/kaggle/working/docvlm_fingerprint_open_vlm_results.zip`.

If Kaggle runs out of GPU memory, use the fallback `MODEL_NAMES` line in the first notebook cell. If it is only slow, set `FULL_EVAL_LIMIT = 20` for a smoke run, then restore `None` for a complete run.


Troubleshooting:

- If `/kaggle/input/docvlm-fingerprint-kaggle-dataset/docvlm-fingerprint` is visible, the notebook can use it directly; no `.zip` file needs to appear in the file browser.
- If you previously ran an install cell that upgraded `protobuf`, `pillow`, `torch`, or CUDA packages, restart the Kaggle session before rerunning the updated notebook from the top.

- If `transformers` fails while importing `torchaudio` with a Torch/TorchAudio CUDA mismatch, use the updated notebook install cell. The project does not use audio, so it uninstalls `torchaudio` before running VLM inference.

- If `bitsandbytes` reports `libnvJitLink.so.13` missing, use the updated notebook install cell. It installs `nvidia-nvjitlink-cu13` and exports the NVIDIA package library paths before subprocess model evaluation.

- The default notebook avoids `bitsandbytes` because Kaggle CUDA images can fail on native quantization libraries such as `libnvJitLink.so.13`. The main 8B-9B configs use BF16/auto-device loading instead. Use a high-memory GPU session for the 8B-class paper models.

- If current source `transformers` reports `safetensors>=0.8.0` is required, use the updated install cell. It upgrades only `safetensors`, not Kaggle's full Torch/CUDA stack.

