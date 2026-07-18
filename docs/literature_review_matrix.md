# Literature Review Matrix

Use this document before implementation to validate whether DocVLM-Fingerprint is still worth building.

Core question:

> Is DocVLM-Fingerprint the highest-impact project to build for a top MS AI/ML application, given existing work?

## Matrix

This matrix should answer two questions:

1. Does prior work already cover the idea?
2. Can this project still become a meaningful, feasible, admissions-strengthening artifact in two months?

| Paper / Repo | Link | Problem | Dataset / Domain | Method / Framework | GitHub? | Limitation | Can I realistically improve it in 2 months? | Engineering difficulty (1-5) | Novelty overlap | Admissions value (1-5) | How DocVLM-Fingerprint Differs |
|---|---|---|---|---|---|---|---|---:|---|---:|---|
| VLMEvalKit | https://github.com/open-compass/VLMEvalKit | Broad VLM evaluation toolkit | Many VLM benchmarks | Plug-in evaluation framework | Yes | Broad framework, not focused on claim-level hallucination robustness under document perturbations | Yes: build a smaller specialized workflow instead of competing with broad infrastructure | 3 | Medium | 4 | Narrower framework: document-centric perturbation stress tests plus claim-level faithfulness and failure fingerprints |
| lmms-eval | https://github.com/EvolvingLMMs-Lab/lmms-eval | Multimodal model evaluation | Text, image, video, audio benchmarks | Unified evaluation harness | Yes | Broad benchmark runner, not focused on document-centric faithfulness stress testing | Yes: create a lightweight reproducible workflow with clearer failure analysis | 3 | Medium | 4 | Emphasizes small reproducible hallucination robustness analysis over broad benchmark coverage |
| HallusionBench | https://github.com/tianyi-lab/HallusionBench | VLM hallucination and visual illusion | Image-question pairs with control groups | Hallucination benchmark | Yes | Focuses on benchmark design, not reusable perturbation + document-centric claim scoring workflow | Yes: focus on visual perturbations and document-centric domains | 3 | Medium | 4 | Combines perturbation sensitivity with claim-level support scoring in charts/OCR/scientific figures |
| POPE | https://github.com/RUCAIBox/POPE | Object hallucination in LVLMs | Object-presence polling questions | Polling-based hallucination evaluation | Yes | Primarily object hallucination, less focused on documents/charts/OCR and free-form claims | Yes: target different domains and free-form answer grounding | 2 | Low-Medium | 3 | Targets document-centric visual evidence and free-form answer claims |
| FaithScore | https://arxiv.org/abs/2311.01477 | Fine-grained LVLM hallucination evaluation | Image descriptions / LVLM outputs | Atomic fact extraction and verification | Paper | Strong overlap on claim/fact-level faithfulness | Partly: do not claim a new metric; use claim scoring inside a broader workflow | 4 | High | 4 | Integrates faithfulness-style scoring with document-centric perturbation stress testing and reproducible failure analysis |
| VLM-RobustBench | https://arxiv.org/abs/2603.06148 | VLM robustness under corruptions | Robustness benchmark | Multiple augmentations/corruptions | Paper | Strong overlap on robustness, less focused on claim-level faithfulness and document evidence | Yes: narrow to interpretable perturbations and faithfulness analysis | 4 | Medium-High | 4 | Reports faithfulness and unsupported-claim rate, not only task performance under corruptions |
| OCR-Robust | https://arxiv.org/abs/2606.26041 | OCR reasoning robustness of LMMs | Documents, receipts, handwriting, math, charts, tables | Visual perturbation benchmark | Paper | Strong overlap with OCR/chart perturbations | Partly: must avoid being only OCR robustness; add framework usability and claim-level failure analysis | 4 | High | 4 | Differentiates through reusable workflow, claim-level faithfulness, scientific figures, and failure fingerprints |
| CharXiv | https://arxiv.org/abs/2406.18521 | Realistic chart understanding for multimodal LLMs | Charts from arXiv papers | Chart understanding benchmark | Paper | Chart-focused; not a general hallucination faithfulness workflow | Yes: include charts as one domain rather than the whole project | 3 | Medium | 4 | Includes charts as one domain and evaluates claim support under perturbations |
| ChartQA | https://arxiv.org/abs/2203.10244 | Question answering over charts | Human-written and machine-generated chart QA | Chart VQA benchmark | Dataset | Chart-focused; does not address perturbation-driven hallucination or claim support | Yes: reuse chart-style tasks in a broader workflow | 2 | Medium | 4 | Treats charts as one document-centric domain and adds perturbation + faithfulness analysis |
| PlotQA | https://arxiv.org/abs/1909.00997 | Reasoning over plots | Synthetic plots with QA pairs | Chart/plot QA benchmark | Dataset | Strong chart QA coverage but not a hallucination robustness workflow | Yes: small synthetic chart subset is feasible | 2 | Medium | 3 | Uses chart QA as a controlled component inside a larger VLM reliability pipeline |
| DocVQA | https://www.docvqa.org/ | Visual question answering on documents | Document images, forms, pages | Document VQA benchmark | Dataset | Document QA focus without the full perturbation + claim-level failure-fingerprint workflow | Yes: emulate document-style examples locally without external private data | 3 | Medium | 4 | Adds deterministic local OCR-style documents and faithfulness-oriented stress testing |
| ScienceQA / AI2D-style science diagrams | https://scienceqa.github.io/ | Multimodal science reasoning | Science diagrams and educational figures | Multimodal QA benchmark | Dataset / paper | Broad science reasoning, not focused on perturbation robustness and unsupported visual claims | Partly: include a small scientific-figure subset later | 3 | Medium | 4 | Uses scientific figures as the third compact domain with standardized perturbation analysis |

## Alternative Project Comparison

Use this table to avoid merely justifying DocVLM-Fingerprint. Compare it against plausible alternatives.

| Project | Novelty | Feasible in 2 months | Paper potential | CV / AI alignment | Admissions value | Verdict |
|---|---|---|---|---|---|---|
| DocVLM-Fingerprint / FaithPrint-style framework | Medium | High | High if experiments are careful | Excellent | 4/5 | Refine and proceed |
| DocRAG evaluation | Medium | High | Medium | Moderate | 3/5 | Good fallback, weaker CV/multimodal signal |
| Video grounding benchmark | Medium-High | Low-Medium | High | Excellent | 3/5 | Risky timeline without strong data |
| Vision-language robustness toolkit | Medium | High | High | Excellent | 4/5 | Similar direction; keep as framework framing |
| Deepfake research extension | Medium-High | Medium | High | Excellent | 4-5/5 | Potentially stronger, but currently out of scope by preference |

## Decision Gate

Decision:

- [ ] Proceed
- [x] Refine and proceed
- [ ] Pivot

Strongest overlaps:

1. FaithScore: claim/fact-level faithfulness overlap.
2. OCR-Robust: OCR/chart perturbation robustness overlap.
3. VLMEvalKit/lmms-eval: broad reusable evaluation framework overlap.

Remaining gap:

The gap is not a brand-new metric. The gap is a lightweight, reproducible evaluation workflow that combines document-centric tasks, standardized perturbations, answer accuracy, claim-grounded faithfulness, and failure analysis in a form another engineer can rerun or extend.

Why this is still worth building:

It remains a strong Master's application project because it demonstrates evaluation design, reproducible ML systems engineering, multimodal reliability analysis, and honest literature-aware positioning. The paper should be framed as a reproducible systems/evaluation report, not a novel benchmark paper.

Required changes before Day 0:

- Do not claim a new faithfulness metric.
- Frame claim-level scoring as an evaluation module inspired by existing faithfulness work.
- Make the paper about the workflow: task curation, perturbations, VLM evaluation, faithfulness scoring, and failure analysis.
- Keep model comparisons as demonstration case studies.
- Keep the reusable plug-in model interface as a central deliverable.

Final go / no-go:
Go, with refined framework-first positioning.
