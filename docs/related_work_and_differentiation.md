# Related Work And Differentiation: DocVLM-Fingerprint

## Bottom Line

Decision: **Refine and proceed**.

DocVLM-Fingerprint should **not** be presented as a completely new research area or as the first hallucination benchmark for vision-language models. Many pieces already exist in the research community.

The honest contribution is:

> DocVLM-Fingerprint is a compact, reusable evaluation framework that combines controlled visual perturbations, document-centric VLM tasks, claim-level faithfulness scoring, and model-specific failure fingerprints in one reproducible MS-level artifact.

This should be framed as a **reproducible systems/evaluation workflow**, not as a new faithfulness metric. FaithScore and related work already cover atomic fact / claim-level faithfulness conceptually; DocVLM-Fingerprint uses that idea as one module inside a broader document-centric perturbation and failure-analysis workflow.

The model comparisons are demonstration case studies. The main artifact is the reusable evaluation framework.

This is not the first VLM hallucination benchmark, not a comprehensive benchmark, and not a leaderboard.

## What Others Are Doing

### 1. General VLM Evaluation Frameworks

Existing projects already provide broad model evaluation infrastructure.

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) is an open-source toolkit for evaluating large vision-language models across many benchmarks and models. Its stated goal includes making it easy for VLM developers to evaluate their own models by implementing a model interface while the framework handles other evaluation workload.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is a one-for-all multimodal evaluation toolkit across text, image, video, and audio tasks.

What they do well:

- Broad benchmark coverage.
- Many supported models.
- Standardized evaluation runners.
- Leaderboard-style model comparison.

What DocVLM-Fingerprint does differently:

- Narrower and more interpretable.
- Focuses specifically on hallucination robustness under visual perturbations.
- Uses claim-level faithfulness rather than only answer-level correctness.
- Designed as a small, readable, reproducible framework for an admissions/research artifact.

## 2. Hallucination Benchmarks

Several benchmarks study hallucination in large vision-language models.

- [POPE](https://github.com/RUCAIBox/POPE) studies object hallucination in LVLMs using polling-based object queries.
- [HallusionBench](https://github.com/tianyi-lab/HallusionBench) evaluates entangled language hallucination and visual illusion through image-context reasoning questions.
- [HallusionBench paper](https://arxiv.org/abs/2310.14566) emphasizes visual illusion, language hallucination, control groups, response tendencies, and failure modes.

What they do well:

- Strong hallucination-focused benchmark design.
- Human-crafted examples.
- Diagnostic evaluation of model failure modes.

What DocVLM-Fingerprint does differently:

- Focuses on document-centric domains: charts/tables, OCR-heavy documents, and scientific figures/pages.
- Adds controlled visual perturbations to test robustness shifts.
- Scores free-form answers at the claim level.
- Produces a failure fingerprint across perturbation types.

## 3. Claim-Level Faithfulness And Atomic Fact Scoring

Claim-level evaluation already exists conceptually.

- [FaithScore](https://arxiv.org/abs/2311.01477) evaluates hallucinations in LVLM outputs by identifying descriptive statements, extracting atomic facts, and verifying consistency between those facts and the input image.

What it does well:

- Fine-grained faithfulness evaluation.
- Atomic fact decomposition.
- Verification against the image.

What DocVLM-Fingerprint does differently:

- Uses claim-level faithfulness as one module inside a full stress-testing framework.
- Combines claim-level scoring with controlled perturbations and document-centric tasks.
- Prioritizes simple reproducible outputs: JSONL records, metrics CSV, plots, and dashboard.

## 4. Robustness Under Visual Perturbations

VLM robustness under corruptions is already an active area.

- [VLM-RobustBench](https://arxiv.org/abs/2603.06148) studies robustness across many augmentation types and corrupted settings.
- Work on VLMs under corruptions studies how image degradations affect multimodal model performance.

What they do well:

- Broad perturbation coverage.
- Systematic corruption severity analysis.
- Robustness metrics across multiple model families.

What DocVLM-Fingerprint does differently:

- Keeps perturbations intentionally small and interpretable.
- Uses only four MVP perturbations: blur/downscale, crop/removal, JPEG compression, and distractor text.
- Looks beyond accuracy drop by also measuring claim-level faithfulness and unsupported-claim rate.

## 5. OCR, Chart, And Scientific-Figure Understanding

There is also direct overlap with OCR and chart robustness.

- [OCR-Robust](https://arxiv.org/abs/2606.26041) evaluates OCR reasoning robustness under visual perturbations across documents, receipts, handwriting, math, charts, geometry diagrams, and tables.
- [CharXiv](https://arxiv.org/abs/2406.18521) evaluates realistic chart understanding in multimodal LLMs using charts from arXiv papers.

What they do well:

- Strong domain-specific evaluation.
- Realistic chart/OCR tasks.
- Careful robustness or reasoning analysis.

What DocVLM-Fingerprint does differently:

- Combines charts, OCR-heavy documents, and scientific figures in one compact framework.
- Adds claim-level faithfulness scoring to detect unsupported reasoning.
- Emphasizes reusable framework design rather than only a benchmark dataset.

## 6. Additional Task Benchmarks

ChartQA, PlotQA, DocVQA, and ScienceQA/AI2D-style science-diagram benchmarks cover important parts of the task space that DocVLM-Fingerprint uses. They are valuable precedents for chart, document, and scientific visual reasoning, but they are not presented here as direct competitors to a compact hallucination robustness workflow.

DocVLM-Fingerprint should cite this family of work as task/domain inspiration. The difference is not that these tasks are new; the difference is the integrated workflow that pairs document-centric examples with standardized perturbations, plug-in VLM evaluation, answer accuracy, grounded faithfulness scoring, and reproducible failure analysis.

## 7. Name Collision

There is a possible naming collision around generic "VLM fingerprinting" terminology in recent literature. Some work uses fingerprinting for model ownership verification or model identity, not hallucination robustness; the `DocVLM-` prefix reduces that ambiguity.

To avoid confusion, possible alternative names include:

- `FaithPrint-VLM`
- `VLM-FaithPrint`
- `DocVLM-FaithEval`
- `VLM-GroundingLab`

If keeping the current name, the README and paper should make the scope explicit:

> DocVLM-Fingerprint refers to failure-mode and faithfulness fingerprints for hallucination robustness, not model ownership fingerprinting.

## What We Are Doing Differently

DocVLM-Fingerprint is best positioned as an **integration and reproducibility contribution**, not a claim of total novelty.

Our differentiators:

1. **Reusable framework, not just a leaderboard**
   - Other users can plug in their own VLM through the model-client interface.

2. **Document-centric evaluation**
   - Charts/tables, OCR-heavy documents, and scientific figures/pages are the core domains.

3. **Controlled perturbation stress tests**
   - Four interpretable visual shifts test how grounding degrades.

4. **Claim-level faithfulness**
   - Answers are decomposed into claims and scored as supported or unsupported.

5. **Failure fingerprints**
   - Each model receives a weakness profile across perturbation types.

6. **Admissions-friendly reproducibility**
   - Small dataset, JSONL artifacts, metrics CSV, plots, dashboard, and LaTeX report.

## What We Should Claim

Use language like:
Primary paper framing:

> We introduce a lightweight, reproducible workflow for evaluating document-centric VLM reliability: curate tasks, apply standardized visual perturbations, evaluate plug-in VLMs, score both answer accuracy and grounded faithfulness, and produce failure analyses.


> We build a compact, reproducible evaluation framework that integrates controlled visual perturbations with claim-level faithfulness scoring for document-centric VLM tasks.

> We demonstrate the framework on representative models and show how answer accuracy can mask unsupported visual reasoning.

> DocVLM-Fingerprint is intended as a reusable diagnostic tool, not as a comprehensive leaderboard.

## What We Should Not Claim

Avoid:

- "First VLM hallucination benchmark."
- "New faithfulness metric" as the main contribution.
- "Completely novel hallucination evaluation paradigm."
- "Solves hallucination."
- "Causal proof of hallucination mechanisms."
- "Comprehensive benchmark for all VLM robustness."
- "GPT is better than Gemini" as the main contribution.

## Implementation Implications

The repo should prioritize:

- Clean model-client interface.
- Clear `Add Your Own Model` documentation.
- Config-driven evaluation.
- Dataset and perturbation modules independent from model code.
- Scoring and metrics independent from specific model providers.
- Reproducible outputs.

The paper and README should include a short related-work/differentiation section so reviewers see that the project is literature-aware and honestly scoped.

## Suggested Final Positioning

> Existing VLM evaluation frameworks provide broad benchmark coverage, hallucination benchmarks study specific failure modes, robustness suites test image corruptions, and faithfulness metrics evaluate atomic claims. DocVLM-Fingerprint combines these ideas into a compact, reusable framework for document-centric VLM hallucination robustness, with controlled perturbations, claim-level faithfulness scoring, and model-specific failure fingerprints.


