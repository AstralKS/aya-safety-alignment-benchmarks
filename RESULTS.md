# Aya factuality baselines

> **Extra-only old evaluator, retired at the user's request on2026-11-11 at09:43 UTC.** The scores below and `metrics.json` are historical outputs. They are **not** the new primary evaluation and must not be reported as it.
>
> **New primary results: none yet.** The user approved all six corrected variants and full3,369 SelfAware. Inputs are assembled at9,530/model (7,552 new,1,978 reused); integrity review and FActScore evidence freeze are in progress. One GLM factual judge; official phrase-first SimCSE for SelfAware. Corrected GLM-only planning cost$9.01/$13.52 with50% retries, before unpriced hosted embeddings; full budget fit is unproved. See `scoring/EVALUATION-DESIGN-20261111.md`. Old answers/grades are preserved; no new generation or scoring has run.

**Historical status: incomplete at retirement; counts below are extra-only.**

Updated: 2026-11-11T09:30:23.576270+00:00. Each model has 1,000 SimpleQA Verified, 978 AbstentionBench subset, and 100 LongFact++ held-out subset questions.

## SimpleQA Verified — full 1,000-question evaluation

| Model | Scored | Correct | Incorrect | Not attempted | Correct yield | Coverage | Accuracy when attempted |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny-baseline | 1000/1000 | 31 | 918 | 51 | 3.1% | 94.9% | 3.3% |
| 8b-baseline | 1000/1000 | 57 | 880 | 63 | 5.7% | 93.7% | 6.1% |

SimpleQA grades the requested reference target. It does not certify the factual accuracy of every extra sentence in an answer. Targeted judge reviews and potential grading disagreements are recorded in `AUDIT.md`.

## AbstentionBench — balanced 978-question subset

| Model | Scored | Appropriate abstentions / 489 | Over-refusals / 489 | Answered should-abstain / 489 | Behavior accuracy | Task success |
|---|---:|---:|---:|---:|---:|---:|
| tiny-baseline | 978/978 | 214 | 45 | 275 | 67.3% | 46.8% |
| 8b-baseline | 978/978 | 249 | 43 | 240 | 71.1% | 54.8% |

Behavior accuracy checks whether the model abstains when it should, regardless of whether attempted answers are correct. Task success additionally requires non-abstaining answers on answerable questions to match references. Source-defined should-abstain labels include ambiguity and false premises; answering these questions is a desired-behavior error, not automatically a verified factual hallucination. The official broad abstention rubric counts uncertainty and premise corrections; it differs from SimpleQA’s not-attempted label. Component details, precision/recall/F1, selective risk, and reference correctness are in `metrics.json`.

**WorldSense sensitivity (primary grades unchanged):** interpreting only bare option-3 answers as source-native abstentions changes behavior accuracy from 67.28% to 67.79% for Tiny and 71.06% to 71.88% for 8B, and task success from 46.83% to 48.88% / 54.81% to 56.44%. This applies on both desired-label strata and excludes longer explanations. Exact affected IDs and denominators: `scoring/worldsense-option3-sensitivity.json`; diagnostic controls and other semantic concerns: `AUDIT.md`.

## LongFact++ — alternative evidence-backed claim evaluation

| Model | Responses scored | Claims | Supported | Contradicted | Unresolved | Evidence resolution |
|---|---:|---:|---:|---:|---:|---:|
| tiny-baseline | 42/100 | 2112 | 163 | 76 | 1873 | 11.3% |
| 8b-baseline | 40/100 | 1798 | 194 | 31 | 1573 | 12.5% |

This is **not the official LongFact++ entity-level score**. Fresh claims are extracted from Aya responses and checked against cached Wikipedia passages and Crossref publication metadata. “Contradicted” requires positive contrary evidence; missing evidence is unresolved. Exact evidence quotes are checked mechanically, then a separate short-context check must confirm that those quotes entail the assigned label. Disagreements become unresolved. Claim granularity is model-extracted and may include compound assertions. Retrieval and judge errors are recorded. Resolved-only precision is not factual accuracy over all claims. The original annotations for other models are never used as Aya labels.

## Execution and scoring record

- Official Aya weights and chat templates; FP16, no quantization, greedy decoding. Inference runs on Kaggle T4 GPUs.
- Generation ceilings: 256 / 512 / 1,536 tokens for SimpleQA / AbstentionBench / LongFact++. Truncated responses remain in evaluation.
- Short-answer rubrics are copied from official releases; the substitute judge is `Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra`. These are local benchmark evaluations, not official leaderboard submissions.
- LongFact scorer `longfact-evidence-claims-v4-reviewed` uses Qwen for extraction/primary verification and `openai/gpt-oss-120b:deepinfra` for single-claim sufficiency/incompatibility checks. Every remaining resolved claim also receives a direct quote-based assistant review, which may confirm or downgrade to unresolved. The legacy receipt identity is GPT-6 Astra; current reviews additionally disclose GPT-6.1 Sol as the actual reviewing model. This is not human annotation or a blinded independent study. Failed retrieval or pending evidence review prevents acceptance as scored.
- Raw generations and runtime manifests: `runs/`. Per-item grades: `scoring/<run>/`. Full judge requests/responses: `scoring/api_cache/`. Retrieved text/URLs: `scoring/evidence_cache/` and per-item `longfact_details/`.
- HF provider-reported judge cost: $2.2237; conservative reserved budget: $5.7046, ceiling $18.00.

| Model | Generated | Graded | Truncated by dataset (SimpleQA / Abstention / LongFact) |
|---|---:|---:|---|
| tiny-baseline | 2078/2078 | 2020/2078 | 25 / 44 / 4 |
| 8b-baseline | 2078/2078 | 2018/2078 | 35 / 23 / 7 |

### Remote compute

- tiny-baseline: model loading 43.0s; generation 143.7 minutes; 2 allocated T4 GPUs; load-plus-generation device-hour estimate 4.813; peak allocated GPU memory [7.921979904174805, 0.0] GiB.
- 8b-baseline: model loading 104.6s; generation 315.2 minutes; 2 allocated T4 GPUs; load-plus-generation device-hour estimate 10.563; peak allocated GPU memory [7.871841907501221, 8.388935089111328] GiB.
Device-hour estimates exclude setup, queueing and earlier attempts, and are not Kaggle quota measurements. Earlier attempt logs are retained separately.

## Experimental interpretation

Wait for complete matched outputs and grading before selecting the next experiment.

See `DATASET_INSPECTION.md` for exact selection, source versions, prompt deviations, and limitations. Bootstrap/claim independence assumptions are not made; Wilson intervals in JSON apply only to prompt-level binomial summaries.
