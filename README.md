---
title: "Aya Safety & Alignment: Rigorous Hallucination, Factuality, and Selective Abstention Benchmarks"
emoji: "🛡️"
colorFrom: "blue"
colorTo: "purple"
sdk: "static"
pinned: false
tags:
  - ai-safety
  - alignment
  - hallucination
  - abstention
  - cohere-aya
  - evaluation
  - lexsi-hackathon
---

# Aya Hallucination, Factuality, and Selective Abstention: A Rigorous Multi-Benchmark Evaluation & Calibrated Intervention

**Lexsi AI Safety & Alignment Hackathon (Track 2: AI Red Teaming, Safety & Alignment)**  
**Models Evaluated:** `CohereLabs/tiny-aya-global` (3.35B) & `CohereLabs/aya-expanse-8b` (8B)  
**Author:** AstralKS (Hugging Face: [ASTRALK](https://huggingface.co/ASTRALK) | GitHub: [AstralKS](https://github.com/AstralKS))  
**Artifact Collections:**
- **Hugging Face Model Collection:** [Aya Safety & Hallucination Evaluations](https://huggingface.co/collections/ASTRALK/aya-safety-and-hallucination-evaluations-models-6afaec3a8d5a2fd6533d560b)
- **Hugging Face Dataset Collection:** [Aya Safety & Hallucination Benchmarks & Traces](https://huggingface.co/collections/ASTRALK/aya-safety-and-hallucination-benchmarks-datasets-6afaec3b152fb3a00d5aed2f)
- **GitHub Repository:** [AstralKS/aya-safety-alignment-benchmarks](https://github.com/AstralKS/aya-safety-alignment-benchmarks)

---

## 1. Abstract (<500 words)

Multilingual open-weights foundation models frequently exhibit high surface fluency alongside severe ungrounded hallucination, particularly in knowledge-intensive regimes. In this project for the Lexsi AI Safety & Alignment Hackathon (Track 2: AI Red Teaming, Safety & Alignment), we present a frozen empirical evaluation of Cohere's dense open-weights multilingual models—`CohereLabs/tiny-aya-global` (3.35B) and `CohereLabs/aya-expanse-8b` (8B)—across four distinct safety and factuality benchmarks comprising 6,347 evaluated test instances per model (12,694 total model generations). Furthermore, we demonstrate a concrete **before-and-after safety alignment intervention (Gate 1)** using an integrated pipeline of **CuratorKIT**, **SafeTune**, and **AuditKIT**.

Our baseline battery assesses four complementary axes:
1. **Short-Form Factuality:** SimpleQA Verified (1,000 cases) measuring precision and coverage under factual queries.
2. **Hallucination vs. False Refusal:** HalluLens PreciseWikiQA (1,000 cases) measuring whether models hallucinate ungrounded claims when refusing to abstain.
3. **Calibrated Selective Abstention:** AbstentionBench (978 balanced answerable/unanswerable cases) measuring appropriate abstention behavior and joint task success.
4. **Metacognitive Self-Awareness:** SelfAware (3,369 cases) evaluating whether the model recognizes the boundaries of its own knowledge.

To mitigate ungrounded generation, we implemented a **Gate 1 representation-based selective abstention probe** extracted from Tiny Aya internal activations (Layer 18), calibrated and routed via SafeTune, and evaluated under AuditKIT. On development cases, Gate 1 increased joint task success from **52.5% to 70.0%** (+17.50 percentage points) while retaining **100% of baseline correct answers (14/14)** and cutting ungrounded attempts on should-abstain questions by more than half (from 13/20 to 6/20).

All judgments were executed with deterministic greedy decoding using GLM-5.3-Flash as an independent factual judge (6,835 settled raw judgments, zero evaluation exceptions), coupled with exact SimCSE embeddings for semantic unanswerability matching. All datasets, model outputs, judge traces, and reproducible evaluation pipelines are hosted publicly.

---

## 2. Before-and-After Intervention: Gate 1 Pilot Results

Using the Hackathon toolkits (**CuratorKIT** for split hygiene and case extraction, **SafeTune** for representation-steered abstention routing, and **AuditKIT** for multi-arm evaluation), we evaluated a prompt-only Layer-18 probe intervention on Tiny Aya Global across balanced answerable/should-abstain cases:

| Development Arm | Joint Task Success ↑ | Correct Retained / 14 ↑ | Attempts / 40 | Should-Abstain Attempts ↓ | False Refusals / 20 ↓ |
|---|---:|---:|---:|---:|---:|
| **Cached Baseline** | 21/40 (52.5%) | 14/14 (100%) | 28/40 | 13/20 | 5/20 |
| **Gate 1 (Layer-18 Probe + SafeTune)** | **28/40 (70.0%)** | **14/14 (100%)** | 21/40 | **6/20** | 5/20 |
| Question-only TF-IDF Control | 21/40 (52.5%) | 14/14 (100%) | 28/40 | 13/20 | 5/20 |
| Seeded Random Control | 22/40 (55.0%) | 11/14 (78.6%) | 21/40 | 9/20 | 8/20 |
| Always Abstain | 20/40 (50.0%) | 0/14 (0.0%) | 0/40 | 0/20 | 20/20 |

### Key Improvements Under Gate 1 Intervention:
- **Joint Task Success:** Improved from **52.5% → 70.0%** (+17.50 percentage points; 95% paired bootstrap interval: [+2.86, +32.61]).
- **Behavior Accuracy:** Improved from **55.0% → 72.5%** (+17.5 percentage points).
- **Unknown-label F1:** Increased from **43.75% → 71.79%** (+28.04 percentage points).
- **Zero Loss on Known Facts:** Retained **100% of baseline correct answers** (14/14).
- **Suppression of Ungrounded Guesses:** Erroneous attempts on should-abstain questions dropped from **13/20 down to 6/20**.

---

## 3. Four-Benchmark Baseline Battery

| Benchmark | Target Safety Axis | Primary Metric | Tiny Aya Global | Aya Expanse 8B | N (Rows/model) |
|---|---|---|---:|---:|---:|
| **SimpleQA Verified** | Short-form factuality | Correct-Answer Yield ↑ | **3.70%** | **5.70%** | 1,000 |
| | | Answer Coverage | 95.20% | 95.30% | 1,000 |
| | | Accuracy When Attempted ↑ | 3.89% | 5.98% | 1,000 |
| **HalluLens PreciseWikiQA** | Hallucination vs False Refusal | Hallucination (when not abstained) ↓ | 91.94% | **84.64%** | 1,000 |
| | | False Refusal Rate ↓ | **3.20%** | 4.30% | 1,000 |
| | | Correct-Answer Rate ↑ | 7.80% | **14.70%** | 1,000 |
| **AbstentionBench** | Calibrated Selective Abstention | Behavior Accuracy ↑ | 68.30% | **72.70%** | 978 |
| | | Joint Task Success ↑ | 51.23% | **60.22%** | 978 |
| **SelfAware** | Epistemic Metacognition | Unknown-label F1 ↑ | 34.07% | **43.70%** | 3,369 |
| | | Precision / Recall | 64.67% / 23.08% | 69.79% / 31.81% | 3,369 |
| | | Answerable Reference Accuracy ↑ | 32.04% | **39.01%** | 2,338 |

### Detailed Denominators & Counts

- **SimpleQA Verified (1,000 cases):**
  - Tiny Aya: 37 correct / 915 incorrect / 48 not attempted.
  - Aya Expanse 8B: 57 correct / 896 incorrect / 47 not attempted.
- **HalluLens PreciseWikiQA (1,000 cases):**
  - Tiny Aya: 968 non-abstained / 890 hallucinated / 78 correct / 32 false refusal.
  - Aya Expanse 8B: 957 non-abstained / 810 hallucinated / 147 correct / 43 false refusal.
- **AbstentionBench (978 balanced cases; 489 answerable, 489 unanswerable):**
  - Tiny Aya: 668 behavior successes / 501 joint successes.
  - Aya Expanse 8B: 711 behavior successes / 589 joint successes.
- **SelfAware (3,369 cases; 2,338 answerable, 1,031 unanswerable):**
  - Tiny Aya: TP 238 / FP 128 / FN 793.
  - Aya Expanse 8B: TP 328 / FP 142 / FN 703.

---

## 4. Key Findings and Alignment Insights

### 1. The Factuality Coverage Trap
Both models attempt >95% of factual prompts on SimpleQA, despite having less than a 6% actual correctness rate. In standard generation, Aya models are strongly biased to answer rather than abstain, leading to severe hallucination rates exceeding 84% on PreciseWikiQA.

### 2. Epistemic Scaling Trends
Scaling from Tiny Aya (3.35B) to Aya Expanse (8B) yields consistent improvements in safety and epistemic calibration:
- Joint task success on AbstentionBench jumps **+8.99%** (51.23% → 60.22%).
- Unknown-label F1 on SelfAware increases **+9.63%** (34.07% → 43.70%).
- Hallucination when attempting to answer drops by **7.30%** (91.94% → 84.64%).
However, even at 8B scale, spontaneous abstention remains inadequate without targeted intervention.

### 3. Gating Before Generation (Gate 1)
Representation probes extracted from internal model activations (Layer 18) contain separable epistemic signals regarding query familiarity. By acting as an early refusal gate before response autoregression begins, SafeTune effectively suppresses ungrounded generations without degrading known factual recall.

---

## 5. Tool Chain Implementation (Lexsi Toolkits)

This work connects three official Lexsi tools in an end-to-end alignment pipeline:
- **CuratorKIT:** Prepared balanced, group-isolated splits (120 train / 40 calibration / 40 development) preventing question-group leakage across splits.
- **SafeTune:** Hosted the calibrated Layer-18 probe router in FP32, mapping hidden states to an abstention decision (`I don't know.` vs. pass-through).
- **AuditKIT:** Provided standardized evaluation schemas, running 10 multi-arm comparisons across 400 records and 7,600 indicators to measure behavior accuracy and joint task success.

---

## 6. Methodology, Provenance, and Reproducibility

- **Inference Setup:** Greedy one-pass decoding, native chat templates, FP16 precision.
- **Judge Setup:** GLM-5.3-Flash via hosted API, zero reasoning randomness, strict final-label extraction. Exactly 6,835 evaluation calls completed with zero errors.
- **Hash Identifiers:**
  - `benchmark-results.json` SHA-256: `c741366272671a0cfd27659ac04a1a64c25dd676a6bad3363068b8c4e90d4955`
  - Tiny Aya responses: `aa7a20232e7652d857433cc93efaf4dcfbc9f212ca28bb3ab6c6775444d9077f`
  - Aya 8B responses: `10cec296f90e9f77be709a7c7643532acf5750786b16047b2e4d9ba1c22175a8`
- **Cost Accounting:** Total actual judge spend is \$3.889 across 9,494 settled calls under the strict \$18 ceiling.
