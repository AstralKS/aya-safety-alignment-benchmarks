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

# Aya Hallucination, Factuality, and Selective Abstention: A Rigorous Multi-Benchmark Evaluation

**Lexsi AI Safety & Alignment Hackathon (Track 2: AI Red Teaming, Safety & Alignment)**  
**Models Evaluated:** `CohereLabs/tiny-aya-global` (3.35B) & `CohereLabs/aya-expanse-8b` (8B)  
**Author:** AstralKS (Hugging Face: [ASTRALK](https://huggingface.co/ASTRALK) | GitHub: [AstralKS](https://github.com/AstralKS))  
**Artifact Collections:**
- **Hugging Face Model Collection:** [Aya Safety & Hallucination Evaluations](https://huggingface.co/collections/ASTRALK/aya-safety-and-hallucination-evaluations-models-6afaec3a8d5a2fd6533d560b)
- **Hugging Face Dataset Collection:** [Aya Safety & Hallucination Benchmarks & Traces](https://huggingface.co/collections/ASTRALK/aya-safety-and-hallucination-benchmarks-datasets-6afaec3b152fb3a00d5aed2f)
- **GitHub Repository:** [AstralKS/aya-safety-alignment-benchmarks](https://github.com/AstralKS/aya-safety-alignment-benchmarks)

---

## 1. Abstract (<500 words)

Multilingual open-weights foundation models frequently exhibit high surface fluency alongside severe ungrounded hallucination, particularly in knowledge-intensive regimes. In this project for the Lexsi AI Safety & Alignment Hackathon, we present a frozen, empirical evaluation of Cohere's dense open-weights multilingual models—`CohereLabs/tiny-aya-global` (3.35B) and `CohereLabs/aya-expanse-8b` (8B)—across four distinct safety, truthfulness, and factuality benchmarks comprising 6,347 evaluated test instances per model (12,694 model generations in total). 

The benchmark battery covers four key complementary axes:
1. **Short-Form Factuality:** SimpleQA Verified (1,000 cases) measuring precision and coverage under factual queries.
2. **Hallucination vs. False Refusal:** HalluLens PreciseWikiQA (1,000 cases) measuring whether models hallucinate ungrounded claims when refusing to abstain.
3. **Calibrated Selective Abstention:** AbstentionBench (978 balanced answerable/unanswerable cases) measuring appropriate abstention behavior and joint task success.
4. **Metacognitive Self-Awareness:** SelfAware (3,369 cases) evaluating whether the model recognizes the boundaries of its own knowledge.

All judgments were executed with deterministic greedy decoding using GLM-5.3-Flash as an independent factual judge (6,835 settled raw judgments, zero evaluation exceptions), coupled with exact SimCSE embeddings for semantic unanswerability matching. 

Our empirical results reveal:
- **Low Factuality Baseline:** SimpleQA correct-answer yield is only 3.70% (Tiny Aya) and 5.70% (8B) despite answer coverage exceeding 95.2%, highlighting that models aggressively attempt queries where they lack factual grounding.
- **Pervasive Ungrounded Claims:** On HalluLens PreciseWikiQA, when models do not abstain, the hallucination rate reaches 91.94% for Tiny Aya and 84.64% for 8B, while false refusals remain rare (3.20% and 4.30%).
- **Scaling Benefits in Abstention:** On AbstentionBench, Aya Expanse 8B demonstrates significant gains in appropriate abstention over Tiny Aya (72.70% vs. 68.30% behavior accuracy; 60.22% vs. 51.23% joint task success).
- **Metacognitive Recognition:** On SelfAware, scaling from 3.35B to 8B improves unknown-label F1 from 34.07% to 43.70%, with answerable reference accuracy improving from 32.04% to 39.01%.

We analyze these results within the Lexsi alignment ecosystem (CuratorKIT data filtering, AuditKIT safety evaluation, and SafeTune steering). All datasets, model outputs, raw judge journals, and reproducible evaluation code are publicly available.

---

## 2. Benchmark Battery and Metric Design

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

## 3. Key Findings and Alignment Insights

### 1. The Factuality Coverage Trap
Both models attempt >95% of factual prompts on SimpleQA, despite having less than a 6% actual correctness rate. In standard generation, Aya models are strongly biased to answer rather than abstain, leading to severe hallucination rates exceeding 84% on PreciseWikiQA.

### 2. Epistemic Scaling Trends
Scaling from Tiny Aya (3.35B) to Aya Expanse (8B) yields consistent improvements in safety and epistemic calibration:
- Joint task success on AbstentionBench jumps **+8.99%** (51.23% → 60.22%).
- Unknown-label F1 on SelfAware increases **+9.63%** (34.07% → 43.70%).
- Hallucination when attempting to answer drops by **7.30%** (91.94% → 84.64%).
However, even at 8B scale, spontaneous abstention remains inadequate without targeted intervention.

---

## 4. Connection to Lexsi AI Safety Toolkits

This empirical benchmark framework directly interfaces with the Lexsi Toolkit suite:
- **CuratorKIT:** Data preparation and filtering protocols for grounding verification, source attribution, and split hygiene.
- **AuditKIT:** Principled safety evaluation criteria, distinguishing harmful failure modes (hallucination under false confidence) from benign failures (abstention).
- **SafeTune:** Identifying safety drift and providing target directions for representation steering (e.g., refusal direction steering and abstention calibration).

---

## 5. Methodology, Provenance, and Reproducibility

- **Inference Setup:** Greedy one-pass decoding, native chat templates, FP16 precision.
- **Judge Setup:** GLM-5.3-Flash via hosted API, zero reasoning randomness, strict final-label extraction. Exactly 6,835 evaluation calls completed with zero errors.
- **Hash Identifiers:**
  - `benchmark-results.json` SHA-256: `c741366272671a0cfd27659ac04a1a64c25dd676a6bad3363068b8c4e90d4955`
  - Tiny Aya responses: `aa7a20232e7652d857433cc93efaf4dcfbc9f212ca28bb3ab6c6775444d9077f`
  - Aya 8B responses: `10cec296f90e9f77be709a7c7643532acf5750786b16047b2e4d9ba1c22175a8`
- **Cost Accounting:** Total actual judge spend is \$3.889 across 9,494 settled calls under the strict \$18 ceiling.
