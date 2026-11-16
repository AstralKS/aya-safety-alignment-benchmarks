# 10 — The rest of Lexsi Labs: other repos worth knowing

> The six hackathon tools are not the whole org. This is the **public listing observed on
> Nov 1, 2026**, plus README/API-level notes on possible extensions. It is not a comprehensive
> maturity audit, proof of maintenance, or a recommendation to expand the three-tool Track 3
> critical path. No additional-repo model runs were executed.

---

## 1. Observed inventory (GitHub API, sorted by last update)

| Repo | Stars | Language | License | Updated | One-line |
|---|---:|---|---|---|---|
| `Lexsi-Labs/aligntune` | 47 | Python | MIT | Oct 31 | post-training toolkit (hackathon) |
| `Lexsi-Labs/CuratorKIT` | 32 | Jupyter/Python | LSAL 1.1 | Oct 30 | data curation + synthesis (hackathon) |
| `Lexsi-Labs/AuditKIT` | 15 | Python | LSAL 1.2 | Oct 30 | evaluation suite (hackathon) |
| `Lexsi-Labs/github-issues-slack` | 0 | Python | — | Oct 29 | posts issues from 6 repos to Slack |
| `Lexsi-Labs/Lexsi-sdk` | 2 | Python | MIT | Oct 29 | SDK for the commercial console.lexsi.ai platform |
| `Lexsi-Labs/TabTune` | 140 | Python | MIT | Oct 25 | unified tabular + time-series foundation-model library |
| `Lexsi-Labs/SafeTune` | 20 | Python | LSAL 1.2 | Oct 23 | safety drift audit/repair (hackathon) |
| `Lexsi-Labs/AgentTune` | 11 | Python | LSAL 1.1 | Oct 23 | agentic post-training (hackathon) |
| `Lexsi-Labs/CircuitKIT` | 29 | Python | LSAL 1.2 | Oct 23 | mechanistic interpretability (hackathon) |
| `Lexsi-Labs/xai_evals` | 18 | Python | MIT | Oct 10 | explanation-method evaluation framework |
| `Lexsi-Labs/Orion-MSP` | 51 | Python | MIT | Oct 5 | tabular foundation model (multi-scale sparse attention) |
| `Lexsi-Labs/alignbeam` | 0 | — | — | Oct 4 | inference-time alignment transfer via cross-vocab logit mixing (code coming soon) |
| `Lexsi-Labs/Orion-BiX` | 16 | Python | MIT | Sep 29 | bi-axial meta-learning tabular foundation model |
| `Lexsi-Labs/DLBacktrace` | 28 | Python | LSAL | Sep 28 | layerwise relevance/attribution; documented support across several modalities, incl. MoE |
| `Lexsi-Labs/c-theta-delta-demo` | 0 | HTML | — | Mar 16 | static C-ΔΘ safety-steering dashboard (Llama-3.2-1B) |
| `Lexsi-Labs/AryaXAI-SDK` | 4 | Python | MIT | 2025-11 | SDK for the commercial AryaXAI components |
| `Lexsi-Labs/.github` | 0 | — | — | 2025-11 | org profile |

The hackathon's six tools are the middle block. Everything else is either infrastructure,
commercial SDK, another research line (tabular), or an interpretability/explainability extension.
Stars and GitHub `updated_at` are snapshot metadata, **not** last source-commit dates, passing
tests or active-maintenance guarantees. The separate clone pins are in
[`13-research-notes.md`](13-research-notes.md); unpinned listing-only entries were not source-audited.

---

## 2. The ones that matter for this hackathon

### 2.1 `c-theta-delta-demo` — the idea behind SafeTune's `CThetaTrainer`

A static HTML dashboard ("AI Safety Steering Demo", Llama-3.2-1B-Instruct) presenting
**C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal**:

- **Reported method/results**: localize refusal-related computations with **EAP-IG** attribution;
- compute a constrained weight update **ΔΘ confined to that circuit**, touching typically
  **<5% of parameters**;
- produce a standard edited checkpoint **without additional inference-time hooks**; this does
  not remove ordinary serving cost or establish agent safety;
- results presented as harmful vs benign refusal-rate trade-off, with the method clustering in
  the ideal high-harmful-refusal / low-benign-refusal zone.

Why it matters:

- SafeTune's feature map labels `apply_ctheta` (+ `_from_state_dicts` +
  `sweep_ctheta_strength`) `Faithful`: an upstream mechanism badge, not our reproduction of
  safety results or proof of equivalence with CircuitKIT's implementation.
- CircuitKIT implements the intervention side as `CircuitWeightSteering` (C-DTheta).
- The paper is on HF as *C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal*
  (Lexsi, 2026).

This motivates a **proposed extension**, not a ready-made outcome: circuit discovery →
circuit-restricted edit → controlled causal validation → separate security/utility Pareto.
The [paper](https://arxiv.org/abs/2602.04521) and static dashboard report chat-refusal experiments;
transfer to Tiny Aya tool decisions is unverified. See `12-neuroai-angles.md`.

### 2.2 `DLBacktrace` (28★)

Model-agnostic explainability: layerwise relevance scores and tracing for CNNs, RNNs,
Transformers, LLMs, and MoE models (JetMoE, OLMoE, Qwen3-MoE, GPT-Oss), across tabular, vision,
and text. DLB v2 reports 20×–1400× speedups over v1 on Llama-3.2-3B/MMLU (RTX 4090), with
token-level relevance tracking and an AutoSampler for decoding strategies. Python ≥3.8,
torch ≥2.6, `pip install dl-backtrace`. License: LSAL.

Hackathon use: **cross-method explanation agreement**. CircuitKIT attributes at the
head/MLP level; DLBacktrace attributes layerwise relevance and token-level traces. If both
methods flag the same layers, that is **agreement**, not proof that the explanation is correct:
methods can share confounds. Compare units/normalization, then test matched random/layer controls
and causal effects on **both** security and utility. Novelty was not established; xai_evals
does not supply a plug-and-play circuit metric. This is outside the three-tool core.

Risk: not part of the six-tool stack, so judges may not weight it; use as an optional extension
after the required before/after table exists.

### 2.3 `xai_evals` (18★, MIT)

An evaluation framework for **post-hoc local explanation methods** (technical report
arXiv:2502.03014). It wraps SHAP, LIME, Torch/TF tabular explainers, and a DLBacktrace
explainer, then computes explanation-quality metrics for tabular and image tasks:
`ExplanationMetricsImage` (robustness, faithfulness, complexity, etc.) and
`ExplanationMetricsTabular`.

Possible use: adapt explanation-quality metrics beyond a graph. CircuitKIT measures task-level
*circuit* faithfulness; xai_evals targets *local explanations* for other modalities. A valid
cross-method measurement requires a specified mapping and causal controls; neither agreement
nor a faithfulness score is an executable agent-safety outcome.

Risk: its metrics are aimed at SHAP/LIME-style local explanations and tabular/image models, not
head-level circuits; you would be adapting them. Treat it as a research extension, not a
requirement.

### 2.4 `alignbeam` (0★, placeholder)

"AlignBeam: Inference-Time Alignment Transfer via Cross-Vocabulary Logit Mixing" — code coming
soon. No implementation. Interesting concept for a moonshot (transfer an aligned teacher's
logit distribution to a student with a different tokenizer), but unusable today. Mention it in a
"future work" section; do not plan around it.

### 2.5 `Lexsi-sdk` (2★, MIT)

SDK for the commercial Lexsi.ai observability platform (explainability, drift monitoring,
synthetic data, risk policies, AutoML) via `lexsi_sdk.xai` + an access token. Not needed for the
hackathon; AuditKIT has a `lexsi:` model backend (`lexsi_login()` / `run_benchmark`) if you ever
want to compare against the hosted platform. Confirm API access/cost separately; the rules
allow the listed source-available toolkits and require eligible Cohere subjects/HF artifacts.

### 2.6 `AryaXAI-SDK` (4★, MIT)

SDK for the commercial AryaXAI components. Same story as Lexsi-sdk: not hackathon material, but
evidence that Lexsi's explainability tooling has an enterprise surface.

### 2.7 `github-issues-slack` (0★)

Automation described as posting issues from the six repos into Slack. That signals intended
issue routing, not proof that it currently runs, that maintainers read every issue, or that a
response/fix is guaranteed. File precise reproductions through the documented issue/security
channels; this review did not verify service operation.

---

## 3. The other research line: tabular foundation models

### 3.1 `TabTune` (140★, MIT) — the flagship

A README-described library for inference and fine-tuning of **21 tabular model entries** (TabPFN
v2/v2.6/v3/v3.5/fast, TabICL/v2, OrionMSP/v1.5, OrionBiX, Mitra/v2, ContextTab, TabDPT, LimiX,
TabFM, xRFM, iLTM, EXAONE Tabular, Causilo, TabLDM) plus **nine time-series foundation models**
(Chronos/Bolt/2, TimesFM 3, Toto 1/2, TiRex/2, Time-MoE) with forecasting, anomaly detection,
imputation, and embeddings. Core components: `DataProcessor`, `TuningManager`,
`TabularPipeline`, `TabularLeaderboard`, `TimeSeriesPipeline`. Strategies: zero-shot inference,
episodic meta-learning, SFT, PEFT/LoRA; includes a deployment-aware registry that checks weight
licenses and capability envelopes before download. Support/strategy behavior and license
enforcement were not executed here; the README itself notes experimental/no-op PEFT routes.
The [original paper](https://arxiv.org/abs/2511.02802) describes seven models, not this later
21-entry/nine-time-series catalog.

Why it is not in the six-tool hackathon stack: the three tracks are about LLM post-training,
safety and agents, and the model rule lists dense Cohere LLMs. The homepage mentions tabular
practitioners, but its now-published statements do not make Orion/TabTune a permitted substitute
subject. TabTune could support a caller-built business-data tool, with separate permission and
license checks; it is not needed for AgentGuard. Re-check organizer updates before expanding scope.

### 3.2 `Orion-MSP` (51★) and `Orion-BiX` (16★)

The tabular foundation models behind TabTune's `OrionMSP`/`OrionBix` entries:

- **Orion-MSP**: multi-scale sparse attention + Perceiver-style cross-component memory for
  efficient tabular in-context learning (arXiv:2511.02818). Scales to large feature sets with
reported scalable width processing; sklearn-compatible according to upstream. No complexity
or accuracy benchmark was reproduced here.
- **Orion-BiX**: bi-axial attention (standard → grouped → hierarchical → relational + CLS
  aggregation) with episode-based meta-learning; targets healthcare/finance few-shot tabular
  classification (arXiv:2512.00181).

Relevance: none to the six-tool chain, but they are HF-hosted Lexsi models
(`Lexsi/Orion-MSP`, `Lexsi/Orion-BiX`) and show the org's tabular strength. A "safety × tabular"
project (e.g., fairness auditing of tabular models with AuditKIT-style metrics) would be a
different kind of submission; it is not what the tracks ask for.

---

## 4. Recommended optional extensions (ranked)

| Rank | Extension | Effort | Upside | Risk |
|---|---|---|---|---|
| 1 | **C-ΔΘ / circuit-restricted edits** (SafeTune + CircuitKIT) | Medium–High | Independent subject reproduction/transfer if validated, not a new method | Model parity, compatible references, causal controls and budget |
| 2 | **DLBacktrace triangulation** (extra Lexsi repo) | Medium | High for a research-flavored write-up | outside the required 6; may confuse judges unless framed as bonus |
| 3 | **xai_evals explanation metrics** | Medium-High | Strong methodology section | adaptation work; not designed for circuits |
| 4 | **alignbeam** | — | Future-work section only | no code |
| 5 | **TabTune / Orion** | High | Only if tabular statements appear | off-track; different pipeline entirely |
| 6 | **Lexsi-sdk / AryaXAI-SDK** | Low–Medium | Hosted-platform comparison footnote | accounts/costs; not core research toolkits |

**Rule of thumb:** the real **CuratorKIT → AgentTune → AuditKIT** trained-policy comparison,
frozen outcomes and released-artifact re-test come first. All listed extensions are optional;
Track 3 is excepted from the two-track-repository minimum. The shared budget funds only one
bounded optional analysis, not all rows. See `08-execution-plan.md`.

---

## 5. What the repo inventory tells you about the organizers

- The six repos expose conventional library documentation, packaging and tests; recency/star
  counts alone do not establish maintainability or passing coverage.
- An observable **documentation practice**: SafeTune's faithfulness badges, CircuitKIT's
  stability tiers, AgentTune's known-issues page, AuditKIT's known issues, AlignTune's
  `ISSUES.md`. A submission that reports failures and limits explicitly is aligned with the
  evidence quality, without claiming to know the judges' preferences.
- **Provenance metadata** appears across all six tools. Preserve it and add actual revisions,
  byte hashes and a validator; their reviewed writers use `git_sha: None`, and nested metadata
  is not a cryptographically bound lineage.
- The org's flagship research (C-ΔΘ, tabular foundation models, Tiny Aya integrations) shows
  where their scientific interest is. A project that reads like a mini-paper (pre-registered
  metrics, appropriate uncertainty and an honest negative result) is a recommended presentation,
  not a predicted judging outcome.
