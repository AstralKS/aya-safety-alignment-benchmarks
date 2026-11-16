# 12 — Representation research: probes, geometry and controlled interventions

Read after [11 — Prior art](11-community-prior-art.md). These are **optional research hypotheses**, not implemented detectors or demonstrated Aya results. “NeuroAI” is useful here as inspiration for studying representations, not as a claim that a model thinks or feels like a brain.

## 1. Three different claims—do not collapse them

| Claim | Example | What would support it? |
|---|---|---|
| **Observation** | Two prompt groups have different activations | Reproducible measurements under controlled inputs |
| **Prediction** | A feature predicts unsupported model answers | Group-held-out labels and stronger-than-shortcut baselines |
| **Causal intervention** | Changing a component reduces those errors selectively | Matched interventions and independent safety/utility outcomes |

The notebooks' I01 capture is at the first level. It does not become a detector by plotting it, or a circuit by naming a layer. [SafeTune](04-safetune.md) supplies some measurement/intervention primitives; [CircuitKIT](05-circuitkit.md) adds causal tests. Neither supplies ground truth about source support automatically.

## 2. Most relevant angle: a support/error probe

A **probe** is a small predictor trained on a model's internal features. A logistic probe uses a weighted combination of activation values to predict a binary label. It can test whether a feature carries useful information without fine-tuning Aya itself.

For candidate 1 in [07](07-project-idea.md), the question is: **does an internal feature help predict an unsupported answer, beyond strong prompting and simple text features?**

### Define the prediction point first

- **Before generation:** use a declared final-prompt-token activation to predict whether the *future answer* will contain an unsupported claim. This is a response-level forecast, not inspection of a claim that has not yet been written.
- **After a draft:** evaluate a particular candidate claim together with its source. Features may include that actual draft, but not its gold support label or a corrected answer. This is a draft-review detector, not an early-warning result.

Choose one. A detector trained after seeing the answer cannot be advertised as predicting the error before generation.

### A small, testable design

1. Collect **real Aya outputs**, including actual unsupported errors and supported answers. Human-review source support, attribution and task eligibility separately.
2. Declare the layer, token/span, chat template and feature extraction point. Last **non-padding** token selection matters in batches; record truncation.
3. Fit on training groups and use separate development groups for layer selection, probability calibration and answer/abstain thresholds.
4. Compare with strong system prompting, language/length/template features and a text-based classifier. A probe may just recognize unfamiliar scripts or missing-evidence wording.
5. Freeze the gate and test supported-task quality plus unsupported assertions at different **coverage** levels—the fraction of tasks it answers.

Report discrimination (AUROC and, with rare errors, precision–recall), calibration and the **risk–coverage curve**: error among answered items versus how many are answered. **Calibration** asks whether predicted error probabilities match observed frequencies. High AUROC does not guarantee that match or a useful abstention policy. Report class counts; AUROC is undefined if the evaluated slice has only one class.

Released Neuron Lens R/S lenses are Qwen/Gemma-specific. Their existence does not give us an Aya support probe; representation extraction, labels, training and validation remain project work.

## 3. Geometry: does one refusal direction generalize across languages?

[SafeTune](04-safetune.md) introduced a difference-in-means direction. Estimate one on each reviewed language/script slice of the **same checkpoint**, with harmful and benign intents matched.

Compare directions using **cosine similarity**, which measures vector alignment. Optionally use **PCA**, which finds directions of high variation, to ask whether the examples occupy more than one dominant direction. High variance is not necessarily safety information.

The useful experiment connects geometry to behavior: does a direction estimated in English preserve harmful/benign discrimination in Hinglish? Does a language-specific direction improve utility at matched safety?

Controls matter: prompt length, script, topic and template can explain alignment differences. Vectors from Tiny Aya and Expanse do not automatically share a coordinate system; direct cross-model cosine or direction transfer requires an explicit mapping.

**Good result:** a bounded, held-out relation between direction alignment and steering effects. **Overclaim:** “we found the multilingual safety manifold” from a two-dimensional plot.

## 4. Causal angle: localized or gated intervention versus a blunt one

Choose one demonstrated failure, then compare:

- No intervention and the strong prompt baseline.
- Fixed-strength steering.
- A development-calibrated gate that applies steering only to selected cases, **or** a CircuitKIT-localized intervention.
- Random directions/components matched for size, layer and intervention magnitude.

Score the **actual wrapper** on both the target error and legitimate tasks. SafeTune's reviewed live utility helper uses the unsteered base, so it does not measure this cost.

For claim grounding, refusal is only a baseline mechanism; a useful source-support intervention requires its own contrasts and labels. For language safety, gated steering already has close prior art in `11`; the contribution must be a specific measurement or extension, not the gate's existence.

If CircuitKIT parity fails, drop that circuit claim. A selective effect supports a narrow causal intervention result, not a unique safety circuit or proof that the probe found “truth.”

## 5. Optional anomaly features: surprise and spectral entropy

**Token surprise** is negative log probability: text the model finds unlikely receives a higher score. A fluent unsupported claim can be unsurprising; a correct rare-language answer can be surprising. Likelihood is not evidence support.

**Spectral entropy** summarizes how spread out a matrix's singular-value spectrum is. This concerns representation structure, not the same thing as token uncertainty. SafeTune's `SpectralEntropyMonitor` calibrates a reference distribution and returns flags such as `(prompt_idx, layer_idx, entropy, z_score)`, not probabilities of hallucination or harm.

Treat either as a candidate feature only. Compare against language/length/template baselines, fit calibration on separate groups, measure extra forward-pass/storage costs and test whether adding the feature actually helps the gate. An anomaly score alone is not a safety certificate.

## 6. What to prioritize

For selective abstention, first establish source-grounded behavioral labels and a strong prompt baseline; then try **one probe/gate**. For code-mixed safety, establish validated behavior differences; then test **one fixed-versus-gated steering question**. Geometry or localization can explain that result later.

Use `07`'s grouped splits. Keep gold labels, corrected answers and post-decision verdicts out of predictive features. Lock selections before final test; preserve failed predictions, false abstentions and negative causal controls.

One risk–coverage plot or a matched causal-control comparison is more informative than several unrelated heatmaps.

**Next:** [13 — Sources and verification](13-research-notes.md), the preserved historical evidence ledger; then [14 — Relevant papers](14-lexsi-papers.md) for deeper reading. The current project shortlist remains `07`, not older framing inside the ledger.
