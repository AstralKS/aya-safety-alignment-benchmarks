# 07 — Project ideas: choose one question for the two Aya models

**Updated:** Nov 2, 2026. **Status:** superseded 2026-11-15. The candidate shortlist below is still a
usable menu, but the project is no longer unselected: what actually got built is a four-benchmark
abstention/hallucination baseline on both Aya models (`part 1/baselines/benchmark.md`), and it uses
none of the six Lexsi toolkits. The open problem is therefore not "choose an idea" but "chain ≥2
track tools around the result we already have and add a before-and-after arm" — see `AGENTS.md`
FINAL STATE. Note that §2's compute and §5's "what to do next" describe the Nov 1–8 state.

**Reading order:** 01 → 02 → 03 → 04 → 05 → 07 → 11 → 12 → 13 → 14. Read this after the tool guides; `11` adds prior art and `12` adds optional representation research.

## 1. What the tools now let us do

| Tool | Meaningful role—not a requirement to use all five |
|---|---|
| [AlignTune](01-aligntune.md) | Train Aya on demonstrations, preferences or rewards |
| [CuratorKIT](02-curatorkit.md) | Prepare traceable, quality-reviewed data |
| [AuditKIT](03-auditkit.md) | Score declared outcomes and compare conditions |
| [SafeTune](04-safetune.md) | Apply safety-specific training, recovery or live steering |
| [CircuitKIT](05-circuitkit.md) | Test a localized mechanism or guide a targeted intervention |

The eventual submission must meaningfully use **at least two Lexsi tools**. For an evaluated corpus, CuratorKIT + AuditKIT can be relevant; add SafeTune for steering or AlignTune for actual trained repair. CircuitKIT is optional unless mechanism localization is the question. The current Transformers-only notebooks are exploration, not a complete submission.

## 2. Models and compute

| Model | Size | Why explore it? |
|---|---|---|
| `CohereLabs/tiny-aya-global` | 3.35B | Lower-cost repeated experiments and a practical first intervention subject |
| `CohereLabs/aya-expanse-8b` | 8B | Same-case comparison on a larger, differently trained Aya model |

Tiny Aya uses `cohere2`; Expanse uses `cohere`. Differences are **not a controlled model-size-only experiment**: architecture, tokenizer and training also differ.

Start unquantized: **FP16 on T4, BF16 on compatible newer GPUs**. Both notebooks default to `LOAD_IN_4BIT = False`. NF4 is a separately labeled comparison arm, not an interchangeable baseline.

**Resources:** two Kaggle accounts for the initial runs; $20 HF credit alongside Kaggle where the covered service/hardware suits the task; $50 Modal reserve for jobs that genuinely need it. Credit is not a known GPU rental or a measured training-time guarantee. Accept both gated models' conditions and use your own `HF_TOKEN`.

The notebooks use Accelerate `device_map="balanced"` layer sharding for inference memory. That does not double generation speed, configure distributed training or establish CircuitKIT fit. GPU loading and generation remain untested here.

## 3. Six directions worth understanding

### A. Multilingual and code-mixed safety

**Question:** does the same request receive different safety treatment across English, Hindi, Romanized Hindi and Hinglish?

Build human-validated, intent-matched variants with benign sensitive-content controls. Score **harmful assistance, over-refusal and language fidelity** separately. Narrow code-switch/script effects are more informative than another broad refusal-rate table.

**Tools:** CuratorKIT + AuditKIT; SafeTune for a repair arm. **Main risk:** close multilingual refusal prior art and non-equivalent translations.

### B. Inference-time steering

**Question:** can an internal activation intervention reduce a demonstrated failure without retraining Aya?

Compare real steering against neutral and strong-system-prompt baselines; select direction/layer/strength on development data. Preserve benign utility. Prompt changes are not activation steering, and a refusal direction is not automatically a grounding direction.

**Tools:** SafeTune + AuditKIT. **Main risk:** buying apparent safety through blanket refusal.

### C. Locating safety mechanisms

**Question:** which components contribute to a precisely defined safety or source-support decision?

Pass Cohere parity, build a controlled contrast, then use patching/ablation and matched random controls. Raw layer activations are only observations; another model's successful circuit is not Aya evidence.

**Tools:** CircuitKIT + AuditKIT; optionally SafeTune for a targeted intervention. **Main risk:** port compatibility and discovery memory. Keep it off the critical path initially.

### D. Safety-aware compression

**Question:** does lower precision change safety/grounding, and can safety-informed selective compression preserve it?

First compare matched unquantized versus NF4 cases, with safety, utility, memory and latency. Ordinary quantization is **not** safety-aware compression; that name requires safety information to guide the compression strategy.

**Tools:** AuditKIT + a meaningful SafeTune or CircuitKIT intervention. **Main risk:** mistaking a precision audit for a new safety-aware method. A bounded precision check is a reasonable secondary axis.

### E. Tamper resistance

**Question:** does safety survive later weight modification or fine-tuning?

Use an actual weight/training attack with a declared budget, then compare defended and undefended checkpoints. Prompt pressure is not weight tampering, and repairing one attacked model does not prove resistance to another attack.

**Tools:** SafeTune + AuditKIT; CuratorKIT/AlignTune as the chosen data/training design requires. **Main risk:** training cost and an ill-defined attack. Defer until a baseline works.

### F. Hallucination detection and selective abstention

**Question:** can Aya answer supported parts and decline unsupported claims without refusing the whole task?

Begin with source-grounded extraction: attribution, qualifiers, numbers, negation and missing evidence. Collect real model errors and compare strong prompting with a calibrated support detector or answer/abstain gate. Internal-feature detectors are additional project work, not a built-in notebook feature.

Faithfully reporting a false allegation can be correct extraction. **Source support and real-world truth need different labels.**

**Tools:** CuratorKIT + AuditKIT; AlignTune for trained demonstrations, SafeTune for an appropriate intervention, CircuitKIT for optional causal analysis. **Main risk:** a gate that answers almost nothing, or a probe that learns document/language shortcuts.

## 4. My recommended shortlist—not a selected project

| Candidate | Small, coherent build | Primary evidence |
|---|---|---|
| **1. Claim-grounded selective abstention** | Labeled supported/unsupported cases → strong prompt baseline → calibrated gate or trained repair | Unsupported assertions at comparable supported-task coverage and quality |
| **2. Code-mixed safety with targeted steering** | Validated language/script failures → real steering → benign controls | Less harmful assistance without excessive refusal or loss of language fidelity |

I favor candidate 1 as the starting question because its evidence can be inspected directly. Candidate 2 is attractive if both notebooks reveal consistent, human-validated language-dependent failures. **Choose one after exploration; do not combine all six.**

Related documents, paraphrases, translations and code-mixed variants stay in one split. Fit on training/calibration groups, choose the method on separate development groups, then freeze a final test. Keep empty/truncated/error cases visible and use group-aware paired uncertainty.

## 5. What to do next

Use [notebook setup](../notebooks/README.md) and the [prompt/rubric guide](../notebooks/PROMPTS.md), then run:

- [Tiny Aya Global playground](../notebooks/tiny-aya-global-kaggle.ipynb)
- [Aya Expanse 8B playground](../notebooks/aya-expanse-8b-kaggle.ipynb)

Start with H01/H04/H05 for grounding and useful abstention, M01–M05 for language/script behavior and benign utility, and I01 for an activation contrast. Match settings and record actual failures before designing a repair.

The notebooks passed local structure/code/helper checks only. Last-prompt-token captures are not trained detectors, live steering or causal circuits. Released Neuron Lens R/S lenses are Qwen/Gemma-specific and need genuine Cohere adaptation.

`06` and the older `08`/`09` plans are not required for this shortlist. These candidates concern safety/alignment; confirm the submission framing and any registration change with the organizers rather than assuming it has happened.

**Next:** [11 — Relevant prior art](11-community-prior-art.md), then [12 — Representation research](12-neuroai-angles.md).
