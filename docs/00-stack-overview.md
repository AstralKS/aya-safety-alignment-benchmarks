# 00 — The Lexsi stack, end to end

> One page to reason about all six tools before committing the remaining build time.
> Detailed guides: `01-aligntune.md` … `05-circuitkit.md`; `06-agenttune.md` is an archived
> reference outside the current shortlist. Candidate projects: `07-project-idea.md`.

---

## 1. The life-cycle picture

The six tools are not six competitors; they are stations on one model life-cycle.

```
             raw data / documents / HF datasets
                        │
          ┌─────────────▼─────────────┐
          │  CuratorKIT                │  ingest → clean/dedup → hygiene → generate →
          │  data curation + synthesis │  quality gates → adaptive recovery → exports
          └─────────────┬─────────────┘
                        │  trainer-ready JSONL / HF dataset folder (+ provenance)
          ┌─────────────▼─────────────┐      ┌──────────────────────────────┐
          │  AlignTune                │      │  AgentTune                   │
          │  SFT / DPO / GRPO / PPO…  │      │  agent rollouts, GRPO/DPO/   │
          │  TRL · Unsloth · ES       │      │  RLOO/BCO, distill, RAG, heal│
          └──────┬───────────┬────────┘      └──────────────┬───────────────┘
                 │           │                              │ trajectories
                 │           │  checkpoints                 │
                 │  ┌────────▼─────────┐                    │
                 │  │  SafeTune        │  diagnose: refusal direction, spectral monitor,
                 │  │  safety drift    │  safety neurons; repair: harden / recover /
                 │  │  audit + repair  │  unlearn / steer; selected-judge diagnostic suite
                 │  └────────┬─────────┘
                 │           │
                 │  ┌────────▼─────────┐
                 │  │  CircuitKIT      │  discover circuit → 6-pillar faithfulness →
                 │  │  mech. interp    │  prune / quantize / edit / steer / selective FT
                 │  └────────┬─────────┘
                 │           │  circuit artifacts (`safety_units`, `layer_suggestions`)
                 └───────────┼──────────────► AuditKIT  eval: benchmarks, judges, red team,
                             │                  agent/RAG, security, performance, provenance
                             ▼
                     Hugging Face Hub: models, adapters, logs, datasets, Space
```

Several pairwise integrations are real in source, but **no model-backed whole-stack round-trip
was executed in this review**:

- **CuratorKIT → AlignTune / SafeTune / AuditKIT**: exported HF configs are declared in the
  folder's `README.md`; use only compatible, nonempty formats and explicit splits. The relevant
  loaders can preserve/embed sibling provenance.
- **CircuitKIT → SafeTune**: CircuitKIT's `*_scores.json` embeds `safety_units` +
  `layer_suggestions`; SafeTune's adapter (`src/safetune/core/circuit_kit/adapter.py`) reads
  either those fields or raw `node_scores` for `CircuitInfo`/SafetyLoRA targeting.
  `safety_circuit_info()` instead derives info from a loaded model and contrast prompts.

AgentTune → AuditKIT is also first-class: `trajectories.jsonl` + `lexsi_provenance.json` in the
training output dir are read by `auditkit.agent_eval.episodes_from_agenttune`. Raw trajectories
do not supply verified final state or reliable cross-artifact case/trial IDs; join genuine
environment sidecars before making outcome claims.

---

## 2. What each tool is best at (and worst at)

| Tool | Best at | Avoid / weak spot | Static/upstream signal, not a maturity benchmark |
|---|---|---|---|
| **AlignTune** | Broad factory, CLI/YAML, tokenizer/merging/composition; optional alignment diagnostics and export regression | `safety` bundle resolves 0/3 tasks; audit callback is non-blocking; RAFT citation loss is a placeholder | 669 non-test / 93 test Python files; includes vendored code |
| **CuratorKIT** | Gates with rejection reasons, recovery loop, compatible exports, source/output checksums | No benchmark-decontamination or rights/version gate; judge quality matters; PDF extra is heavy | 82 non-test / 39 test Python files; 0.1.21 |
| **AuditKIT** | Zero-required-dep core, executable generic agent harness, outcome oracles, loss-aware importers | No bundled agent-security world; default red-team rate counts detector rows; bootstrap p-value is defective | 96 non-test / 155 test Python files; upstream test-pass claims not rerun here |
| **SafeTune** | Broad drift/intervention catalog, explicit upstream faithfulness badges, MART adaptive hardening, runtime primitives | Main suite uses one selected judge across benchmarks; WildGuard route scores refusal, not response harmfulness; no agent-security suite | 123 map rows, including 78 intervention rows; badges/results are upstream judgments/reports |
| **CircuitKIT** | 13 discovery algorithms, 6-pillar faithfulness, interventions and reloadable exports | Faithfulness is not safety; Tiny Aya is experimental/parity-gated; no multi-GPU discovery or saved steering hooks | 489 non-test / 192 test Python files; upstream 6 stable / 7 research tier |
| **AgentTune** | EventLog, five configurable agentic training paths, tools/RAG/distillation/healing | Exposed code tools have host authority; process isolation can fall back; task gate is not security verification | 343 non-test / 115 test Python files; README reports 43 notebooks, not rerun here |

Counts are files, not coverage, passing tests or independently measured reliability. See
[`15-missing-components.md`](15-missing-components.md) for source-backed gaps and conditional severity.

---

## 3. License map (read before publishing)

| Tool | License | Hackathon use | Keep in mind |
|---|---|---|---|
| AlignTune | MIT for own code; bundles Unsloth (Apache-2.0 **with AGPL-3.0 components**) and mergekit (LGPL-3.0) | Review all applicable terms | Prefer explicit `backend="trl"` for predictable execution; bundled code/distribution obligations still need review. Backend choice alone is not a compliance determination. |
| CuratorKIT | LSAL v1.1 (source-available, noncommercial) | Fine for the hackathon | Commercial use needs a license; PDF extra installs AGPL MinerU |
| AuditKIT | LSAL v1.2 | Fine | Organizations must acknowledge use (Section 1A); commercial license needed to sell/embed |
| SafeTune | LSAL v1.2 | Fine | Same Section 1A; **unrepaired drifted checkpoints may not be deployed in production** (Responsible Use clause) |
| CircuitKIT | LSAL v1.2 | Fine | Same Section 1A |
| AgentTune | LSAL v1.1 | Fine | Restricts commercial use and "removal/weakening of a model's safety behaviors without permission" — do not frame the project as an uncensoring tool |

Hackathon rules say projects stay open and teams retain IP. That does not override code, model,
dataset or redistribution terms. Check each artifact's license and any organizational
acknowledgment/permission requirement; this table is a reading guide, not legal advice.

---

## 4. Dependency and environment reality

Prefer separate environments for heavy stages; **a combined install was not tested**. This is
operational isolation, not a demonstrated incompatibility among all six packages:

- **Modern HF stack (`transformers>=5.15,<6`)**: AlignTune, SafeTune, CircuitKIT, AgentTune, and
  AuditKIT's `transformers` extra. AuditKIT's serving-extra constraints should be read separately
  from AgentTune's optional vLLM stack.
- **Zero-dep / light stack**: AuditKIT core (stdlib only), CuratorKIT core.
- **Python floors**: AuditKIT/CircuitKIT ≥3.10; AlignTune/CuratorKIT ≥3.11; SafeTune/AgentTune ≥3.12.
  Use **Python 3.12** as the common denominator.
- **Declared extra conflicts**: AuditKIT's `sglang` extra is mutually exclusive with `vllm` and
  `transformers` extras (its own `[tool.uv] conflicts` says so). Pick one serving path.
- **Torch/TRL constraints**: AgentTune pins `torch==2.11.0` and `trl==1.7.1`; SafeTune's
  `trl>=0.12,<1.15` admits that TRL version. Do not invent a TRL conflict. Resolve/freeze each
  real environment, check CUDA/native dependencies, and exchange explicit artifacts.

Linux/Bash install sketches (not tested locks); Windows activation uses
`.venv-<name>\Scripts\Activate.ps1`. Detailed separate-stage sketches are in doc 09:

```bash
# env A: minimal data + core eval
python -m venv .venv-data
source .venv-data/bin/activate
pip install "curatorkit[hf]" auditkit
deactivate

# optional SFT and safety stages each get their own environment; do not combine blindly

# env C: interpretability
python -m venv .venv-interp
source .venv-interp/bin/activate
pip install circuitkit            # imports circuitkit before transformer_lens
deactivate

# env D: agentic
python -m venv .venv-agent
source .venv-agent/bin/activate
pip install -e research/AgentTune # from this field-guide root; or pip install agenttune
```

**Verified import ordering trap (CircuitKIT):** import `circuitkit` *before* `transformer_lens`.
CircuitKIT patches TransformerLens 3.8 at import time; importing TL first can load an unpatched
model for `cohere2`/`cohere1` checkpoints.

---

## 5. Source-supported chains (model-backed runs still required)

### Track 1 chain (alternative project)

```
CuratorKIT ──(curated SFT/DPO dataset folder)──► AlignTune ──(checkpoint/adapter)──► AuditKIT
```

- CuratorKIT writes selected exports plus a `README.md` with HF config names. Instruction/response
  rows do not become DPO pairs; count compatible rows in every requested file.
- AlignTune accepts `dataset_name="output/"` + `config_name="sft_messages"`.
- AuditKIT `ak.load_dataset("eval/", config="sft_alpaca", split="test")` preserves metadata;
  `load_hf(..., name=...)` drops extra metadata. Use a separate evaluation artifact.
- Optional AlignTune auditor/callback provides diagnostics, not a mandatory release gate; its
  named safety bundle is broken at this pin.

### Track 2 chain (alternative safety-drift project)

```
CuratorKIT (safety mix + domain data)
      │
      ▼
SafeTune Harden (SafeGrad/LISA/…)  ──────┐
      │  or AlignTune SFT/DPO (drift)     │
      ▼                                   │
SafeTune Recover / Unlearn / Steer  ◄─────┘   (post-hoc weight-space or inference-time)
      │
      ├──► CircuitKIT discover → 6-pillar eval → circuit-guided intervention
      │
      ▼
AuditKIT evaluate (guard judges, red team, comparison, significance)
```

- The drift recipe and repair baselines are **reported** in SafeTune's EMNLP 2026 demo
  (`docs/emnlp-2026-demo/index.md` in the SafeTune clone): one epoch of benign LoRA SFT on
  `math / code / dolly / medical / legal` produced refusal drops in those runs. The figures are
  not independently reproduced or transferable guarantees for Tiny Aya.
- Upstream reports fast post-hoc recovery; real wall time/RAM and safety/utility depend on the
  selected method, references and hardware. ReSta takes loaded compatible models and requires saving.
- CircuitKIT adds the interpretability "why" and an intervention path, but Tiny Aya support is
  experimental — see the risk table below.

### Chain for the current shortlist (safety/alignment)

```
CuratorKIT (grounded cases, reviewed labels, grouped splits)
   └─► baseline + strong prompt variant
          └─► SafeTune steering / AlignTune SFT (optional intervention)
                 └─► AuditKIT task-specific scoring (frozen test)
CircuitKIT: optional causal extension only after Cohere parity passes
```

- Candidate 1 (claim-grounded selective abstention) relies on CuratorKIT + AuditKIT, with an
  optional SafeTune or AlignTune arm. Candidate 2 (code-mixed safety) has the same shape with
  harmful-content and benign-utility labels.
- Neither a tool metric nor a training reward is an independent outcome oracle. Score source
  support and harmful assistance with task-specific labels, keep unknown/error counts, and group
  correlated rows by source/task.
- `AgentTune` and the earlier agentic-injection plan are archived; they are not the current build.

---

## 6. Cross-tool artifact contracts

| Producer | Artifact | Consumer | Contract |
|---|---|---|---|
| CuratorKIT | output dir with `README.md` configs, `sft_messages.jsonl`, `dpo.jsonl`, … + `lexsi_provenance.json` | AlignTune | `dataset_name="output/", config_name="sft_messages"`; provenance is embedded if the dir is passed |
| CuratorKIT | separate frozen eval folder | AuditKIT | `ak.load_dataset(folder, config=..., split="test")` preserves metadata/provenance |
| CuratorKIT | training-only folder | SafeTune | raw dataset or source string with `dataset_config`/`dataset_split`; distinct safety-training data; method-specific format |
| AlignTune / SafeTune | HF checkpoint or PEFT adapter dir | AuditKIT | `"hf:<dir>"` model spec |
| AlignTune / SafeTune | full/merged checkpoint dir | CircuitKIT | `load_model` / `Pipeline(model_name="<dir>", task=...)`; supported architecture and parity required |
| CircuitKIT | `*_scores.json` with `safety_units` + `layer_suggestions` | SafeTune | `load_circuit_info_from_file()` → `CircuitInfo`/SafetyLoRA; safety relevance must be demonstrated |
| CircuitKIT | pruned/edited HF checkpoint + `lexsi_provenance.json` | AuditKIT / SafeTune | standard HF loading |
| AgentTune | `trajectories.jsonl` + real state/artifact sidecars | AuditKIT | stable case/trial join; `AgentEvalSpec(episodes=...)` for recorded or `agent=...` for deployed evaluation |
| AgentTune | `EventLog` object | AuditKIT | `episode_from_eventlog(log)` |
| Any tool | `lexsi_provenance.json` | supported reader | nested lineage metadata only; writers use `git_sha: None`; caller must verify revisions/hashes |

See `09-cross-tool-integration.md` for source-checked templates and footguns, not an executed
whole-stack workflow. Agent reports are `AgentEvalResult`; ordinary text reports are `RunResult`.

---

## 7. Cohere Labs model fit

| Model | Size | Type | Gated | Best use here | Watch out |
|---|---|---|---|---|---|
| `tiny-aya-base` | 3.35B (`cohere2`, 36 layers) | Base LM, no chat template | Yes | SafeTune base reference; AlignTune distillation student; CircuitKIT discovery | No chat template → AlignTune falls back to a template; SafeTune warns about Tiny Aya's ~366-token system preamble |
| `tiny-aya-global` | 3.35B | Instruct/chat | Yes | Main experimental subject for the shortlist | Chat template does not render `tools`; that affects agent work only |
| `tiny-aya-fire/water/earth` | 3.35B | Regional instruct variants | Yes | Multilingual/regional angle | Same as global |
| `aya-expanse-8b` | 8B (`cohere1`) | Instruct/chat | Yes | Stronger main model if you have ≥24 GB | Native on vLLM/SGLang; CircuitKIT supports 8B (validated), 32B partly |
| `aya-expanse-32b` | 32B | Instruct/chat | Yes | Only if you get A100 80GB time | CircuitKIT discovery/eval/weight-steering **do not run yet** (TransformerLens multi-GPU bug; 64 GiB weights) |

Practical picks:

- **Behavioral work (SafeTune/AuditKIT)**: `tiny-aya-global` for iteration, `aya-expanse-8b` for a
   same-case comparison. They differ in architecture, tokenizer and training, so this is not a
   clean size-only control.
- **Drift references**: `tiny-aya-base` + `Lexsi/tiny-aya-global-hh-rlhf-aligned` (or
  `CohereLabs/tiny-aya-global` itself).
- **CircuitKIT**: start on `tiny-aya-base` with `greater_than` (recommended smoke task because it
  builds a single-token pool against the real tokenizer). Treat IOI as risky on the 262k vocab.
- **AgentTune**: not part of the current shortlist; its agent/tool-training notes are archived in `06`.

---

## 8. Compute plan (given: $50 Modal + 2 Kaggle accounts)

Planning rates (verify on Modal before running), effective $/hr: T4 $0.59 · L4 $0.80 · A10 24GB $1.10 ·
A100 40GB $2.10 · A100 80GB $2.50 · H100 $3.95. $50 buys roughly **62 h L4**, **45 h A10**, or
**23 h A100 40GB**.

The earlier `08` allocation (~$33 core, $5 optional, $2 overhead, $10 reserve) is archived. Treat
$50 as a hard ceiling: Kaggle first, at most one measured Modal job, and no spend before a runtime
estimate. These are caps, not measured runtimes. Paid generation/judging APIs are separate bills,
not covered by a "$0 CPU" label or Modal/HF credits.

Use the two available Kaggle accounts only within its account/quota rules; GPU type, queue,
weekly allowance and sessions are not guaranteed or automatically pooled. Reserve Modal for
time-boxed jobs with persistent artifacts; avoid making correctness depend on a free queue.

Compute rules that save real money:

1. Put the HF cache on a **Modal Volume**, not in the image; re-downloading Tiny Aya (3.35B,
   ~6.7 GB fp16) on every cold start burns minutes.
2. Use `modal run --detach` for training, then log artifacts to a volume and `huggingface_hub`
   upload; never leave a container idle.
3. Do all evaluation twice smaller first: `max_prompts=50–100`, then scale the winning config.
4. Smoke-test QLoRA at short context/batch size before budgeting an L4/T4 run; fitting is a
   hypothesis until peak memory is measured. RL references, judges and KV caches can change it.
5. Use **Modal batched inference** for SafeTune/AuditKIT generation; per-request CPU overhead is
   what makes serverless expensive, not the GPU.

---

## 9. Risk table (decide with eyes open)

| Risk | Where | Severity | Mitigation |
|---|---|---|---|
| Tiny Aya is **gated** | all tools | Low | Accept license on HF, `hf auth login`, pass `HF_TOKEN` on Modal/Kaggle |
| CircuitKIT Tiny Aya support is **experimental**; parity gate must pass | CircuitKIT | High if mechanistic claims depend on it | Run the real-weight parity gate; GPT-2 smoke proves mechanics only. Drop stretch work if it fails; do not substitute another model's heatmap for subject evidence. |
| SafeTune's default judge (`allenai/wildguard`) is gated 7.25B, ~14.5 GB two-byte weights before overhead | SafeTune | Medium | AdvBench still uses the selected main-suite judge. Use a separate explicit StringMatch/2B-guard diagnostic path and calibrate it; neither is verified agent ASR. |
| `transformers>=5.15` everywhere; many tutorials are 4.x | all | Medium | Pin the repo's own versions; do not mix a 4.x env with these tools |
| AgentTune CLI has no `--tools`; tool-calling training is Python-API only | AgentTune | Medium | Write a script, not CLI; start from the README Tiny Aya example |
| AgentTune known silent zeros (ROUGE/BLEU/DistilledJudge) | AgentTune | Medium | Prefer its trajectory metrics (TAC/ARR/LCFR) and AuditKIT ToolCall metrics over ROUGE |
| AuditKIT pairwise fallback silently uses token overlap | AuditKIT | Medium | Pass real `candidates`/`pairwise_results` context or do not use those metrics |
| AuditKIT bootstrap p-value is constructed from uncentered observed differences | AuditKIT | High for inferential claims | Independent paired binary test and task-clustered intervals; see finding 5/A.10 in doc 15. |
| AlignTune packaging comments disagree on dependency floors | AlignTune | Low | Resolve/freeze the exact pin and environment; comments are not the resolver contract. |
| Unsloth AGPL components / other bundled licenses | AlignTune | Conditional | Explicit TRL reduces backend surprises, not all distribution obligations; review licenses. |
| SafeTune/AgentTune ML stacks are heavy | SafeTune, AgentTune | Medium | Separate locks; selected-judge safety scores are not native capability graders. |
| CuratorKIT `[all]` + PDF extra is heavy | CuratorKIT | Low | Install `[all]` without `[pdf]` unless you actually parse PDFs |
| Raw agent traces lack final-state evidence | AgentTune → AuditKIT | High | Stable case/trial IDs, fresh environment and genuine state/artifact sidecars; unknown is not success. |
| Exposed shell/Python tools retain host authority | AgentTune | Critical if exposed to untrusted actions | Use narrow mock tools; explicit capability grants and a real sandbox for hostile code. |

---

## 10. The five "interesting aspects" that are easy to miss

1. **Provenance is a first-class artifact.** CuratorKIT writes `lexsi_provenance.json`,
   `manifest.json`, `rejected.jsonl`, `dataset_card.md`, `checksums.txt`; SafeTune and AuditKIT
   read provenance too. A demo that shows the lineage graph from *source document → sample →
   fine-tune → repair → eval* is more distinctive than another metric bar chart.
2. **SafeTune's explicit upstream faithfulness badges.** Each method is labeled Faithful / Simplified /
   Variant with `file:line` evidence. You can build a project *on top of* that honesty: reproduce
   a selected method and report effectiveness yourself; a badge is not a safety certificate.
3. **CircuitKIT's `safety_units`/`layer_suggestions` are an interop format.** The circuit scores
   file is not just a visualization; SafeTune consumes it to target SafetyLoRA. That's a real
   "interpretability changes training" hypothesis once the hand-off, reload and causal outcome
   checks actually run.
4. **AuditKIT's zero-dependency spine is an architecture choice, not just a slogan.** The core
   `evaluate()` runs on stdlib; heavy things load lazily. That makes it the perfect glue layer for
   a Space or a Lambda-style demo.
5. **AgentTune's EventLog is the only shared trajectory schema in the stack** (archived for this
   shortlist). It would matter only if agent/tool training returns; the current candidates do not
   need it.

---

## 11. Decision checklist before you choose a project

Answer these in writing before building:

- [ ] One submission framing, confirmed with the organizers (the repo previously recorded Track 3); at least two tools used substantively.

> **Status 2026-11-15:** one item is answerable and none of the build steps below were done. The
> subject models are `tiny-aya-global` + `aya-expanse-8b` (permitted, Rules §3) and a frozen
> four-benchmark baseline exists in `part 1/baselines/benchmark.md`. Everything else on this list is
> still open, including the tool chain, the before/after metrics, the HF uploads and the demo. The
> track is **not** Track 3 — no agentic work was run. See `AGENTS.md` FINAL STATE.
- [ ] Which **permitted Cohere model** is the subject, and which is the reference?
- [ ] What is the **one-sentence thesis** ("X becomes Y by Z")?
- [ ] What are the **two or more before/after metrics**, and which tool computes each?
- [ ] What is the **baseline** (untuned model? unfiltered data? un-repaired checkpoint?)
- [ ] What is the **demo story in 3 minutes** (input → pipeline → metric movement → artifact)?
- [ ] What is the **biggest single risk**, and what is the fallback that still produces a result?
- [ ] What gets uploaded to **Hugging Face** and under what names?
- [ ] Does the demo run on **someone else's machine**, not just ours?
- [ ] Do the evidence, model/adapter revision, oracle version, frozen splits and unknown/error counts support the precise claim?
