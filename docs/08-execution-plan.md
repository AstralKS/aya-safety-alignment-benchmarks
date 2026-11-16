# 08 — Execution plan: 14 days, $50 Modal, 2 Kaggle accounts

> **Archived (Nov 8, 2026):** this is the earlier 14-day AgentGuard plan. The current decision
> brief is [`../DECISION-BRIEF.md`](../DECISION-BRIEF.md) and the project shortlist is
> [`07-project-idea.md`](07-project-idea.md). Do not follow the schedule or track framing below.

> Plan for **Nov 1 → Nov 14, 2026** (build + submit), with a **Nov 16–21** bonus window if
> shortlisted. **One registered Track 3 AgentGuard submission**: CuratorKIT → AgentTune →
> AuditKIT. SafeTune/probes are optional; CircuitKIT is stretch. Dates/credits come from the
> official-site snapshot in doc 13; confirm the submission form and credit terms. This is a
> proposed build plan, **not work already executed**. **(Live-site deadline is Nov 15, 11:59 PM AoE,
> not Nov 14 EOD, and the Track 3 framing is void — see `13-research-notes.md` §4 and `AGENTS.md`
> FINAL STATE.)**

---

## 1. Operating principles

1. **Core plumbing first.** Test the three core tools, real Tiny Aya tool calls and evidence
   joins early; do not spend Day 1 installing all six heavy stacks.
2. **Behavioral result is the insurance policy.** The verified-state ASR table must exist by
    Day 10 with baseline + prompt defense + at least one genuinely trained policy, even if
    GRPO is unfinished. Preserve separate security/utility and unknown/error accounting.
3. **One model family, one environment per tool.** Tiny Aya + Python 3.12; separate venvs;
   checkpoints are the interface.
4. **Opportunistic free compute.** Use permitted Kaggle sessions alongside Modal; queues,
   device/session quotas and pooling across accounts are not guaranteed.
5. **Pre-register the metric plan and decision rules** (a lightweight version: commit a
    copy of `starter/metric-plan.md` in the project on Day 2 and do not change it after results,
    except with a
   logged deviation note). This is cheap and visibly separates the submission from notebook
   entries.
6. **Publish as you go.** HF repos live from Day 2; the write-up skeleton exists from Day 3.

---

## 2. Compute plan

### Modal (`$50` available credit; per-second rates checked Nov 1, 2026)

| GPU | Effective $/hr | Best for |
|---|---|---|
| T4 16GB | $0.59 | smoke tests/tiny evals; not a guaranteed QLoRA fit |
| L4 24GB | $0.80 | 3B QLoRA, batched generation, the workhorse |
| A10 24GB | $1.10 | same as L4 with a bit more compute |
| A100 40GB | $2.10 | CircuitKIT discovery, judge-heavy evals, 8B experiments |
| A100 80GB | $2.50 | larger-memory experiments; 32B training/discovery is not guaranteed to fit |
| H100 80GB | $3.95 | avoid unless blocked |

Official [pricing](https://modal.com/pricing) lists L4 **$0.000222/sec**, A10
**$0.000306/sec** and A100 40GB **$0.000583/sec**; hourly rates above are rounded planning
values. CPU/RAM/storage, region and non-preemptible premiums are additional. Check the dashboard
before every job; credit eligibility/expiry and external API bills are separate.

**Shared budget across docs 00, this plan and the decision brief:**

| Core job envelope | GPU | Cap in GPU hours | Approx. GPU cost |
|---|---|---|---|
| Environment/access/tool-call + evidence-join smoke | L4 | 3 | $2.40 |
| Training-only local corpus generation/gates | L4 | 3 | $2.40 |
| Baseline/prompt-defense validation rollouts | L4 | 6 | $4.80 |
| Safe collection + SFT/distill route | L4 | 5 | $4.00 |
| GRPO route (conditional; otherwise core reruns) | A10 | 6 | $6.60 |
| Frozen final matrix + published-artifact re-test | L4 | 16 | $12.80 |
| **Core GPU envelope** | | **39 h cap, not a runtime forecast** | **~$33** |

| Remaining allocation | Approx. cap |
|---|---:|
| One optional analysis: e.g. 2 h A100 40GB + 1 h L4, not every stretch goal | $5 |
| CPU/RAM/storage/startup allowance (must be measured; may need reserve) | $2 |
| Reserve for reruns/overhead; no automatic extra scope | $10 |
| **Total Modal envelope** | **$50** |

Measure peak memory, seconds/episode and actual spend on a small validation run; estimate
`arms × tasks × frozen attacks × trials`, including benign runs, retries and artifact re-tests.
If the envelope does not fit, reduce/lock the suite **before test results**, drop optional
arms, or report a smaller study with wider uncertainty. Hours above do not promise completion.
Paid generation/judge APIs need a separate confirmed budget; CPU use is not automatically $0.

Modal hygiene:

- Put the HF cache on a **Volume**: `modal.Volume.from_name("hf-cache", create_if_missing=True)`
  mounted at `/root/.cache/huggingface`; Tiny Aya is ~6.7 GB and re-downloading it per cold
  start wastes both time and money.
- Use `@app.function(gpu="L4", timeout=3600*4, volumes={...}, secrets=[modal.Secret.from_name("hf-token")])`.
- Use `modal run --detach` for long jobs; write outputs to the volume and upload to HF from a
  separate short job.
- Use the installed Hub client's supported cache/transfer settings; do not assume an older
  `hf_transfer` environment flag is required or effective.
- Keep a `modal shell`-style debug entrypoint for fast iteration; do not rebuild the image for
    code-only changes when the configured image/source mounts permit it; verify the actual job.

### Kaggle (2 accounts)

- **Possible runtime**: historically T4×2/P100 sessions and roughly 30 GPU h/week per eligible
  account. Inspect the current account's quota/device/session limits; these are not entitlements.
- **Use for**: data generation with small local models, AuditKIT evaluations, Tiny Aya LoRA
  smoke tests, SafeTune steering demos, reading/plotting results.
- **Persistence**: do not rely on a transient notebook environment; internet must be
  enabled per session, HF token via Kaggle Secrets, session dies on idle. Save adapters to
  `/kaggle/working` and push to HF immediately; do not rely on the working dir surviving.
- **Two available accounts do not create one pooled quota.** Use only legitimate team accounts
  within Kaggle's terms, without bypassing quotas. Sync immutable artifacts, not shared live state.
- T4 lacks native bf16 support; choose fp16/quantization deliberately. A 16 GB QLoRA fit at
  short context/batch size is a smoke-test hypothesis; RL references/judges/caches add memory.

### Hugging Face artifacts

- Create the org/user repos on Day 1–2: dataset, model cards (one per arm), logs dataset, Space.
- Use `huggingface_hub` `upload_folder`/`upload_file`; keep a `lexsi_provenance.json` in every
    artifact, plus explicit model/data revisions and byte digests. Nested provenance alone is
    not validated lineage (`git_sha` is unset in the reviewed writers).
- The site reports `$20` Discord and conditional `$200` finalist **HF credits**. Confirm
  redemption/eligibility/expiry and covered HF services; they are not Modal credit or an automatic
  ZeroGPU entitlement. Prepare a CPU replay Space and a recorded backup if live GPU access fails.

---

## 3. Day-by-day plan (Nov 1–14)

Assumes a team of 2–3. Solo builders: cut the stretch column, keep the critical path.

### Day 1 (Nov 1) — Kickoff + environment

| | Critical path | Stretch |
|---|---|---|
| AM | Register/confirm; accept Tiny Aya licenses; `hf auth login`; create HF repos | Read kickoff notes / Discord announcements |
| PM | Separate `data/eval` and `agent` environments; minimal extras, exact pins/locks; AuditKIT oracle controls + AgentTune scripted spine | Optional tools get separate environments only when needed |
| Output | `env-report.md` with exact versions, smoke outcomes and untested paths; starter labelled scripted | — |

### Day 2 (Nov 2) — World + corpus skeleton

| | Critical path | Stretch |
|---|---|---|
| AM | Narrow mock tools, trusted authorization and independent safety/utility oracles; fresh reset tests, missing-evidence controls, stable IDs | No host shell/code tools; process isolation alone is not a sandbox |
| PM | Allocate task/source/payload groups before generation; small train/validation/test corpus; nonempty compatible exports; freeze copied metric plan | Publish a labelled draft dataset/card |
| Output | Group split/lineage manifest, outcome controls and metric plan; memory category marked deferred unless implemented | First HF dataset commit |

### Day 3 (Nov 3) — First live baseline

| | Critical path | Stretch |
|---|---|---|
| AM | Real Tiny Aya tool-call/parse/execution smoke; explicit TRL and no-vLLM first; bounded training smoke after collection works | Serving acceleration only if necessary |
| PM | Import traces, join independently recorded state/action sidecars by case/trial IDs; separate security/utility reports | Optional scripted SafeTune API smoke, labelled as such |
| Output | AgentTune → genuine evidence → `AgentEvalResult` skeleton; peak memory/time measured | — |

### Day 4 (Nov 4) — Baseline attack measurement

| | Critical path | Stretch |
|---|---|---|
| AM | Baseline on **validation**, with task-level V-ASR, separate utility and unknown/error counts | Small guard diagnostic with calibrated polarity |
| PM | Fix wiring; size and freeze final task/attack/trial matrix from measured costs, before repairs; preserve validation evidence | Optional SafeTune selected-judge diagnostic, not native ASR/XSTest scoring |
| Output | **Validation before numbers exist**; final suite/limits/IDs locked | — |

### Day 5 (Nov 5) — Registration deadline + policy repair arm A/B

| | Critical path | Stretch |
|---|---|---|
| AM | Confirm registration is complete (deadline!) | — |
| PM | Arm A prompt defense; collect full safe logs or valid messages for B, with supplied SFT/distill factory | Start GRPO config for C; task-grounded safety rewards |
| Output | Arm A evaluated; Arm B data ready | — |

### Day 6 (Nov 6) — Arm B training

| | Critical path | Stretch |
|---|---|---|
| AM | LoRA SFT/distill on safe trajectories (Kaggle or Modal L4) | — |
| PM | Validate B on the matched validation suite; start C only if time/memory/budget gates pass | Real-weight Tiny Aya parity only if stretch is funded |
| Output | B trained + validation/evidence report; scripted episodes are not a trained policy | Optional parity status |

### Day 7 (Nov 7) — Arm C GRPO

| | Critical path | Stretch |
|---|---|---|
| All day | Conditional GRPO with on-policy mock-world rollouts; monitor utility, safety, refusal, errors and loops independently | — |
| Output | C checkpoint/evidence or recorded deferral; B remains the trained fallback | — |

### Day 8 (Nov 8) — Evaluation sweep

| | Critical path | Stretch |
|---|---|---|
| AM | Complete validation/tuning; choose cheapest eligible trained arm under frozen rules; lock every arm/revision | Optional D requires actual live-wrapper evaluation |
| PM | Start **frozen test** matrix: matched tasks/attacks/trials, fresh worlds, all predeclared arms | Pre-action probe features with group-disjoint splits |
| Output | Locked arms + complete evaluation progress/coverage; no tuning on test outcomes | — |

### Day 9 (Nov 9) — Core evaluation and uncertainty

| | Critical path | Stretch |
|---|---|---|
| AM | Finish frozen matrix; audit expected/completed/observed cases and all failures | One optional calibrated text/refusal diagnostic |
| PM | Explicit any-attack-per-task V-ASR; clustered intervals/paired effects; exact McNemar only on complete binary task pairs | Pre-action outcome probe, not attack-presence classification |
| Output | Security/utility table + unknown bounds; no AuditKIT bootstrap p-value significance claim | Optional monitor report |

### Day 10 (Nov 10) — Robustness, published-artifact checks; optional analysis

| | Critical path | Stretch |
|---|---|---|
| AM | Check frozen held-out source/payload-family slice and all category regressions; no test-driven attack replacement | Circuit discovery/causal controls only after parity and within $5 optional cap |
| PM | Save/reload selected adapter/full artifact; re-test the **exact deployment precision/wrapper**, including int4 if used | A labelled GPT-2 smoke establishes mechanics only |
| Output | **Core matched before/after table by Day 10**, artifact parity and residual failures | Optional causal result or honest deferral |

### Day 11 (Nov 11) — Artifact freeze + write-up

| | Critical path | Stretch |
|---|---|---|
| AM | Upload permitted adapters/logs, traces + sidecars, agent/text JSONs and hash manifest to HF; model/dataset cards | Publish Space skeleton |
| PM | Write the HF Article: problem, threat model, pipeline, metrics, results, limitations, reproduction | Record demo video |
| Output | Complete HF artifact set + article draft | — |

### Day 12 (Nov 12) — Demo Space + final numbers

| | Critical path | Stretch |
|---|---|---|
| AM | Build the demo Space (Gradio): select arm, choose attack, run episode, show state panel + metrics | ZeroGPU note |
| PM | Reproduce the frozen table from released evidence; record a real demo or labelled replay; do not replace bad trials for a cleaner result | Measured ROI table |
| Output | Public demo/replay + backup video + versioned final evidence | — |

### Day 13 (Nov 13) — Polish, review, submit dry-run

| | Critical path | Stretch |
|---|---|---|
| AM | Independent review: can a stranger reproduce the pipeline from the README? | Ask a friend/judge-style reviewer |
| PM | Submission dry-run: required links, HF artifacts public, write-up links, metrics table, license notes | — |
| Output | Submission draft | — |

### Day 14 (Nov 14) — Submit EOD

| | Critical path |
|---|---|
| AM | Final link check; make repos public; confirm Space is running (test incognito) |
| PM | Submit; post a short summary in the Discord channel; archive the submission hash/tag |
| Output | **Submitted** |

### Nov 16–21 (only if shortlisted)

- If the conditional HF credit is redeemed for an eligible service, consider an explicitly
  **new** Aya Expanse 8B replication/category/language experiment. Do not assume it pays Modal
  or retroactively change the submitted frozen table.
- Add the ROI comparison (below), a 2-minute video, and a Q&A appendix with likely judge
  questions.
- Rehearse the live demo three times; the second rehearsal should be on the exact Demo Day
  hardware/network settings.

---

## 4. Team roles (2–3 people)

| Role | Owns | Artifacts |
|---|---|---|
| **Data/red-team lead** | CuratorKIT corpus, attack taxonomy, benign controls, provenance, dataset card | HF dataset, metric plan, per-category attack report |
| **Training lead** | AgentTune rollouts + repair arms (SFT/GRPO), Modal jobs, adapter cards | HF adapters/models, training logs, trajectories |
| **Eval/interp lead** | Independent security/utility oracles, trace/sidecar joins, aggregation/uncertainty; optional calibrated diagnostics | AgentEvalResult + text JSONs, manifest/coverage, before/after table; optional probe/circuit report |
| Shared | README, HF Article, Space, demo rehearsal, submission checklist | Public repo + article + Space |

Solo: do the critical path only; pick **prompt defense + SFT arm + AuditKIT** as the minimum
publishable comparison, retaining CuratorKIT data and AgentTune rollouts with a real training
factory. Add GRPO only if time permits; do not turn a scripted demo into a model result.

---

## 5. Risk register

| Risk | Likelihood | Impact | Trigger | Mitigation |
|---|---|---|---|---|
| Tiny Aya gated access delayed | Low | High | cannot `from_pretrained` | Start plumbing on an open model; keep the final run on Tiny Aya; request access Day 0 |
| AgentTune tool-call parse rate low | Medium | High | <60% parseable calls in 20 rollouts | Simplify tool protocol; use `tools_fallback_prompt`; reduce to 2 tools; consider ToolCallAdapter-style prompt mode |
| GRPO unstable / reward hacking | Medium-High | Medium | reward up, utility down; loops up | Keep SFT arm as the headline; use reward composition + loop penalties; clamp rewards |
| Evaluation too slow/expensive | Medium | High | measured seconds/episode exceed envelope | Downsize validation early; lock final task/attack/trial matrix before repair results; report smaller N and uncertainty |
| CircuitKIT parity fails | Medium | Low (stretch only) | real-weight parity test fails its source tolerances | Drop stretch; other-model screenshots are not subject evidence |
| Over-refusal rises after repair | Medium | Medium | explicit benign refusal/utility violation | Tune only on validation; report final Pareto and failures without changing test thresholds |
| Kaggle session dies mid-training | Medium | Medium | timeout/OOM | Checkpoint every N steps to HF; use Modal for the must-finish jobs |
| Modal spend exceeded | Medium | High | projected committed spend > $40 before reserve | Drop optional work; explicit per-job/time/token caps and nightly shutdown; credits do not pay external APIs |
| Canary/tool demo accidentally destructive | Low | High | real network/credentials | Mock tools only; no real secrets; run the demo offline |
| License/attribution mistake | Conditional | High | artifact/organizational/bundled terms unchecked | Explicit TRL for repeatability, not legal immunity; review LSAL/model/data/redistribution terms and notices |
| Missing/unmatched evaluation evidence | Medium | High | absent state, duplicate IDs, zero cases or arm mismatch | Fail the comparison gate; keep unknown/errors and lower/upper V-ASR bounds; never fabricate state |
| False statistical confidence | High if library p-value reused | High | row-level bootstrap or AuditKIT significance flag | Cluster by task; independent paired binary test; small N is a limitation, not evidence of equivalence |

---

## 6. Submission checklist

**Rules compliance**

- [ ] Uses a permitted Cohere model as the subject (`tiny-aya-global`/`fire`/`water`/`earth`/`base`
      or `aya-expanse-8b/32b`).
- [ ] **One Track 3 submission**; two-track-repository minimum is excepted. AgentTune + AuditKIT
      recommended, CuratorKIT data core; at least one real trained agentic policy.
- [ ] All artifacts hosted on HF: model/adapters, logs, datasets.
- [ ] Public repo + public demo (Space) by Nov 14 EOD.
- [ ] Before/after metrics, more than one, with rationale.
- [ ] HF Article write-up.
- [ ] Original work; third-party libraries credited; licenses respected.
- [ ] Team size 1–3; registration complete by Nov 5.

**Artifacts**

- [ ] HF dataset(s) with dataset card, `manifest.json`, `rejected.jsonl`, provenance, checksums.
- [ ] HF model/adapter per arm, each with a model card that states exactly what it is (baseline,
      prompt-defense, SFT, GRPO, steered) and any residual risk.
- [ ] Stable-ID traces + genuine state/action/artifact sidecars; `AgentEvalResult` and separate
      text `RunResult` JSONs; revisions/hashes, coverage, unknown/errors and task-level statistics.
- [ ] Space that runs the attack/defense demo.
- [ ] README with a 60-second **CPU evidence-table/scripted smoke** path; model training has
      separate hardware/access/dependency instructions and realistic runtime limits.
- [ ] `metric-plan.md` + deviations log.
- [ ] Core figures: category before/after + task-clustered intervals and safety–utility trade-off;
      probe/circuit figures only if actually run with appropriate controls.

**Write-up outline (HF Article)**

1. TL;DR with the headline before/after table.
2. Threat model and why indirect injection is the right target.
3. The corpus and benign controls (CuratorKIT gates + provenance).
4. The agent and the repair arms (AgentTune/SafeTune).
5. The scoreboard (AuditKIT verified outcomes + diagnostics + fingerprints).
6. Results, per category, with CIs and a negative-result section.
7. Internal monitor / circuit localization (if run).
8. Limitations: small model, mock tools, sample/task count, seed count, nonadaptive frozen attacks,
   language coverage, missing evidence and judge calibration; no broad robustness certificate.
9. Reproduction: pinned versions, seeds, hardware, commands, artifact links.
10. Optional ROI: self-hosted Tiny Aya vs closed API at a realistic request volume.

---

## 7. ROI appendix (optional but preferred by the rules)

Use measured numbers, not vendor guesses:

- **Throughput**: measure the actual served artifact's output tokens/sec at realistic
  concurrency, prompt length and precision. MedCurate's ~307 output tok/s is its reported
  1.5B/T4/SGLang run, not a Tiny Aya 3B forecast. Include prompt processing and tool/judge latency.
- **Self-host cost**: GPU $/h ÷ (tokens/s × 3600) × 1e6 → $ per million output tokens; add
  CPU/RAM/storage/egress, prompt processing, judge/tool calls and idle-factor (e.g. 30%
  utilization → 3.3× the idealized GPU-only marginal cost). Report cost per successful safe task too.
- **Closed API cost**: current price per million input/output tokens for the comparable closed
  model × your average prompt/completion ratio.
- **Break-even volume**: monthly tokens where self-host becomes cheaper, including the repair
  pipeline's one-time cost (Modal hours + engineer hours).
- Report the range and the assumptions in a table; do not claim a single precise number.

---

## 8. Judge Q&A prep (write answers before Demo Day)

1. "Why should I believe the ASR number?" → Declared executable attacker objectives, genuine
   state/action/artifact evidence, task-level any-attack aggregation, coverage/unknown bounds,
   clustered uncertainty, released hashes and separate utility.
2. "Did you just make the model refuse everything?" → Over-refusal and benign utility are
   first-class metrics; Pareto frontier reported.
3. "Is this robust to attacks you didn't train on?" → Group-disjoint frozen source/payload-family
   slice; say what ran and do not imply protection against adaptive/unseen attack classes.
4. "Why a 3B model?" → Bounded cost/latency study; larger/multilingual transfer remains untested.
5. "What's new relative to prior red-team repair work?" → Tool-output-borne injection with
   verified outcomes in a small-model training recipe; optional pre-action monitor. The complete
   combination was not found in five reviewed public projects; AgentDojo already verifies outcomes.
6. "What failed?" → Have a real answer ready: parity result, a category where repair did not
   help, over-refusal cost.
7. "What would you do with 3 more months?" → Aya Expanse 8B/32B, real benchmark integrations
   (AgentDojo), multi-agent/memory poisoning, a learned automated attacker.
