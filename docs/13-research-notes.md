# 13 — Research notes, sources, and verification status

> How these docs were produced, what was verified in code, what is documented-only, and what
> remains uncertain. Use this file to defend or update any claim in the other docs.

---

## 1. Method

1. **Cloned all six hackathon repos** (shallow, pinned) plus the other public Lexsi repos that
   looked relevant, plus five public hackathon-adjacent community repos found via GitHub search.
2. **Read source, not only READMEs.** For each tool: package `__init__`, factory signatures,
   registries, metrics/gates/benchmark definitions, CLI entry points, tests, and known-issues
   docs. Source-checked behavior, upstream reports, execution results and proposed experiments
   are distinguished in the text and in §3 below; reading a test is not running it.
3. **Read the hackathon site** (`hackathon.lexsi.ai`) for rules, tracks, model list, dates,
   structure, prizes, FAQ.
4. **Searched GitHub and HF** for public projects and artifacts using the tool names; cloned and
   read the five most relevant community projects.
5. **Checked official Modal pricing** and separated GPU-only arithmetic from CPU/RAM/storage,
   credits and unconfirmed Kaggle account quotas.
6. **Cross-checked Cohere model facts** (sizes, gating, template behavior) against the tool docs,
   the HF org pages, and Cohere's blog.
7. **Reviewed docs 00–14, README, decision brief and starter documentation** against the
   [15-component audit](15-missing-components.md). Bounded CPU checks cover deterministic
   evaluation plumbing and failure controls (§3.1), not live-model behavior.

All work was done on **Nov 1, 2026** (with the user's original prompt on Oct 31, 2026 IST).
Versions and star counts move; re-check before quoting.

---

## 2. Exact repo pins (commit hashes)

| Repo | Commit | Date | Notes |
|---|---|---|---|
| `Lexsi-Labs/aligntune` | `57e9a95e79ace15014beede175c4c3d72405f3b3` | 2026-10-07 | docs theme commit; `__init__` fallback version 0.1.10 |
| `Lexsi-Labs/CuratorKIT` | `dd9e835301fcddc74a37501d9e207a5a060d209f` | 2026-10-12 | `curatorkit.__version__ == 0.1.21` |
| `Lexsi-Labs/AuditKIT` | `71316da345784ff220d5ed59337441a59ab31526` | 2026-10-07 | release 1.1.3 |
| `Lexsi-Labs/SafeTune` | `26f59e6d1bdcacafe5f2be3b8d982fb214369659` | 2026-10-21 | `pyproject` version 0.1.8 |
| `Lexsi-Labs/CircuitKIT` | `e71d5e17fe6e2abfde47ec9a7fbaf31649601b78` | 2026-10-08 | `__version__ == 0.2.0`; docs claim 1.0.0 |
| `Lexsi-Labs/AgentTune` | `d4db4c12d52342227c4113a70183d0f64aa2f345` | 2026-10-07 | `pyproject` version 1.1.0 |
| `Lexsi-Labs/TabTune` | `d44e716677213e3b772318e65545cdb7b4299198` | 2026-10-06 | README update |
| `Lexsi-Labs/DLBacktrace` | `aa069acadb79087211ba1b8a1c10cda5fd7a749b` | 2026-05-13 | MoE upgrade |
| `Lexsi-Labs/xai_evals` | `b4f11b70873a4dbc7cee909c652268bba5549ffe` | 2025-11-05 | README only |
| `Lexsi-Labs/Orion-MSP` | `c89aea6dc425bc701a9557fdd53e3dc3d766f983` | 2025-11-06 | preprocessing |
| `Lexsi-Labs/Orion-BiX` | `060d34606f56c591c47e0b6cbe971e5f5035f33e` | 2025-12-02 | paper link |
| `Lexsi-Labs/Lexsi-sdk` | `70732c9c7e006b85bda34890b622edfb2bbda3ab` | 2026-10-29 | evals fixes |
| `Lexsi-Labs/alignbeam` | `52f68b207077449477093564582c01cb97ef5012` | 2026-10-04 | placeholder README; no released code at this pin |
| `Lexsi-Labs/c-theta-delta-demo` | `5e98bbf9d4f1dd285b88164897fc0709929b0973` | 2026-03-09 | method demo; no independent result reproduction |
| `Esimuda/pidginguard` | `8fa065dab18b8cb98239526929491045da7cb167` | 2026-10-31 | offline harness demo |
| `Zhiyilang/lexsi_redteam` | `b5851082a981df8edcfa7bc837f93c144bca1681` | 2026-10-30 | pre-window build |
| `abdullahabtahi/refusal-parity` | `7f1a6f92b172d9497c13c8f3ab6abf603c883ba2` | 2026-10-29 | v0.2.0 |
| `umeraamir69/HifazatTab` | `e35baaf291fed12e4c83c2f2d3b7404b80bc2b57` | 2026-10-31 | T4 GRPO diet |
| `TirthNandha/medcurate-pubmedqa` | `9eba03cb44f668976d1054fe4f6f67d7a71daf3b` | 2026-10-25 | results README |

Shallow clones of the six hackathon repos and the other Lexsi repos live in
`research/`; community repos in `research/community/`. These **19 pinned working copies are
evidence**, not the project deliverable. Preserve them while checking claims; removal is a
separate user decision, not part of this documentation review.

---

## 3. Verification status by tool

**Terminology:** source-checked means read in implementation/tests at the pin; documented means
reported by upstream. Neither means installed, model-tested or mathematically audited in full.
Only §3.1 records local execution. Model-backed snippets are templates requiring the named
datasets, models, credentials and extras.

### AlignTune — source/documentation checked
- Factory functions and signatures: `src/aligntune/core/backend_factory.py` (lines ~760, ~1059,
  ~1599, ~1753, ~1917, ~2081).
- Algorithm list and backend matrix: `docs/algorithms/overview.md`,
  `docs/compatibility/backend-matrix.md`, `BackendFactory` registries.
- Reward catalog: `src/aligntune/rewards/registry.py`, `rewards/core.py`, `rewards/verifiable.py`
  (six verifiable reward classes).
- Benchmark bundles: `src/aligntune/eval/benchmarks/presets.py`.
  Task resolution is broken: `safety` **0/3**, `alignment_core` **1/5**, `reasoning` **1/3**
  (`lm_eval_integration.py:380–516`, `eval/evaluator.py:119–142`); shape-only bundle tests miss it.
- AlignmentAuditor probes: `src/aligntune/eval/alignment_auditor.py` (refusal, sycophancy,
  verbosity, reward-hacking fallback; per-language filtering).
- Cohere support: `src/aligntune/core/hf_compat.py` (sequence-classification registration),
  `core/model_loader.py` (fallback chat template), `core/peft/lora.py` (VLM language scoping),
  notebooks `test_algos_trl_aya.ipynb` / `test_algos_unsloth_aya.ipynb`.
- Known issues: `docs/ISSUES.md` (sliced splits, RAFT citation loss placeholder, RAFT not in
  BackendFactory).
- **Documented-only:** `advise`/`compose` behavior details (read in docs, not executed);
   performance numbers in `docs/backends/comparison.md`.
- **Source-checked caveats:** SFT LoRA requires `use_peft=True`; `percent=1` means 1%;
  `AlignmentAuditCallback` is optional/non-blocking; export regression is a separate path.
  Own MIT code does not erase the wheel's bundled Unsloth/mergekit license obligations.
- **Inconsistencies found:** pyproject comments vs dependency floors (`transformers==5.14.1`
  comment vs `>=5.15,<6` dependency); algorithm-count wording varies. Doc 01 separates 13
  common SFT/preference/RL entries from RAFT, distillation and ES; no support matrix was executed.

### CuratorKIT — source/documentation checked
- Interfaces (`curatorkit/interfaces.py`), config fields (`curator.py` `CuratorConfig`),
  pipeline order (`_build_steps`), exports (`exporters/`), gates (`gates/`), hygiene
  (`hygiene/`), diagnostic taxonomy (`diagnostic/failure_modes.py`), CLI commands (`cli.py`).
- Provenance files and HF-folder behavior: README + `manifest.py`, `curator.py` `_write_provenance`.
  Instruction/response rows do not produce DPO pairs; select an existing export file explicitly
  for Hub upload. `manifest.json`/cards/checksums are real, not a rights or contamination gate.
  Toxicity filtering can remove the very attack prompts under study; keep evaluation splits out
  of generation and training exports.
- **Documented-only:** PDF/MinerU behavior specifics, some generator prompt internals.
- **Not verified by execution:** end-to-end generation with a live judge.

### AuditKIT — source/documentation checked
- `evaluate()` signature and resolution rules (`src/auditkit/api.py`).
- Metric families and exports (`src/auditkit/__init__.py`, `metrics/README.md`).
- Scenarios (`scenarios/*.py`), red-team probes/detectors (`redteam/`), guard profiles
  (`docs/guard_models.md`), agent eval (`agent_eval/README.md`, `reliability.py`, `bfcl.py`).
- Known issues (`docs/known_issues.md`), pyproject extras and conflicts.
- **CPU-checked defects:** red-team detector-row aggregation/refusal polarity and empty outputs;
  uncentered bootstrap p-value; automatic callable-cache identity omits defaults (§3.1).
- **Agent boundary:** executable harness/oracles exist, but the caller supplies security tasks,
  authorization, environment resets and genuine evidence. Raw trajectories use `trajectory_id`,
  EventLogs use `id`; metadata does not supply a case/trial join. Recorded matching can ignore
  excess/unmatched episodes, error status bypasses the oracle, and budget rows have a separate
  summary bucket. Doc 09 adds complete-matrix matching and independent security labels.
- **API corrections:** no registered `"agent"` scenario or `"tool_call"` adapter; `"tools"` is
  registered. Ordinary persistence is `Experiment.add()` then `ExperimentDB.save()`;
  `AgentEvalResult.to_json()` is separate, not a drop-in `RunResult` comparison/persistence path.
- **Version mismatch:** README table says v1.1.2; `pyproject.toml` and `__init__.py` say 1.1.3.
- **Documented-only:** SGLang specifics and vLLM known issues (long docs, not executed).

### SafeTune — source/documentation checked
- Taxonomy and feature map (`docs/getting-started/taxonomy.md`,
  `docs/reference/feature-map.md`), scope/faithfulness rules (`docs/community/scope.md`).
- Benchmark registry (`docs/user-guide/evaluate/benchmarks/index.md`), judges, attacks,
  monitoring, Pareto, TamperBench docs.
- `evaluate()` signature (`src/safetune/instrumentation/evaluate/suite/evaluate.py:206`).
- Registry aliases (`src/safetune/runner/_registry.py`), CLI/YAML behavior (`docs/user-guide/usage.md`).
- Cohere notes: README, `docs/user-guide/recover/layer/resta.md`.
- EMNLP demo numbers and drift/repair tables (`docs/emnlp-2026-demo/index.md`).
- **Documented-only / not reproduced:** the drift numbers behind the Faithful badges (the project
   itself says CI does not yet reproduce them); per-method math (spot-checked, not fully audited).
- **Source-checked caveats:** main `evaluate()` uses one selected judge; WildGuard's measured
  field is refusal, not harmfulness. AdvBench selection does not remove the selected judge.
  The gated 7.25B default judge needs separate memory/access. Live-steering safety uses the
  wrapper while `_eval_utility()` loads the base `model_id` (`runner/steer/_base.py:92–125,168–195`);
  actual-wrapper utility needs separate evaluation. MART is adaptive training, not an independent
  held-out tool-injection benchmark. `apply()` does not save; use `save_checkpoint()` explicitly.
- **Historical heading, not version defect:** feature-map `Audit status (v0.1.0)` describes
  that release's audit. Package version is 0.1.8; the 123 upstream mechanism badges are not our
  empirical validation. `trl>=0.12,<1.15` admits AgentTune's `1.7.1`, so do not invent a TRL conflict.

### CircuitKIT — source/documentation checked
- Algorithm tiers and capability matrix (`docs/algorithms/overview.md`,
  `capability-matrix.md`, `docs/trust/scope.md`).
- Pipeline API (`docs/user-guide/pipeline-overview.md`), interventions
  (`docs/user-guide/applications.md`), 6-pillar framework (`docs/evaluation/framework.md`).
- Tiny Aya port (`docs/advanced/tiny-aya.md`, `backends/_tl_compat`), interop fields
  (`artifacts/scores.py`).
- **Version mismatch:** `__version__` 0.2.0 vs FEATURES.md "1.0.0".
- **Source/docs caveats:** six stable/seven research algorithms; no safety metric or SAE;
  runtime steering hooks are not a portable model-weight export. `Pipeline(model_name=...)`
  and custom CSV/template contracts differ from some upstream examples. Tiny Aya parity and
  transport/causal utility checks remain unexecuted.
- **Documented-only:** experimental models/TL bug details; not executed on CUDA.

### AgentTune — source/documentation checked
- Spine and EventLog (`src/agenttune/agentic/README.md`, `events.py`), strategies, memory,
  harness, tools, rollouts.
- `create_agentic_trainer` signature and registry (`core/backend_factory.py`), reward registry
  (26 names), DECIDE stage types, trajectory metrics (`docs/advanced/trajectory-metrics.md`),
  reward-hacking defenses (`docs/advanced/reward-hacking-defenses.md`), tool isolation
  (`docs/advanced/tool-isolation.md`), known issues (`docs/community/known-issues.md`).
- Cohere fallback and Tiny Aya example: README.
- **Source-checked caveats:** five configurable post-training algorithms, not five equivalent
  on-policy loops; DPO/BCO accept offline data. `backend="auto"` can select Unsloth. `Project`
  inference/collection logs are light-tier; valid full-tier collection/training needs its own
  contract. GraphMemory aging counts logical writes; external-memory deletion is not weight
  unlearning. A child process/tool timeout does not sandbox host authority.
- **CPU-checked gate defect:** zero task cases plus absent trajectory evidence can approve
  deployment without either model being called; nonempty no-regression/regression controls
  behaved as expected. `Project.heal()` detection is not automatic retraining/redeployment.
- **Upstream reports, not reproduced:** RAG synthesis model-backed end-to-end validation,
  notebook counts and performance. Gold-chunk-disjoint splitting is not general benchmark
  decontamination; service/RAG/new-tool paths were not executed.
- **Known issues are unusually specific** and should be assumed live unless the commit is newer
  than this review.

### Other Lexsi repos — README/API-level review only
- TabTune, DLBacktrace, xai_evals, Orion-MSP/BiX, alignbeam, Lexsi-sdk, c-theta-delta-demo: read
   READMEs and (where present) key source files; no execution.

### 3.1 Local execution and audit record (Nov 1, 2026)

CPU checks used **Python 3.14.0 on Windows**, AuditKIT's pinned stdlib core and directly loaded
AgentTune source. This is **not** a resolved Python 3.12 training environment. The deployment
gate check bypasses only AgentTune's top-level initializer with a source-path package shell;
the real `decide.state`/gate modules execute unchanged. No model download/network call is part
of these CPU checks.

| Check | Observed result | Limit / reproduction |
|---|---|---|
| Source/test inventory | **2,039 non-test / 687 test Python files** across the six tools; per-tool counts in doc 15 match a fresh count | Includes AlignTune vendor tree; not coverage or passing tests; all 19 repo pins re-read |
| AuditKIT red team | Constant refusal **52/78** detector rows pass; empty strings **78/78**; missing outputs **0 rows, 100% pass** | 26 fixed prompts, no model ASR; audit A.9 |
| AuditKIT bootstrap | 26 uniform paired gains/losses each give **`p=1.0`, not significant**; exact two-sided McNemar **`2.9802322387695312e-08`** | Synthetic scores expose the null-distribution defect; A.10 |
| AgentTune deployment gate | Empty evidence gives **approved, 0 cases, 0 calls**, task scores 0, no trajectory scores/check; equal nonempty scores approve and a full regression blocks | Synthetic verdict functions, not live deployment; A.11 |
| AuditKIT callable cache | Different default responses share a fingerprint; second model is not called and replays `1.0`; explicit distinct identity yields `0.0` | Synthetic callable/default-argument collision; A.12 |
| Cookbook ID/evidence join | Actual `Trajectory`/`trajectory_writer` and serialized actual EventLogs each score **all four frozen case/trial cells**, despite reordered files | Synthetic sidecars, not authenticated environment observations; doc 09 §5 |
| Cookbook rejection/oracle controls | **34** missing/duplicate/unmatched/wrong-case/trial/incomplete/type-invalid joins reject; **8** policy truth-table controls pass | Null/missing state stays unknown; harm can be decided with otherwise incomplete observations |
| Independent security vs execution | Harm before later `error` remains independent **security failure**; runner keeps `target_error`; harmful budget row is `failure` but summary buckets it as over-budget | Original statuses/errors preserved, not rewritten as successes; doc 09 and audit B.2 |
| Next-action APIs | Invalid scenario/adapter reject; explicit `Sample` + prompt-mode `ToolCallAdapter` gives valid-call F1/validity **1.0**, malformed-call **0.0** | Only HF model argument replaced by distinct named callable stubs; not a full agent/model test |
| Scripted starter | 12 rows; eight attack seeds give scripted **8/8 unsafe → 0/8 safe** | Preserved code/seeds; flags are illustrative, not the metric plan or Tiny Aya robustness |
| Paper metadata | All **14** linked arXiv abstract pages in doc 14 fetched for titles/authors/reported claims | No paper experiment reproduction; not one dedicated paper per toolkit |

Local CPU fixture scripts/results are retained outside the workspace in
`C:\Users\AstralK\AppData\Local\Temp\opencode\lexsi-doc-review-20261101`.
Appendix A.9–A.12 supplies standalone reproduction commands. The review also checks local
documentation links, fenced Python syntax, finding fields and protected-file integrity.

**Not executed:** GPU training; live Tiny Aya/agent robustness; calibration with real human
labels; combined six-tool installation; CUDA/TL parity; model-backed cookbook/Modal/Kaggle
jobs; full upstream test suites; deployment or HF artifact publication. Proposed AUROC,
causal steering, runtime/cost and novelty outcomes remain hypotheses until measured.

---

## 4. Hackathon facts and sources

| Fact | Source |
|---|---|
| Dates, tracks, rules, model list, prizes, submission requirements | [Official homepage](https://hackathon.lexsi.ai), its named sections (fetched Nov 1, 2026) |
| Teams 1–3, registration by Nov 5, submit Nov 14 EOD, shortlist Nov 16, Demo Day Nov 22 | site Structure/Schedule/Rules |
| Permitted models: Tiny Aya base/global/fire/water/earth, Aya Expanse 8B/32B | site Rules §3 |
| At least 2 track repos (3 preferred); Track 3 excepted | site Rules §2 |
| Host artifacts on HF (models, adapters, logs, datasets) | site Rules §4 |
| Before/after metrics, >1, with write-up rationale; HF Article; optional ROI | site Rules §5 |
| $20 HF credit (Discord, first 800), $200 for Demo Day finalists | site Structure/Prizes |
| Prize amounts and 6 months HF Pro | site Prizes |
| Kickoff Bengaluru, Polaris School of Technology, Oct 31 | site Schedule |
| Track 2 = AI Red Teaming, Safety and Alignment; Track 3 = Agentic Fine-tuning | site Tracks |
| Only the final submission before the deadline is considered | site Tracks / Rules §5 / FAQ |
| Problem statements are now listed; Track 3 includes business workflows/approvals, self-healing, RAG and reward-hacking defenses | site Tracks; AgentGuard fits a trained-policy framing, not an audit-only substitute |

The homepage's **Tracks and Rules sections** are the authority used here; the separate
`https://hackathon.lexsi.ai/rules` URL returned **404** on re-check. Do not substitute a stale
search snippet. The user is registered for **Track 3**; use one final framing and confirm the
submission form/any organizer updates. This records a dated rules snapshot, not a guarantee
that the organizers will never change it.

> **Re-checked 2026-11-15 (live site).** The snapshot above is stale on two points. The actual
> **submission deadline is Nov 15, 2026, 11:59 PM AoE** (Nov 16, 11:59 AM UTC) — not "Nov 14 EOD";
> Nov 22, 11:59 PM AoE is the *finalist* video/slides deadline. Registration closed **Nov 8**
> (extended by 12h). Rule §2 reads "at least 2 repos from the track, used together (except
> Track 3)", with 3+ preferred, and track membership is fixed on the site: **T1** CuratorKIT +
> AlignTune + AuditKIT, **T2** CuratorKIT + SafeTune + CircuitKIT + AuditKIT, **T3** AgentTune +
> AuditKIT. Track 2a problem statements #2 (adversarial/safety evaluation sets — "the deliverable
> can be the dataset and a report") and #13 (hallucination detection and abstention) are the
> closest fits to the Aya abstention work. The live homepage Rules/Tracks sections remain the
> authority; `/rules` still 404s.

---

## 5. Community projects and HF artifacts (sources)

| Item | Source |
|---|---|
| PidginGuard repo | `github.com/Esimuda/pidginguard` (cloned, README + docs read) |
| Refusal Parity repo + paper | `github.com/abdullahabtahi/refusal-parity` (cloned, README read) |
| lexsi_redteam repo + design | `github.com/Zhiyilang/lexsi_redteam` (cloned, README + DESIGN.md read) |
| HifazatTab repo | `github.com/umeraamir69/HifazatTab` (cloned, README read) |
| MedCurate repo | `github.com/TirthNandha/medcurate-pubmedqa` (cloned, README + results read) |
| HF artifacts | HF API searches for `aligntune`, `curatorkit`, `safetune`, `circuitkit`, `agenttune`, `lexsi` |
| Lexsi HF org | `huggingface.co/Lexsi` (models/collections: Tiny Aya HH-RLHF aligned, SafeTune recovery, C-ΔΘ-related repair checkpoints) |
| Cohere Labs HF org | `huggingface.co/CohereLabs` (55 models: Tiny Aya variants, Aya Expanse, North Micro Vision, North Mini Code, etc.) |
| AgentDojo | [Debenedetti et al., arXiv:2406.13352](https://arxiv.org/abs/2406.13352): 97 tasks / 629 security cases, extensible environments and executable checks; not merely trace judging |
| AgentHarm | [Andriushchenko et al., arXiv:2410.09024](https://arxiv.org/abs/2410.09024): 110 malicious tasks, 440 augmented tasks / 11 categories; direct misuse differs from indirect injection |
| OWASP Agentic Top 10 2026 | [Official resource](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/), published Dec 9, 2025; threat taxonomy, not a measured attack rate or legal mandate |
| EU AI Act | [Commission overview/timeline](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai): role-/risk-dependent requirements; Annex III Dec 2, 2027 / Annex I Aug 2, 2028, not a universal Aug 2026 red-teaming mandate |
| NIST AI 600-1 | [Generative AI Profile DOI](https://doi.org/10.6028/NIST.AI.600-1), July 2024; [NIST AI RMF overview](https://www.nist.gov/itl/ai-risk-management-framework) expressly describes voluntary use |

Primary pages above were fetched Nov 1, 2026. Star counts, artifact names and competitor
statuses are point-in-time and will change. The public-project scan is bounded to the five
cloned projects, not private submissions or the entire literature; their numerical results
and statistical claims are attributed upstream reports, not independently reproduced here.

---

## 6. Compute facts and sources

| Fact | Source |
|---|---|
| Modal L4 **$0.000222/sec**, A10 **$0.000306/sec**, A100 40GB **$0.000583/sec**; about $0.80/$1.10/$2.10 per GPU hour | [Official pricing](https://modal.com/pricing), fetched Nov 1, 2026; A100 80GB ~$2.50/h, H100 ~$3.95/h |
| $50 ≈ 62.6 h L4 / 45.4 h A10 / 23.8 h A100 40GB **before any other charges** | GPU-only arithmetic, not a whole-project runtime guarantee |
| Shared plan: **33 L4 h + 6 A10 h ≈ $33 core GPU**, $5 one optional analysis, $2 overhead allowance, $10 reserve | [Execution plan §2](08-execution-plan.md); overhead may consume reserve; hours are job caps, not measured runtimes |
| CPU/RAM/storage, startup/idle time, region/non-preemptible premiums and paid external APIs are additional | Modal pricing/FAQ; measured spend and provider caps required; a budget alert is not a hard dollar limit |
| Kaggle T4×2/P100 / roughly 30 GPU h/week are historical possibilities, not confirmed quotas | Inspect each legitimate account's current device/quota/session/storage limits; no pooled or guaranteed quota |
| HF `$20` / conditional `$200` credits have redemption, eligibility and 30-day expiry conditions | Official hackathon Structure; confirm covered HF GPU services and demo access; not Modal credit or automatic ZeroGPU entitlement |

Do not rely on these prices for a budget without re-checking; Modal changes rates and the
dashboard is authoritative.

---

## 7. Unresolved questions / to re-check on Day 1

1. **Final Track 3 fit/form.** Problem statements are now published. Map AgentGuard to trained
   business-workflow/tool/RAG policy repair and reward-hacking defenses; confirm any organizer
   updates. The user is already registered for Track 3; switching is a separate team decision.
   **(Superseded 2026-11-15: see the live-site re-check in §4. No Track 3 / AgentGuard submission
   exists or is planned; treat the registered track as unconfirmed and confirm it with the
   organizers rather than assuming.)**
2. **Real training route and environment.** Resolve/freeze separate Python 3.12 environments;
   measure model fit and run the chosen AgentTune SFT/distill/GRPO smoke. A prompt-only or scripted
   defense is not a genuinely trained repair arm.
3. **AgentDojo/AgentHarm ports.** Whether any AuditKIT/AgentTune integration appeared after this
   review; if yes, reuse instead of building.
4. **Tiny Aya tool-call parse rate.** Must be measured on Day 1; the entire agentic plan depends
   on it, along with complete producer-ID/evidence joins and reset/authorization checks.
5. **CircuitKIT Tiny Aya parity on our hardware.** Must be tested before committing to the
   interpretability stretch.
6. **SafeTune judge availability.** Which judge models are accessible without additional gating;
   choose the fallback metric path accordingly.
7. **HF credit redemption mechanics** (payment method linkage, expiry 30 days after redemption).
8. **New competitor repos.** The crowding map can change by Nov 5; re-run the GitHub searches in
   `11-community-prior-art.md` §8.

---

## 8. Reproducing this research

```powershell
# from D:\ai-sp
git -C research\aligntune log -1 --format='%H|%ad|%s' --date=short
# ... same for each repo in §2

# GitHub org inventory
(Invoke-RestMethod 'https://api.github.com/orgs/Lexsi-Labs/repos?per_page=100&sort=updated' `
  -Headers @{ 'User-Agent'='research' }) |
  ForEach-Object { "{0} | {1} | {2} stars" -f $_.name, $_.description, $_.stargazers_count }

# community search
(Invoke-RestMethod 'https://api.github.com/search/repositories?q=lexsi+hackathon' `
  -Headers @{ 'User-Agent'='research' }).items |
  ForEach-Object { $_.full_name + ' | ' + $_.description }

# HF artifact search
(Invoke-RestMethod 'https://huggingface.co/api/models?search=safetune&limit=20') |
  ForEach-Object { $_.modelId + ' | ' + $_.downloads }
```

---

## 9. Confidence summary

| Area | Confidence |
|---|---|
| Six-tool API/source contracts at the pins | High for source facts; no whole-stack execution claim |
| Caveats/defects | High where source/CPU-supported; absence claims bounded to searched pins/trees |
| CPU reproductions and synthetic join/oracle controls | High for these fixtures only; no model/environment robustness inference |
| Hackathon rules/dates | High for the Nov 1 official-homepage snapshot; final form/updates must be checked |
| Cohere metadata/templates | Source/Hub-supported; account access, parse rate and real hardware fit unmeasured |
| Five public-project descriptions | Source/README-supported bounded scan, not a comprehensive competitor map |
| Community/paper/model results | Upstream reports only; no independent empirical confirmation |
| Modal list pricing | Official point-in-time rates; total job cost/runtime/credit coverage unmeasured |
| NeuroAI monitor, causal and transport feasibility | Hypotheses only; no measured AUROC, intervention efficacy or cost claim |
| CircuitKIT Tiny Aya parity / combined installation | Unexecuted; require real environment and parity gates |
