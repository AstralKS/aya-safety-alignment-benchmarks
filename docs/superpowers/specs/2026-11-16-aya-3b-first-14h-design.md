# Aya 3B first: complete prompt-only abstention study within 14 hours

## Decision and fixed deadline

The user selected **Full 3B study**: a 200-case pilot, evidence review, then a
bounded larger AbstentionBench study with held-out evaluation and controls.
Both Kaggle accounts serve the 3B study. **No 8B work starts before the 3B result
is complete and verified.** Starting 8B also requires separate approval.

The original request is session message `msg_1e096f017001mqQ5T1FGHUCVB8`:

- Start: **2026-11-15 19:02:30.811 UTC**.
- Hard stop: **2026-11-16 09:02:30.811 UTC / 14:32:30.811 IST**.
- Planning, installation, queue time, reviews and recovery consume this same window.
  Never restart the clock. This is a delivery target, not a guarantee of a positive
  result or permission to bypass an experimental review.

This workspace is not a Git repository. No repository initialization, worktree,
commit or publication is implied. All experiment writes stay under
`part 1/abstention_pipeline/`; this separate design leaves original research intact.

## Approaches considered

1. **Full prompt-only 3B study — selected.** Complete one interpretable experiment
   with held-out utility/coverage evidence before pursuing a second model.
2. **200-case pilot only.** Faster, but fitting and exploratory development scores
   alone do not establish held-out performance. Rejected as the requested endpoint.

Splitting the accounts between 3B and 8B is not an option under the latest direction.
Response-conditioned Gate 2, retrieval, steering, extra datasets and publication
are outside this design. LayerMix is not required for completion; retain the fixed
single-layer method rather than expanding the critical path.

## Scientific endpoint

Test whether a prompt-only linear probe can suppress **undesirable answering on
questions that should be abstained from**, while retaining useful cached answers.
This is not a claim that the probe predicts every factual hallucination or improves
Aya's underlying knowledge. Rejection cannot increase correct-answer yield.

Core model: `CohereLabs/tiny-aya-global`, revision
`af89d219b53ed9b13b8a4645f9c8028973510324`, FP16, SDPA, serial forwards.
Reuse the verified smoke's native system-plus-user rendering, tokenizer contract
and seven layers `[12,16,18,19,23,26,30]`. The primary probe uses fixed layer **18**,
standardized logistic regression `C=0.001`, `max_iter=1000`, seed `20261115`.
Reference answers, cached responses and labels never enter feature text.

Required controls: unchanged cached policy, always abstain, deterministic random
rejection and prompt-only TF-IDF logistic regression. Random matching is an
evaluation control, not a deployable method selected using final-test labels.
Any matched-rate random control must be labelled as such and cannot choose the
primary probe or its operating threshold.

## Data and review boundaries

- Only existing **978 AbstentionBench cases** and their model-specific cached
  answers/judgments are in scope. Do not rescore the frozen benchmarks or add a
  benchmark. Do not use the dropped datasets or archived projects.
- Import existing abstention and applicable conditional-correctness labels with
  exact journal, response and model/revision bindings. Missing/error/unknown labels
  cannot become a negative class or quietly disappear from evaluation.
- CuratorKIT runs real schema validation and counted exact deduplication, with
  cleaning, synthesis, judging and recovery disabled. Canonical accepted records
  retain exact text and lineage and are consumed downstream.
- Preserve the existing question-group partition algorithm and seed. Audit exact
  and high-overlap duplicates before freezing membership; group integrity outranks
  target ratios. A grouping change needs review and a new artifact, not overwriting
  the smoke's split. Target train/calibration/development/final proportions are
  approximately **60/10/10/20**; report actual independent-group counts.
- Pilot target: **120 train / 40 calibration / 40 development**, balanced by
  answerability where group constraints permit. Never sample by Aya correctness.
  Final-test cases are neither uploaded nor extracted during pilot tuning.
- Reuse verified smoke features for a pilot case only when every source, prompt,
  token-ID, layer, revision and feature binding agrees. Do not rerun those cases
  merely to populate a new feature directory.
- Fit scalers and probe/text weights on training only. Select the threshold on
  calibration only, requiring observed correct retention >=0.90 and attempted-answer
  retention >=0.50. Maximize joint task success subject to these floors; break ties
  by higher attempt retention, then the lower score threshold (stricter rejection,
  since a higher should-abstain score is rejected). Keep-all/abstain-all endpoints
  are explicit policy branches, never infinities fed to a runtime logit threshold.
- With no applicable baseline-correct calibration cases, the useful-retention
  constraint is undefined: stop for review. Small calibration denominators are
  disclosed with uncertainty and do not establish a population-level guarantee.
- Present the pilot development report **before** any larger extraction. Scaling
  requires explicit review of utility, controls, provenance and measured runtime.
  A negative/inconclusive pilot is retained; it is not relabelled a successful full
  held-out study. Bounded expansion of an inconclusive pilot needs user approval.
- After scale approval, extract only missing non-final features, refit on full
  training and recalibrate on full calibration. Freeze the complete release policy
  before final-test extraction/evaluation. Evaluate the final policy once. Do not
  retune it on final-test outcomes.

## Two-account execution design

Accounts: **asura212** and **astralks**. Verify each identity and reconcile current
jobs before dispatch. Report quota as unknown if no authoritative interface exposes
it; do not infer hours remaining from the completed smoke or assume both have quota.

- One active experiment GPU notebook per account, at most **two total**.
- Submit private **notebooks**, with `machine_shape` omitted. Record actual selected
  hardware; no silent precision/runtime substitutions.
- Allocate each stage's approved missing cases into disjoint account shards using
  verified prompt-token totals, longest-first balancing, with deterministic ties.
  Sampling/splits remain independent of account allocation and model correctness.
- Each shard has a unique stage/account/manifest-derived notebook identity, immutable
  code/metadata and a dispatch receipt. A possibly accepted push is never replayed.
- Normal extraction uses **one measured pass**, retaining necessary warm-up and
  prompt checks, not three full repeat passes. No unapproved batching/decoder sweep.
- While notebooks run, local work covers source validation, probe/control fitting,
  toolkit integration and synthetic tests. Do not keep a GPU idle for local fitting.
- Collect and verify each completed shard once. Merge only after exact case-set,
  ordering, prompt/token hashes, revision, layers, finite states and output hashes
  agree. Reject duplicates, missing cases and incompatible shards.
- Failures do not trigger speculative replacement on the other account. Reconcile
  accepted work first; only undispatched cases may be reassigned. If one account is
  unavailable, use the other without lowering scientific checks or restarting work.

Cross-account numerical variation is recorded rather than silently treated as exact
tensor replay. Source/token parity and finiteness are required on both accounts.

## Real toolkit chain and component interfaces

`source-bound cached cases → CuratorKIT accepted records → grouped feature shards
→ local fitting/calibration → SafeTune guard decisions → AuditKIT paired run packets
→ independently verified private report`.

1. **Preparation/labels:** canonical JSONL and counted curation manifest, consumed by
   extraction and fitting. Pin the three existing local toolkit source revisions;
   install only in isolated environments and save resolver/version locks.
2. **Feature runner/collector:** immutable input manifest to FP16 numeric features
   and execution receipts. Generalize the reviewed notebook submission path without
   changing its smoke history, output-containment checks or replay barriers.
3. **Probe/control fitting:** train features/text and labels to numeric NPZ + JSON,
   including scaler, coefficient, class orientation, threshold and split hashes.
   No arbitrary-object pickle. Deterministic reload/score tests are required.
4. **SafeTune routing:** use `LinearProbe` and actual inherited `guard()` routing,
   not `generate()` (which bypasses the gate at the pin). For offline evaluation a
   case-bound feature adapter supplies verified cached states; label it **cached
   policy evaluation**, not live model generation. Kept output is byte-identical;
   rejected output is exactly `I don't know.`. No new Aya responses are generated
   for this study. Include real runtime-wrapper parity
   validation on approved training cases in a uniquely named validation notebook;
   this checks deployment equivalence, not a duplicate extraction study.
5. **AuditKIT:** separate raw/control/gated `precomputed` runs, custom source-bound
   indicator metrics, concurrency 1 and zero retries. No generator or judge client.
   Exact case joins and response hashes precede inherited correctness labels.
6. **Reporting:** verified paired counts, useful-coverage frontier, uncertainty,
   runtime/resource measurements, failure status and reproduction commands.

Independent specification/code-quality reviews remain at the source-label,
dispatch/merge and routing/evaluation boundaries as required by the existing plan.
Do not count imports or mocked integrations as substantive toolkit use.

## Report and completion contract

Report, on the identical held-out IDs for every arm:

- Attempted coverage and attempt retention.
- Correct-answer yield and correct retention.
- Undesirable-answer rate on should-abstain questions and false refusal on answerable
  questions.
- Behavior accuracy, joint task success and answerability confusion counts/F1.
- Selective factual risk only where existing correctness judgments apply.
- Paired question-group bootstrap intervals (2,000 fixed-seed draws), including
  undefined-draw counts; no defective upstream bootstrap p-value.
- Actual sample/group counts, leakage audit, extraction/guard overhead, peak memory
  and package/hardware provenance. Cached lookup is not generation latency.

A complete 3B result needs actual held-out artifacts, all case joins reconciled,
the three real toolkit stages, fresh verification and an honest conclusion. A
positive effect is not required. An interval including no benefit is inconclusive,
not confirmed improvement. If a review/contract failure prevents final evaluation,
report the stopped stage and verified partial findings, not "experiment complete".

The final split is held out from **probe fitting/selection**; the historical baseline
labels were already observed, so it is not a pristine unseen benchmark.

## Time allocation and hard stops

These are elapsed targets from the original start, not fresh allowances:

| Elapsed time | Target |
|---|---|
| 0–2 h | Isolated environments, source adapter, curation/splits, pilot notebooks |
| 2–4 h | Pilot fitting/controls, SafeTune/AuditKIT integration, evidence review |
| 4–8 h | Approved larger non-final extraction, refit and recalibration |
| 8–10 h | Freeze policy, approved final extraction and one held-out evaluation |
| 10–12 h | Independent verification and private report/reproduction bundle |
| 12–14 h | Queue/error/review reserve; no scope expansion |

At the deadline, start no new remote work or fitting. For active accepted notebooks,
use a supported cancellation method only if available; never guess a session ID.
If cancellation needs the user, disclose the active job and do not claim all compute
has stopped. Preserve receipts, verified partial outputs and the exact stop state.

Earlier hard stops: unauthorized cost/platform; changed anchors/cached text; template
or token mismatch; leakage; unusable supervision; unresolved remote status; merge
failure; utility constraints failing; or unreviewed scale-up.

Keep exactly one 15-minute recovery timer, currently
`sch_1e039c0e20016x7HbBQioO5B3I`. It cannot bypass pending input/permissions or launch
duplicate work. The latest user instruction to keep it active supersedes the older
smoke-only automatic-pause wording; stop active work at the deadline and preserve
the checkpoint without rearming another timer.

No paid judging execution, public upload, human agreement campaign, 8B, Gate 2 or
change to `part 1/baselines/` is authorized here. After written-design review the
user offered **up to $3 for verification** and requested short reviews/inference
optimization. Record that allowance within the existing $18 ceiling, but leave
paid execution disabled until the verification workload/judge/cost contract is
defined; the cached study still requires no new paid judgment. Historical spend
remains $3.889081230; remaining headroom is $14.110918770, not a new $18 allowance.
Optimization must preserve inference parity and is not required before obtaining
the first 3B result. Do not access `not needed/`, `docs/06*` or `docs/08*`. Preserve
original research/reference documents.
