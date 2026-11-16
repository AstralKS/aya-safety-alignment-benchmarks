# Frozen Gate1 / Gate2 Four-Benchmark Implementation Plan

> **For agentic workers:** Use subagent-driven-development for disjoint tasks and
> specification then quality review. Explicit user parallel authorization applies.

**Goal:** Deliver verified preliminary Gate1/Gate2 comparisons on100held-out cases
per existing benchmark by2026-11-16T10:39:22.388237Z.

**Architecture:** Keep the accepted200-case Gate2 pilot's fit/calibration intact.
Add a separate source-bound test manifest, additive native test feature envelope,
and frozen-policy evaluator. Share two disjoint native jobs while preserving every
old experiment and the failed preflight.

**Tech Stack:** Existing Python CPU/GUARD environments, Tiny Aya global,
Transformers4.57.6, pinned SafeTune/AuditKIT, NumPy/scikit-learn, private Kaggle.

---

## Ownership and fixed roots

All production paths below are relative to `part 1/abstention_pipeline/`.
This workspace has no git repository; use immutable source snapshots and hashes.

- Feature owner: `gate2_contracts.py`, `gate2_features.py`,
  `kaggle/gate2_template.py`, their tests.
- Operations owner: `gate2_operations.py`, optional
  `gate2_extension_inventory.py`, their tests.
- Pilot science owner: `gate2_science.py`, its tests; existing accepted task.
- Data owner: new `gate_benchmark_data.py`, `tests/test_gate_benchmark_data.py`.
- Evaluation owner: new `gate_benchmark_eval.py`,
  `tests/test_gate_benchmark_eval.py`.
- Parent: design/plan/decision/checkpoint, source authorities and stage dispatch.

```python
PILOT_RUN = '3b-gate2-extension-20261116'
TEST_RUN = '3b-gates-fourbench-20261116'
DEADLINE = '2026-11-16T10:39:22.388237+00:00'
BENCHMARKS = ('simpleqa_verified', 'precisewikiqa', 'abstentionbench', 'selfaware')
ARMS = ('raw', 'gate1', 'gate2', 'combined', 'response_text', 'combined_text')
```

## Task1: deadline and explicit recovery

- [ ] Snapshot all changed old sources byte-for-byte under the original Gate2
  `reviews/source-before-extension/`; retain failed-phase receipts.
- [ ] Set Gate2 contracts to the new root/deadline. Parse exact timezone-aware
  deadline datetimes in the native template rather than millisecond-only equality.
- [ ] Retain module name and sanitized traceback on preflight failure. Establish
  missing import through bounded import diagnostics before changing an interpreter.
- [ ] Update only new execution/inventory adapters to accept the extension while
  binding the historical55-ref allowlist as old evidence. Do not edit Gate1 helpers.
- [ ] Verify deadline propagation and exclusive phase/intent behavior with focused
  tests, then obtain SPEC and QUALITY decisions for the new source identities.

## Task2: source-bound test selection

- [ ] Implement `gate_benchmark_data.prepare()` with a standard-library-only
  read-only adapter for each final benchmark artifact identified in
  `reviews/benchmark-data-discovery.md`. Never import/run an old scorer.
- [ ] Join cached response to exact source ID and existing verdict. Validate every
  selected response hash and judge/source binding, preserving null applicability.
- [ ] Build the union of actual fit/calibration groups plus all200pilot groups.
  Preserve their191original groups and merge cross-source identities using original
  NFKC/casefold/punctuation-to-space/whitespace normalization. Exclude components
  touching exposed members. Abstention candidates must be original final-test rows.
- [ ] Implement deterministic group selection exactly as the design specifies:

```python
def rank(benchmark, canonical_group_anchor):
    import hashlib
    return hashlib.sha256(('aya-fourbench-test-v1\n' + benchmark + '\n' +
                           canonical_group_anchor).encode('utf-8')).hexdigest()
```

- [ ] Freeze `inputs/manifest.json`, `inputs/cases.json` and
  `inputs/label-free-cases.json` in TEST_RUN with exclusive writes. Manifest binds
  selected IDs, case/source hashes, exclusions, dataset label mappings and every
  applicable denominator. Cases retain both gate-feature and original-generation
  prompt identities; any context difference is explicit.
- [ ] Meaningful synthetic checks: variant normalized questions excluded across
  benchmarks, source duplicate joins rejected, correctness-null preserved, and
  selected IDs unchanged after labels/responses are permuted within source IDs.
- [ ] Obtain SPEC then QUALITY review; parent executes preparation once using
  a bounded process and records exact manifest hashes before any test scoring.

## Task3: additive dual-gate native test features

- [ ] Feature owner consumes only the hash-bound label-free test envelope in
  addition to original pilot rows. Keep pilot `COUNTS`/splits/diagnostics unchanged.
- [ ] Validate/tokenize each test case with the fixed cached tokenizer and prompt
  helpers, preserving exact cached bytes and no truncation. Unsupported context
  becomes an explicit unavailable test feature, not case replacement.
- [ ] Allocate all available pilot/test traces longest-first, balancing token load
  including the additional prompt-only Gate1 forward. Diagnostics select only
  assigned pilot training rows. Each test ID appears on exactly one account.
- [ ] Native template emits separately identified FP16[N,2048] question-only and
  response-final states, bool masks, exact case IDs and traces. Question-state
  forward follows the frozen Gate1 recipe; response-state forward follows Gate2.
- [ ] Extend output verification/merge to validate extra files and source identity
  without changing the pilot's split-array schemas. Test arrays/receipts are saved
  under the test root only after both account proofs reconcile, or as explicit
  per-account partial features for reporting if one account cannot complete.
- [ ] Test mixed pilot/test allocation, train-only diagnostic selection, mask and
  wrong-coordinate rejection, exact arrays/IDs/files; SPEC then QUALITY review.

## Task4: fixed pilot science and test transfer

- [ ] Finish the already accepted `gate2_science.py` stages; review and execute
  fit→score-calibration→freeze before any development or benchmark scoring.
- [ ] Implement `gate_benchmark_eval.run()` to require bound frozen Gate1 fit/policy,
  four frozen Gate2 policies, both trained heads and verified test feature receipts.
- [ ] Score available baseline attempts with the actual pinned SafeTune numeric
  component. Use the existing exact Gate1 policy decision and four independently
  calibrated Gate2 policies. Text transforms use the saved training vocabulary only.
- [ ] Route all six arms. Every pass preserves exact cached UTF-8; every new veto
  emits `I don't know.`. Baseline refusals remain unchanged. Missing features do
  not enter a head; composite veto from the other available frozen head may apply.
- [ ] Save local per-case scores/decisions/emissions with model/policy/source hashes.
  Compute only label-supported named counters. In particular:

```python
correct_yield = retained_correct / n if correctness_is_complete else None
unknown_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
coverage = retained_attempts / n
```

  The exact metric-specific applicability is recorded with numerator/denominator;
  no nullable field may be silently coerced to false.
- [ ] SelfAware forced-refusal official classification remains null unless an
  exact inherited label exists. Report separate operational unknown F1; raw official
  F1 and retained reference-substring accuracy preserve their original semantics.
- [ ] Run installed AuditKIT on precomputed emissions using dataset-appropriate
  named counters, concurrency1/retries0. Assert zero generate/judge calls.
- [ ] Emit paired2000-draw seeded group-bootstrap contrasts, preserving undefined
  draws and neutral exploratory wording. Do not rerun a fit or tune a threshold.
- [ ] Test label-independent routing, equality-threshold behavior, component
  orientation, composite/missing-feature behavior, null correctness and denominators.
  Obtain SPEC then QUALITY review before actual evaluation.

## Task5: execution and delivery

- [ ] Parent verifies source-bound positive decisions and issues new preflight
  authority. Run one exclusive preflight; inspect actual output before building.
- [ ] Inspect old saved receipts, reconcile both accounts authentically, reserve
  unique new refs, record intent and accept at most one new job per account.
- [ ] Wait/collect exact accepted versions once; bind diagnostics and native arrays.
  No duplicate work after an ambiguous push or failed scientific phase.
- [ ] Verify/merge pilot features, execute reviewed fixed pilot stages once, then
  execute frozen test evaluator. Reuse completed raw labels without any new judge.
- [ ] Independent reviewer recounts final per-case artifacts and checks exact
  source/policy/feature identities and every reported denominator without retraining.
- [ ] Deliver test `reports/RESULTS.md` and `reports/summary.json`, including actual
  sample counts for each benchmark/arm, pilot link and explicit incomplete subsets.
- [ ] Update checkpoint/tasks/decisions and pause the sole15-minute reminder.

Every local execution is bounded by the new cutoff with actual process receipts.
Retain20-minute dispatch and5-minute science admission minima. If time expires,
deliver only verified actual results and exact missing stages; do not imply a
completed four-benchmark gate test when extraction/scoring is incomplete.
