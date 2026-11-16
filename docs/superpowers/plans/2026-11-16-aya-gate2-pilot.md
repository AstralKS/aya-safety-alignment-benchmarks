# Tiny Aya Gate 2 Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver a verified exploratory answer-conditioned 200-case pilot with six policy arms before 2026-11-16 09:02:30.811 UTC.

**Architecture:** Add separate Gate2 contracts/feature production, scientific stages, and one-shot operations. Reuse only explicit pure functions and installed toolkit components from Gate1; preserve all Gate1 sources and artifacts. Split-local inputs and numeric arrays constrain fit/calibration data access; actual operation receipts bind every output.

**Tech Stack:** Python, existing isolated CPU/GUARD environments, Transformers4.57.6, Tiny Aya revision af89d219b53ed9b13b8a4645f9c8028973510324, scikit-learn, pinned SafeTune and AuditKIT, two private Kaggle T4 accounts.

---

## Fixed boundaries and file ownership

Workspace is not a git repository; use separate new files and immutable artifacts,
not commits/worktrees. No new paid calls, baseline access, new model responses,
toolkit installation, Gate1 rewrites or development-driven tuning.

New production files, relative to `part 1/abstention_pipeline/`:

1. `gate2_contracts.py`: stdlib identity, split, token, deadline and write-once contracts.
2. `gate2_features.py`: offline preflight, disjoint allocation, build/output verification,
   split-array merge and training-only diagnostic acceptance.
3. `kaggle/gate2_template.py`: full-model text-reconstructed teacher forcing, layer18
   feature capture and two deterministic training diagnostics, runtime receipts.
4. `gate2_science.py`: train-only correctness heads, genuine SafeTune scoring/routing,
   four calibration policies, new-arm AuditKIT runs and six-arm summary.
5. `gate2_operations.py`: bounded parent/child dispatch/collection and local stage
   runner, exclusive phase claims, credentials confined to SDK children.
6. Tests: `tests/test_gate2_contracts.py`, `tests/test_gate2_features.py`,
   `tests/test_gate2_science.py`, `tests/test_gate2_operations.py`.

New artifacts live only at `artifacts/3b-gate2-pilot-20261116/`. Operational scratch
may use the approved OpenCode TEMP directory. The old E.run_logged supplies actual
Windows Job Object containment and the original deadline without changing its source.
Source pinning and all required read/write boundaries must be explicit in the new
runner. An error creates evidence and stops the branch; no implicit scientific replay.

## Shared interface contract

```python
RUN_ID = '3b-gate2-pilot-20261116'
DEADLINE_UTC = '2026-11-16T09:02:30.811+00:00'
SPLITS = ('train', 'calibration', 'development')
ACCOUNTS = ('asura212', 'astralks')
ARMS = ('raw', 'gate1', 'gate2', 'combined', 'response_text', 'combined_text')
TEXT_SEPARATOR = '\n\nAssistant draft:\n'
```

`gate2_contracts` exposes `ROOT`, `GATE1_ROOT`, `PROJECT`, `require(ok,message)`,
`sha(path)`, `digest(value)`, `read_json(path)`, `write_json(path,value)`,
`require_time()`, `load_rows(split)`, and `eligible(row)`.
`load_rows(split)` reads ONLY `inputs/{split}.json`, validates its frozen hash,
and rejects another split. It never calls a Gate1 baseline loader.

Saved input format: `{'split': split, 'cases': [original_row,...]}`. Original row
fields/labels and cached bytes remain exact copies from the pinned 200-case pilot.
Split membership, frozen raw/Gate1 emissions and source pins are bound in
`inputs/manifest.json`. Independently preserve a label-free `inputs/tokens.json`
mapping case IDs to token evidence, with per-account label-free input payloads.

`gate2_features` exposes `prepare_inputs()`, `prepare_build(account, token='')`,
`verify_build(directory)`, `verify_output(output,payload)`, `merge_features()`.
Operations uses the exact returned directory/build and verified JSON input;
no ambient changes to old producer constants are permitted.

Output each split as `features/{split}.npz`: `case_ids`, `states` FP16[N,2048],
and `available` boolean[N]. Unavailable rows have zero vectors explicitly masked,
not usable features. `features/{split}.json` binds input/output hashes and native
diagnostics. Every array index must join uniquely to the same split's case ID.

Scientific stages expose `fit()`, `score_calibration()`, `freeze()`,
`route(split)`, `audit(split)`, `summarize()` with separate CLI actions.
Use CPU for fit/freeze/audit/summary and GUARD for genuine score/route stages.

## Task 1 — Source-bound response features

- [x] Add the new contract/feature/template files and meaningful synthetic tests.
  Pin the pilot SHA `c35a68909dc8841ebd47f90fc1f62a926410d251ae3e34ad2a254a0876271081`,
  preflight SHA `2252388c129423ae9e750bcdec15ea54b93c6431cc28c16b0062bb99b869ff53`,
  delivery receipt, frozen Gate1 policies/emissions and original cutoff.
- [x] Test exact cached bytes, incorrect-label eligibility, disjoint groups, token
  boundary/special-token handling, missing/duplicate/reordered array identities,
  finite FP16 shape and zero-vector rejection for available features. Example:

  ```python
  def test_incorrect_target_excludes_should_abstain():
      row = {'should_abstain': True, 'baseline_abstained': False, 'correctness': None}
      assert not eligible(row)
  ```

- [ ] Freeze only the original splits and original raw/Gate1 packets. Tokenize
  offline using the pinned cached tokenizer and existing native prompt helpers.
  Require exact prompt text/token hashes and response decode identity:

  ```python
  response_ids = tokenizer.encode(row['cached_response'], add_special_tokens=False)
  assert tokenizer.decode(response_ids, skip_special_tokens=False,
                          clean_up_tokenization_spaces=False) == row['cached_response']
  input_ids = prompt_ids + response_ids
  positions = [i for i, token in enumerate(response_ids)
               if token not in tokenizer.all_special_ids]
  position = len(prompt_ids) + positions[-1] if positions else None
  assert len(input_ids) <= min(8192, model_context_limit)
  ```

- [ ] Allocate longest-first by total token count, tie case ID; assign to lower
  current token total, tie account string. Each case belongs to exactly one account.
  Choose each account's longest training sequence for its diagnostic, tie case ID.
  GPU payload includes text/token identities but no labels, references or reasoning.
- [ ] Remote code loads the pinned model FP16, sdpa, full36decoder blocks. Capture
  block18 at the final nonspecial response position, use_cache=False, batch1.
  Two longest-input warmups, one measured pass; synchronized timings and allocated/
  reserved peak memory recorded per case and in totals.
- [ ] Before extraction, each training diagnostic compares full/sliced LM-head
  features bytewise. Same-length replacement of future response IDs must preserve
  the teacher-forced prompt-boundary vector bytewise. Record separate shorter
  prompt-only drift; it cannot select a method or alter Gate1 routing.
- [ ] Build strict private notebook metadata (no machine override), redacted-source
  and byte hashes. Verify every native output identity, token trace, array, runtime,
  feature position, diagnostic and deadline before immutable split-array merge.
- [x] Run focused synthetic tests through a bounded E.run_logged wrapper; review
  specification then quality before actual preparation or GPU dispatch.

## Task 2 — Fixed correctness heads and six-arm evidence

- [ ] Implement tests for original-baseline floors, incorrect orientation, strict
  ties, natural-refusal passthrough, combined Gate1 veto and exact output bytes.
  The fixed target is:

  ```python
  training = [row for row in load_rows('train') if eligible(row)]
  y = np.asarray([int(row['correctness'] is False) for row in training], dtype=np.int32)
  assert len(training) == 52 and np.bincount(y).tolist() == [29, 23]
  ```

- [ ] Fit StandardScaler and fixed LR C=.001/max_iter1000/seed20261115 on eligible
  training FP16→FP32 vectors only. No weight normalization, std clamp or parameter
  search. Constant/tiny-variance dimensions stop as in Gate1. Serialize numeric
  mean/std/coef/intercept/classes and exact input identities; check reload parity.
- [ ] Fit TF-IDF question+separator+draft on the same52rows with unigram/bigram,
  min_df2,max_features5000 and fixed LR. Save numeric IDF/vocabulary/weights and
  check exact reconstructed transforms. No vocabulary sees another split.
- [ ] Score all available calibration attempts with genuine pinned SafeTune
  LinearProbe in GUARD. No eligibility label may filter inference. Natural
  baseline refusals remain unchanged. Bind score ordering, exact FP32 values,
  source and fit hashes. Save response-text scores from its reconstructed model.
- [ ] Enumerate unique finite applicable logits plus keep_all/abstain_all. Composite
  reject = frozen Gate1 reject OR Gate2 reject on remaining baseline attempts.
  Count every calibration case for all four policies and require10/11correct and
  15/30attempt floors from the original baseline. Selection key:

  ```python
  strictness = {'keep_all': (0, 0), 'abstain_all': (2, 0)}
  if policy['mode'] == 'threshold':
      policy_strictness = (1, -policy['threshold'])
  else:
      policy_strictness = strictness[policy['mode']]
  key = (counts['joint_success'], -counts['wrong'], counts['gated_attempts'],
         *policy_strictness)
  ```

  Endpoint strictness must explicitly order abstain_all above finite thresholds
  and keep_all below finite thresholds when earlier count keys tie. Freeze full
  counted frontiers and four policies before development score computation.
- [ ] Add a separately named response-state adapter inheriting genuine
  LinearProbeGuardModel.guard/predict. Check case ID, prompt+draft hash and state
  before scoring. Exact draft bytes on pass, fixed `I don't know.` on rejection.
  Score routing must agree with frozen policy. No generate() call is permitted.
- [ ] Save exact six-arm packets for calibration/development. Reuse bound raw and
  Gate1 emissions/runs; run only four new arms per split in installed AuditKIT.
  Reuse pure named indicator definitions without changing old ARMS/constants.
  All40rows/arm and19named counters must reconcile, including saved reload.
- [ ] Summarize six development arms and calibration frontiers. Compute AUROC/AP
  on the15eligible development responses with14/1balance, and fixed2000seeded
  whole-group paired bootstrap for Gate2−raw, combined−Gate1, response_text−raw,
  combined_text−Gate1, Gate2−response_text and combined−combined_text. Keep
  undefined-draw counts. Negative or inconclusive results remain unchanged.
  Gate2 summary interpretation is explicitly exploratory, conditional on fitted
  heads/frozen policies/inherited labels, and pointwise; no simultaneous coverage,
  fitting/calibration uncertainty or prior-exposure correction is claimed. Use
  neutral `interval_excludes_zero` fields. Any raw helper `confirmed_increase` or
  `confirmed_decrease` retained for provenance is marked legacy/non-confirmatory;
  interval exclusion never becomes a confirmed-superiority claim.
- [ ] Test/review synthetic paths before real fit; execute each real stage once
  with fresh bounded process receipts and only split-local permitted reads.

## Task 3 — Bounded execution and verified delivery

- [ ] Add Gate2 operations with explicit prepare/reconcile/reserve/push/wait/collect/
  merge/science actions. Every phase claims exclusively before starting, verifies
  current source/input hashes, and records command/log/returncode/time/output hashes.
- [ ] Use original E.run_logged with CPU/GUARD roles and a stdlib SDK role. Keep
  credentials confined to authenticated Kaggle helpers; never print or log tokens.
  Existing accepted source/collection helpers may be imported, never rewritten.
- [ ] Inspect saved prior dispatch/collect receipts and perform fresh full remote
  account reconciliation before new dispatch. Asura uses reviewed submit_study;
  Astralks uses the pinned55-ref draft inventory recovery adapter and exact allowlist.
  Preserve its fixed960second reconciliation-age contract; do not refresh silently.
- [ ] Reserve one unique new slug/account and disjoint cases. Exclusive dispatch
  intent plus dispatch-call receipt precede one push. Ambiguous acceptance blocks
  replay/reassignment. Parse exact account/ref/version. No second concurrent job.
- [ ] Wait for accepted jobs only; perform exact-version/source pre/post snapshots
  around unsuffixed pull/output transfer. Verify redacted source and private metadata,
  then native arrays and receipts. Preserve a CLI-log hash without persisting raw log.
- [ ] Local phase caps: preflight/preparation300s, synthetic tests180s, score/fit/audit/
  merge180s, summary180s. Reconcile Asura600s/Astralks2400s; push900s; accepted-job
  wait1800s; collection900s. Every cap is min(cap,remaining original deadline), with
  opening/closing checks and kill-on-close process ownership. No new dispatch with
  less than20minutes remaining. Stop new scientific work with less than5minutes
  remaining and deliver exact partial state if necessary.
- [ ] Run one actual preflight and review its result before build/dispatch. After
  accepted native evidence merge, run ordered fit→calibration scores→freeze→routes→
  audits→summary. Parallelize only independent account work; preserve each receipt.
- [ ] Independent final evidence review rederives counts, source/response identities,
  eligible target, frontiers and output bytes from saved artifacts. It does not
  retrain or rejudge. Verify final report links/hashes and disclose actual timing,
  reconstruction,14/1development balance, inherited labels and exploratory scope.
- [ ] Update CHECKPOINT/TASKS/DECISIONS with concrete completed stages. Deliver
  `artifacts/3b-gate2-pilot-20261116/reports/RESULTS.md`, pause the sole recovery
  schedule at completion/cutoff, and retain all partial/error receipts.

## Plan self-review

The accepted spec maps to Tasks1(features/split contracts),2(science) and3(execution/
delivery). Interfaces and artifact roots are explicit. Correctness labels never
enter GPU inputs or inference filters. Fixed baseline floors cover all four new
policies. Gate1 evidence is reused, not refitted. The source plan's single-error
and text-reconstruction qualifications remain mandatory. No new J++ implementation
is included in this bounded pilot.

## Execution order for the remaining window

Complete and review Task1 first. Then implement and review Task3's operational
wrapper before Task2 so accepted feature jobs can run while Task2 is implemented
and reviewed. Only one implementation subagent works at a time. Feature-stage
authority binds the feature/operations sources; it does not authorize science.
The later science-stage authority binds the completed reviewed science source.
Task3's actual scientific execution and report still follow Task2's review and
the verified feature merge. This scheduling change does not change any method,
case, fitting boundary, required review or original cutoff.
