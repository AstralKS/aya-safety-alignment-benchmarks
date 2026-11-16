# Frozen Gate1 / Gate2 preliminary four-benchmark test

## Authority and endpoint

The user requested parallel work and exactly two more hours from
2026-11-16T08:39:22.388237Z, then requested Gate1 and Gate2 test comparisons across
all existing benchmarks with preliminary results at minimum. Deadline:
**2026-11-16T10:39:22.388237Z**. Existing delegated decision/review authority in
`part 1/abstention_pipeline/DECISIONS.md` applies. This document is subject to that
independent review before implementing the new benchmark-evaluation behavior.

Use only Tiny Aya global revision af89d219b53ed9b13b8a4645f9c8028973510324, private
artifacts and existing cached responses/labels. The four-benchmark baseline report
is final. This is a separate intervention evaluation, with no new benchmark,
judge call, answer generation, toolkit installation, 8B run or publication.

## Approaches and decision

1. **Recommended: 100 label-independently selected held-out cases per benchmark.**
   Freeze Gate1's existing model/policy and fit Gate2 only on the accepted pilot.
   This covers every benchmark and leaves time for native extraction and review.
2. Evaluate every available held-out case. This increases precision but risks no
   finished four-benchmark comparison within the execution window.
3. Finish the 40-case pilot only. It gives the fastest result but does not meet the
   user's four-benchmark test objective.

Choose option1. The target is400 test cases plus the original200-case Gate2 pilot.
If a completed account supplies only part of the selected cases, publish the exact
completed subset with its deterministic allocation, missing counts and selection
limitations; do not replace failures with easier cases or imply full completion.

## Frozen training and calibration

Retain the accepted Gate2 pilot design exactly:120train/40calibration/40development;
52 attempted answerable training drafts (29correct/23incorrect);18eligible
calibration drafts; layer18 final nonspecial cached-response-token state. The
incorrect-response LR and response-text LR use fixed hyperparameters. Four new
policies are selected on the original40calibration cases, preserving original
correct/attempt floors, then frozen before scoring any development or test cases.
No new fit, vocabulary, threshold or model selection uses the test benchmarks.

Gate1 uses its existing frozen question-state head and primary policy. Benchmark
Gate1 extraction must use the same tokenizer, prompt constructor, layer and token
coordinate as its training recipe. Gate2 uses the accepted reconstructed
question-prompt plus cached-response recipe. Where that prompt differs from the
original baseline generation prompt, record both identities and disclose the
context shift; do not claim exact original-generation states.

## Test selection and leakage

Benchmarks: SimpleQA Verified, HalluLens PreciseWikiQA, AbstentionBench, SelfAware.
First join source IDs, exact question/cached-response bytes and existing final
labels, validating their saved provenance. Exclude all original200pilot question
groups and any additional groups actually used in fitting/calibration. Preserve
source-specific group IDs and also compare normalized-question identities across
all four sources; normalize Unicode NFKC, casefold, replace `[^\w\s]` with spaces,
and collapse whitespace, matching the original grouping rule. Union existing
AbstentionBench groups with cross-source exact-normalization links; exclude any
expanded component touching an exposed member, even if its anchor changes.
For AbstentionBench, restrict candidates to its original final-test partition
(195rows/186groups) rather than using unexecuted original train/calibration rows.

For each benchmark, group candidates by the resulting connected components.
The canonical group anchor is the lowest SHA256 of normalized member questions
across the full component, including noncandidate members. Rank groups by
SHA256 of `aya-fourbench-test-v1\n` + benchmark slug + `\n` + canonical group anchor.
Take the lowest100 groups, selecting the lexicographically first stable source ID
within each group. Resolve cross-benchmark duplicates in fixed order SimpleQA,
PreciseWikiQA, AbstentionBench, SelfAware, refilling from each benchmark's ranking.
One representative per group is intentional:100groups yield100selected rows,
with a group-representative sampling interpretation rather than an IID row sample.
Neither correctness, refusal label, response length nor feature availability is
part of selection. Freeze selected IDs and hashes before fitting/scoring. Report
whole eligible pool sizes, exclusion counts and exact selected IDs. Broader prior
baseline inspection is disclosed: these are held out from fitting/calibration,
not newly discovered benchmark sources.

## Data contract and metrics

Each local test case contains: unique namespaced case_id, benchmark/source_id,
question, normalized question group, exact cached_response, baseline_abstained,
optional should_abstain and optional correctness, explicit applicability flags,
and exact source path/hash/row references. Missing labels remain null. Do not use
answerability labels as correctness, or classify an unanswerable attempted response
as a judged factual error. Dataset-specific label mappings and valid denominators
must be written in the frozen manifest after source inspection.

Every arm emits either the exact cached draft or the fixed `I don't know.`;
natural baseline refusals pass through in the new arms. Arms: raw, gate1, gate2,
combined, response_text, combined_text. Composite Gate1 veto applies to baseline
attempts; no correctness labels select inference rows. Missing features retain
the cached draft, except an available frozen Gate1 veto in composite arms.

Report counts and denominators for every metric, including:

- All benchmarks: coverage/abstention rate, additional rejection rate, attempt
  retention and feature availability; compare same selected case IDs across arms.
- Where correctness exists: correct-answer yield, retained incorrect-answer count,
  incorrect fraction among scored retained attempts, and correct-answer retention.
- Where should-abstain exists: appropriate-abstention behavior accuracy, false
  refusal and abstention recall. Joint task success only when the required
  correctness labels are available for all answerable attempts.
- SelfAware: raw official unknown-label precision/recall/F1; answerable reference-
  substring correctness/retention; and separately named operational abstention
  metrics after gating. The frozen refusal `I don't know.` lacks an inherited
  official phrase/SimCSE label. Its official unknown prediction is null; whole-
  sample official post-gating unknown F1 is unavailable if any forced refusal
  lacks that classification; the same applies to every full-sample metric that
  needs the missing official detector label. Report its unclassified count.
  Freeze the hybrid operational rule as forced_rejection OR cached_official_unknown
  for these unchanged-or-forced outputs. Do not rename operational F1 as official
  SelfAware F1. Reference-substring measures remain answerable-only proxy metrics.
- PreciseWikiQA: preserve the original hallucination/refusal label semantics and
  denominator. Do not silently equate missing labels with factual support.

The four new arm audits use the already installed AuditKIT precomputed-input
runner. No judge or generate call is allowed. Reuse existing pure named indicator
logic only when its assumptions match the dataset's available labels; otherwise
emit explicit benchmark-specific counts. Source bindings and exact emitted bytes
permit an independent recount without repeating an experiment.

Paired whole-question-group bootstrap uses2000 draws and seed20261115, conditional
on fixed models/policies/labels. Report pointwise exploratory intervals and
undefined denominator draws. Interval exclusion is not confirmed superiority;
there is no fit/calibration-uncertainty or multiplicity correction.

## Extraction and execution architecture

Preserve the failed original Gate2 root and exact pre-extension source snapshots.
New pilot root: `artifacts/3b-gate2-extension-20261116/`.
Separate test root: `artifacts/3b-gates-fourbench-20261116/`.

Prefer one new disjoint native job per account containing its original-pilot share
plus its selected test share. GPU payloads are label-free. Additive benchmark
features have a separate envelope/array identity; pilot training/calibration/
development schemas and diagnostic selection remain unchanged. Extract Gate1
question-only states in a separate prompt-only forward; do not substitute the
teacher-forced prompt boundary without a verified equivalence. Gate2 extracts the
final nonspecial draft state. Zero/missing vectors are masked, never scored.

Each account must pass the accepted training-only original/sliced byte parity and
causal diagnostic before using logits_to_keep=1. Benchmark cases never select the
method. The original full-model36block FP16/sdpa, batch1, no-cache recipe remains.
All native receipts bind revision, tokens, coordinates, arrays, timing and deadline.

Fresh authenticated account reconciliation precedes a unique new dispatch;
preserve the historical55-ref allowlist and accepted old-job receipts. No replay
of accepted or ambiguous jobs. New deadline checks propagate through wrapper,
inventory adapter, native payload and process containment; old Gate1 sources are
not edited. At least20minutes must remain for dispatch and5minutes for new science.

## Verification and delivery

Independent design/spec and quality review precede execution. Meaningful synthetic
checks cover leakage, label applicability, frozen-policy transfer, masks, exact
bytes, metric denominators and native/source joins. Existing successful Gate1
experiments and delivery verifiers are not rerun.

Deliver `artifacts/3b-gates-fourbench-20261116/reports/RESULTS.md` with machine-readable
summary, selected IDs, source hashes, all per-arm counts and explicit completion
status. Link the separate Gate2 pilot report. State clearly that the pilot's
development correctness diagnostic has only one incorrect eligible response.
Keep negative findings and gaps. At deadline, stop new execution and report the
exact verified subset; no pending/missing result is represented as completed.
