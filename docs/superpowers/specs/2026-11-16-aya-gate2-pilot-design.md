# Tiny Aya Gate 2: buffered response-probe pilot

**Status: accepted for bounded implementation on 2026-11-16 after Gate1 delivery
and its full feedback grace; no Gate2 result exists.**
Presented during Gate 1 native validation on 2026-11-16. The user authorized moving
to Gate 2 after Gate 1 and delegated decisions while absent. The native-inclusive
Gate 1 delivery, its full 15-minute feedback grace, and a recorded design-review
decision precede Gate 2 implementation/execution. The original cutoff remains
**2026-11-16 09:02:30.811 UTC**.

## Question and recommendation

Does a frozen model's response-conditioned state improve selective release of its
existing drafts beyond the frozen prompt-only gate, without losing useful answers?

| Approach | Trade-off | Decision |
|---|---|---|
| Final non-special response-token state, fixed layer18 | One vector per response; matches the previously specified Task9; easy to bind to a whole-response label | First Gate2 pilot |
| Mean pooling over response-token states | Represents the whole draft more directly, but introduces another feature choice and aggregation contract | Separate later ablation |
| Genuine J++ expert-Jacobian lens | Adds a different concept-readout signal; no released Tiny Aya lens was found and Aya fitting/LRP compatibility are unverified | Separate extension, not part of this pilot |

The layer and estimator are fixed from the prior method specification, not selected
against Gate1 development gains. This is buffered post-generation rejection.

## Data and supervision

Reuse precisely the saved 200-case pilot and its question-group partitions:
120training /40calibration /40development. Source pilot SHA-256:
`c35a68909dc8841ebd47f90fc1f62a926410d251ae3e34ad2a254a0876271081`.

The correctness head is trained only on attempted answerable responses with an
applicable inherited correctness label. Positive means **incorrect**, negative
means **correct**. Naturally abstained drafts and cases without applicable
correctness labels are excluded from this supervision. An attempt on a
should-abstain question is not automatically assigned a false factual label.

The read-only design inventory found:

| Partition | All cases | Eligible responses | Correct | Incorrect |
|---|---:|---:|---:|---:|
| Training |120|52|29|23|
| Calibration |40|18|11|7|
| Development |40|15|14|1|

All eligible rows in each partition have distinct question groups. Sampling stays
unchanged; selecting more errors after seeing these counts would change the study.
One development error permits only weakly powered factuality diagnostics. Existing
development labels/results have already been observed, so this is exploratory.

Split-local manifests and numeric arrays are materialized separately. Training
receives only training inputs; calibration receives only calibration inputs.
Development scoring starts after estimator and threshold artifacts are frozen.

## Feature contract

- Model/revision, tokenizer, native prompt helpers and FP16 extraction runtime are
  the same pinned Tiny Aya configuration as Gate1.
- Reconstruct the exact prompt IDs using the existing system/user prefix. Append
  token IDs obtained by encoding the unchanged cached assistant text with
  `add_special_tokens=False`. Do not rerender a completed assistant turn or add EOS.
- Preserve prompt/response byte hashes, prompt IDs, response IDs, concatenated IDs,
  their lengths/hashes and the final non-special response-token position.
- Check prompt identity against the saved prompt-only preflight. Check response
  encode/decode identity with cleanup disabled. Report original-versus-reconstructed
  token counts; absent original generated token IDs, call this **text-reconstructed
  teacher forcing**, never exact generation-state replay.
- Capture the zero-based decoder-block18 output at the final non-special
  response-text token, before any terminal special token. One FP16 vector is saved
  per feature-bearing response; fitting/scoring uses FP32.
- Empty/all-special responses have no feature and keep their original draft with
  an explicit unavailable-feature status. They cannot be silently dropped from
  policy metrics. A labelled training/calibration row with no valid feature stops
  fitting/calibration for review.
- No text truncation is allowed. A reconstructed sequence beyond8192tokens or the
  model's supported limit stops the proposed batch before dispatch.
- GPU input contains the required native prompt and cached draft, case identity
  and token hashes; it contains no reference answer, correctness/answerability
  label, judge reasoning or split-specific target.
- Save a training-only causal diagnostic: same-length replacement of future
  response IDs must leave captured prompt-boundary states byte-identical under
  otherwise identical execution. Report prompt-only versus teacher-forced prefix
  drift separately; the combined policy always reuses the frozen Gate1 decisions.

## Fitting and policy calibration

Use StandardScaler and logistic regression with `C=0.001`, `max_iter=1000`,
`random_state=20261115`, default class weights and no hyperparameter/layer search.
Fit only the52eligible training responses. Serialize numeric scaler/weights and
class orientation, with exact training-ID and feature hashes.

Fit a response-text control on the same52rows using question + cached answer,
TF-IDF unigrams/bigrams, `min_df=2`, `max_features=5000`, and the same classifier.
The exact input is `question + "\n\nAssistant draft:\n" + cached_response`.

Calibrate standalone Gate2 and its combination with frozen Gate1 separately:

1. Gate2 scores every available baseline attempt; natural baseline refusals pass
   through unchanged. Apply no answerability/correctness-based inference filter.
2. Enumerate unique applicable calibration logits plus keep-all/reject-all.
   Reject when the incorrect-response logit is at or above the threshold.
3. For the combination, Gate1 rejection is final; Gate2 acts only on attempts that
   passed Gate1. Keep Gate1's estimator, threshold and existing decisions fixed.
4. Each complete policy must retain at least `ceil(.9*11)=10` baseline-correct
   answers and `ceil(.5*30)=15` baseline attempts on the40calibration cases.
   These floors refer to the original baseline, not a shrinking subset.
5. Among feasible policies maximize joint task successes; then minimize retained
   incorrect eligible answers; then maximize retained attempts; then choose the
   stricter threshold. A keep-all mode is weaker than every finite threshold.
   Record the full counted frontier and subset denominators.
6. Freeze both selected thresholds before scoring development. Apply this same
   procedure to the response-text control and its Gate1 combination.

## Routing, evaluation and interpretation

Compare six arms on the same40development cases: raw, frozen Gate1, Gate2,
Gate1+Gate2, response-text control and Gate1+response-text control. Reuse existing
raw/Gate1 emissions and metrics by their exact bindings. Each new rejection emits
`I don't know.`; every kept draft retains exact original UTF-8 bytes.

Use the pinned SafeTune LinearProbe and inherited guard routing for the new head,
with a separately named case-bound response-state adapter. Feed actual saved
emissions to the pinned AuditKIT precomputed runner. Use named indicator counters
and independently computed aggregate ratios; generic `prediction.correct` and
saved score directions are not accuracy or metric-direction authority.

Report joint success, correct-answer retention, attempt retention/coverage,
should-abstain attempts, false refusals, and eligible factual error with all counts.
Report head AUROC/AP on applicable development responses only, prominently noting
the14/1class balance. Use the prior fixed-seed2000whole-group paired bootstrap for
policy differences, retaining undefined-draw counts. Do not interpret a single
removed error as a robust factuality result or a positive point estimate as
confirmed superiority. Threshold selection happens only on calibration.

All bootstrap intervals are exploratory, conditional on fitted heads/frozen
policies/inherited labels, and pointwise. They omit fitting/calibration uncertainty,
prior development exposure and multiplicity adjustment. Even an interval excluding
zero is not confirmed superiority. Gate2 machine-readable summaries use neutral
`interval_excludes_zero` language. If raw helper output is retained for provenance,
its legacy `confirmed_increase`/`confirmed_decrease` names are explicitly marked
non-confirmatory and never exposed as scientific conclusions.

## Execution, artifacts and verification

Use a separate `part 1/abstention_pipeline/artifacts/3b-gate2-pilot-20261116/` root
and separately named source/config files. The existing Gate1 configuration,
sources, fitted probe, policies, failures and scientific artifacts remain binding.
All new work stays private, on Tiny Aya, with zero new paid judging or generation.

After Gate1 delivery/grace and design review, the implementation plan defines the
exact source modules and two disjoint account shards, using existing bounded
process/dispatch/collection patterns. Maximum concurrent GPU jobs remains two,
one per account. Do not dispatch until prior account jobs are reconciled; preserve
accepted or possibly accepted work. The original cutoff applies throughout.

Meaningful checks cover response-token boundaries and special tokens, unchanged
prefix/draft bytes, split-local fitting, correctness-label eligibility/orientation,
threshold floors and ties, composite routing, artifact reload parity and exact
AuditKIT denominators. Native causal/feature evidence is checked before fitting.
No scientific stage is repeated to improve a disappointing score.

Completion means a verified exploratory Gate2 report with actual before/after
counts, controls, runtime and limitations. If the bounded work cannot finish,
retain exact partial artifacts and state which stage remains incomplete.

## Design self-review

- Head target, feature position and threshold direction are explicit.
- All policies preserve the original source/splits and Gate1 release behavior.
- Original baseline denominators govern combined retention floors.
- Causal diagnostic uses training inputs; development cannot select features.
- The single-error development limitation and text-reconstruction limitation are
  explicit. No J++ or held-out/full-study claim is implied.
- No Gate2 implementation or execution had occurred at design acceptance. User
  explicitly directed “start gate 2”; the full grace ended07:03:37.547UTC before
  the recorded acceptance. The prior delegation covers concrete implementation
  choices and independent reviews without additional questions.

## Accepted runtime clarification

The parent accepts last-token LM-head slicing (`logits_to_keep=1`) for this new
extraction only, conditional on same-input original/sliced FP16 feature parity on
the longest training sequence assigned to each account. These two training-only
checks also perform the same-length future-token replacement causal diagnostic.
No new policy fit or old extraction is involved. Prompt-only versus full-prefix
drift is descriptive, not a reason to change the frozen Gate1 decisions.

Before dispatch, local tokenization must validate every sequence, including exact
response round-trip with cleanup disabled. Any failure stops with its case ID and
reason. Diagnostic rows are selected deterministically by total token length,
then case ID; labels do not affect selection. Both conditional parity checks must
pass before features can be accepted for fitting. The measured optimization
evidence from Gate1 alone does not establish parity for these new response inputs.
