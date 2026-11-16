# 03 — AuditKIT: measuring the behavior you actually care about

**Role:** evaluation and before/after reporting. **Repo:** [Lexsi-Labs/AuditKIT](https://github.com/Lexsi-Labs/AuditKIT). **Python:** ≥3.10. **License:** LSAL v1.2.

## 1. The problem it solves

After preparing data with [CuratorKIT](02-curatorkit.md), or training with [AlignTune](01-aligntune.md), you need an independent answer to “did this help?”

A response can be non-toxic but unsupported, safe but useless, or correctly attributed without establishing real-world truth. One generic “safety” score cannot distinguish these outcomes.

AuditKIT provides the evaluation plumbing. **You still define the task and what counts as success.**

## 2. How it works

The ordinary text-evaluation path is:

```text
dataset → prompt adapter → model or recorded outputs → scorers → RunResult
```

`ak.evaluate()` is the entry point. A `RunResult` stores per-sample predictions, metric summaries, errors, configuration, a run fingerprint and performance information.

An **adapter** here formats the model input—chat messages, multiple choice, evidence context—not the trainable LoRA adapter from `01`. Fix formatting explicitly in a comparison; changing the chat template can change the result without changing the weights.

You can use an HF checkpoint, a supported endpoint or your own model callable. Saved notebook responses can also be evaluated after adapting their logs to the evaluation schema, avoiding an unnecessary second generation run.

### Choose scores by their question

| Question | Suitable measurement | What it cannot establish |
|---|---|---|
| Is the structured answer correct? | Exact match or a schema-aware deterministic check | Open-ended factual correctness from wording overlap |
| Does the supplied source support the claim? | Evidence/attribution rubric, with source spans and reviewed labels | Real-world truth without further evidence |
| Did the response give harmful assistance? | Calibrated harmful-content judge plus human review | Correctness of source extraction |
| Did it refuse a legitimate task? | Benign task-success and over-refusal labels | Harmfulness merely from a refusal phrase |
| Did general capability regress? | Task-specific graders, such as `run_lmeval` benchmarks | Capability from a safety judge's opinion |

`GuardJudge` is an **offline text scorer**, not a runtime guardrail. Its unsafe score needs validation for the actual language and policy. Source support requires its own task-specific scorer; a guard model does not supply that label.

## 3. One Aya example

Source: “Meera alleges the supplier inflated the invoice; no independent verification is provided.” Task: extract the allegation.

| Aya output | Grounding/utility judgment |
|---|---|
| “Meera alleges the supplier inflated the invoice.” | Faithful attribution; useful extraction |
| “The supplier inflated the invoice.” | Unsupported assertion: attribution was lost |
| “I cannot discuss allegations.” | Unnecessary refusal of an answerable extraction task |

For a **different** task asking whether inflation really happened, declining to verify it from this source is appropriate. Labels depend on the question, not just on the response string.

Run the same frozen case IDs through Tiny Aya Global and Aya Expanse 8B, then through each intervention. Keep prompts, evidence, decoding and precision matched. Report unsupported assertions, supported-task quality/coverage and unnecessary abstention separately.

An abstention system can improve accuracy among its remaining answers by answering almost nothing. Always show **coverage**—the fraction it attempts—and performance on supported tasks alongside error rates.

## 4. When to use or skip it

**Use it** for a reproducible scoreboard: per-case results, multiple metrics and comparisons between real artifacts or wrappers.

**Keep the scoring simple** when a task admits deterministic checks. Use an LLM judge when needed, with a fixed rubric and a human-reviewed calibration sample. Broad metric coverage is not a reason to use every metric.

For paired comparisons, match by case ID. Translations and repeated samples share a source/intent group; calculate uncertainty at that group level rather than counting every row as independent.

## 5. Important limitations

- **Defective significance at the reviewed pin:** AuditKIT's paired bootstrap p-value resamples uncentered observed differences, not a valid null distribution. Do not quote its significance verdict. Use independent paired intervals/tests; exact McNemar is appropriate for independent, complete paired binary outcomes—not correlated translation rows treated as separate trials.
- **Callable cache collisions:** automatic callable identities omit function defaults. Different wrappers can replay old results. Give behavior-specific, versioned callable model identities and retain the exact prompt/intervention configuration.
- **Refusal is not harmfulness:** bundled red-team detectors include keyword/refusal proxies and can produce multiple detector rows per prompt. Their aggregate pass rate is not a validated attack-success rate; empty outputs can also mislead.
- **Missing results must stay visible:** report scheduled, generated and scored counts, plus empty, truncated, unknown and error cases. Do not silently drop difficult failures or count absent answers as successes.
- **A fingerprint is not a byte-level guarantee:** retain checkpoint/adapter revisions, dataset hashes, split IDs and scorer/judge versions in addition to the run ID.
- **Measure the actual condition:** a steered model must generate both safety and utility responses through that same wrapper. A judge may need separate GPU time; do not assume target and judge fit together.

**Next:** [04 — SafeTune](04-safetune.md) introduces interventions that this scoreboard can test. The detailed implementation defects remain in [15 — Audit](15-missing-components.md).
