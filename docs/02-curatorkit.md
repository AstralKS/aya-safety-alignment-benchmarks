# 02 — CuratorKIT: building data you can trust and trace

**Role:** data preparation, synthetic generation and quality checks. **Repo:** [Lexsi-Labs/CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT). **Python:** ≥3.11. **License:** source-available LSAL v1.1; review its noncommercial terms.

## 1. The problem it solves

[AlignTune](01-aligntune.md) learns from the data you give it. If demonstrations invent facts, lose qualifiers or contain duplicates across train and test, successful training can still produce a misleading result.

CuratorKIT turns documents or existing rows into a cleaner, traceable dataset. It can **generate** examples, **filter** them against explicit criteria and **export** them for training. These are different operations: formatting a bad answer as chat messages does not make it good.

It is also useful before training—for constructing a labeled evaluation corpus—provided the frozen evaluation examples are kept separate from any repair or regeneration loop.

## 2. How it works

The main entry point is `Curator(CuratorConfig(...)).run()`. Think of the pipeline as:

```text
read source → optionally generate examples → check/filter → clean/deduplicate → export
```

- A **reader** turns files or dataset rows into samples and retains source information.
- A **generator** writes questions, answers or preference pairs. Its outputs are candidates, not labels you can trust automatically.
- A **gate** accepts or rejects a sample against a rule. Schema gates check required fields; grounding and quality gates can ask an LLM judge.
- A **normalizer** transforms text. Deduplication removes repeats; hygiene can detect secrets or pseudonymize personal information.
- An **exporter** writes a format suitable for a particular consumer.

The generator and judge are separate roles. Using different or larger models does not eliminate shared errors; check judge agreement against human-reviewed examples in the languages you use.

### Grounded does not mean true

The hallucination gate compares an answer with its **source chunk**, rather than treating the judge's general knowledge as evidence. An accurately attributed false allegation can pass a grounding check: the source really contains that allegation. Establishing real-world truth is another task.

### Filtering can also change the task

A toxicity filter can accidentally delete legitimate analysis of harmful text. Unicode normalization can erase a script distinction; deduplication can remove intended matched variants. For a code-mixing experiment, preserve the language treatment and review what each transformation removes.

## 3. One Aya example

Continue the invoice example from `01`. Create a source group with:

| Field | Example content |
|---|---|
| Source | Meera alleges invoice inflation; no independent verification |
| Extraction task | Who alleged what? |
| Desired answer | Meera alleges that the supplier inflated the invoice |
| Unsupported answer | The supplier inflated the invoice |
| Evidence task | Can this source establish that inflation actually occurred? |
| Desired answer | No; it records an allegation, not independent verification |

Generate reviewed English, Hindi and code-mixed variants if that is the experiment. Keep the source ID, task type, expected evidence span, language/script and parent IDs. **Provenance** means this record of where an example came from and how it changed.

Export SFT demonstrations to `sft_messages.jsonl`. For DPO, first construct genuine nonempty **chosen/rejected** pairs; merely requesting `export_formats=["dpo"]` does not invent them. GRPO needs prompt-shaped data and suitable scoring, not an SFT answer renamed as a reward.

Inspect the full funnel: input rows, gate rejections, dedup removals, accepted samples and **rows written to each export**. These counts need not match. Keep `manifest.json`, `rejected.jsonl`, checksums and provenance alongside the data.

## 4. When to use or skip it

**Use it** when data generation, grounding, filtering or traceable preparation is a meaningful part of the experiment. It naturally feeds the training stage explained in `01`.

**Keep it minimal or skip it** for a small already-validated corpus. Core cleaning can run on CPU; generation and judge calls add API-token or GPU costs. Adding every gate is not automatically better curation.

Adaptive recovery can diagnose rejected candidates and regenerate them. That is useful for **training-data yield**, but should not rewrite a frozen test set until it becomes easier.

## 5. Important limitations

- **Missing context can pass:** the reviewed hallucination gate has `skip_if_no_context=True` by default. Attach retrievable source context or configure that gate to reject context-less samples. A grounding threshold alone is insufficient.
- **Judge failure is not poor content:** judge-call or malformed-JSON errors normally reject samples. Count them separately; `judge_on_error="pass"` is a fail-open policy, not a quality fix.
- **Built-in splits are row-based:** `output_split` uses a seeded row shuffle, not document/intent isolation. Assign train, calibration/validation and test groups **before synthesis**; keep translations and paraphrases of one source together.
- **Filtering can bias labels:** inspect class and language balance after every gate. Removing difficult “unknown” examples can make abstention evaluation artificially easy.
- **Exports need adapters:** accepted task types differ by exporter. Inspect output warnings and counts, then validate the receiving trainer's schema.
- **Metadata is not verified lineage:** preserve exact source/model revisions and artifact hashes yourself. A provenance file does not prove that the recorded bytes or splits were actually used.

**Next:** [03 — AuditKIT](03-auditkit.md) explains how to measure whether a model uses this evidence correctly. [15 — Audit](15-missing-components.md) retains the detailed gate/export findings.
