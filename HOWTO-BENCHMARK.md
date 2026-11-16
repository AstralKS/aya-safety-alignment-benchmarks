# How to benchmark Aya baselines — operations manual

Practical guide to reproducing or extending the Aya v2 baseline runs in this workspace.
Written 2026-11-12 after the four-benchmark pass (SimpleQA Verified, AbstentionBench,
HalluLens PreciseWikiQA, SelfAware), including the failures that cost the most time.

Read this before launching anything that spends money or GPU quota.

---

## 1. Scope and moving parts

| Piece | Where it runs | Cost model |
|---|---|---|
| Aya answer generation (chat) | Kaggle GPU kernels, private | free GPU quota (T4), time only |
| SelfAware embeddings (SimCSE) | Kaggle GPU kernel, private | free GPU quota, time only |
| Factual judging (GLM-5.3-Flash) | local process → HF router / DeepInfra | **real USD** from a shared $18 ceiling |
| Scoring/report assembly | local CPU | free, but I/O heavy |

Judge identity (fixed, do not substitute): `zai-org/GLM-5.3-Flash:deepinfra` via
`https://router.huggingface.co/v1/chat/completions`, $0.15/M input, $0.50/M output,
mandatory thinking, `reasoning_effort="low"`, `max_tokens=2048`, no implicit retries.

Two Kaggle accounts are available. That is the single biggest lever for the GPU side
(see §4.2). It does **nothing** for GLM grading, which is a separate, local, paid path.

---

## 2. Architecture in one picture

```
data/v2/manifest.json ──┐
data/v2/generation.jsonl├─► Kaggle generation shards (per model) ─► runs/v2/<model>/responses.jsonl
data/v2/evaluation.jsonl┘                                        └─► runs/v2/<model>/jobs/<job>/output/
                                     │
                                     ▼
                     scoring/v2_glm_scoring.py  (GLM factual labels)
                        requests/ + request-journals/ + budget.json
                                     │
                                     ▼
                 scoring/v2-glm/results/<run>/snapshots/<hash>.json   (immutable packet)
                                     │
     runs/v2/compute/selfaware-*/output/.../grades.jsonl ─┤
                                     ▼
                  scripts/v2_benchmark_report.py  (re-verifies every journal)
                                     │
                                     ▼
                    benchmark.md + scoring/v2-glm/benchmark-results.json
```

Everything is **append-only or write-once**. Nothing in the pipeline rewrites a journal,
a snapshot, a collected receipt, or a published report.

---

## 3. Invariants — break these and the results are void

1. **One label = one journal = one ledger attempt.** No replays, no implicit retries.
2. **Every published number is re-derivable** from a raw HTTP journal whose SHA-256 is
   recorded in the snapshot and re-checked at report time (`journal_label()`).
3. **Never grade a partial row.** A benchmark is reported only at its full denominator
   (`COUNTS` in `v2_benchmark_report.py`).
4. **Never fabricate or interpolate** a missing label; `halted=True` snapshots are unusable.
5. **Preserve failures.** A failed run stays on disk with its receipt; recovery is additive.
6. **Concurrency is a property of the writer, not of the job.** See §5.3.

---

## 4. Stage 1 — generation on Kaggle GPU

### 4.1 Sharding
- `data/v2/generation.jsonl` is the full prompt inventory per model (e.g. 7,549 rows:
  3,369 SelfAware + 1,000 PreciseWikiQA + 3,000 NonExistentRefusal + 180 FActScore bios).
- Split into ~750-row shards; one shard = one kernel = one `job.json` +
  `kernel-metadata.json` + `run.ipynb` under `runs/v2/<model>/jobs/<job-name>/`.
- Job names are deterministic (`8b-precisewikiqa-000-e31d49c2`), so a re-run can prove
  it is not creating a duplicate.

### 4.2 Using two accounts (explicit parallelisation)
Kaggle allows **one concurrent GPU session per account**. With two accounts you get two.

Rules that keep this safe:
1. Round-robin shards across accounts: account A takes even indices, account B odd.
2. **Never submit the same job twice**, on either account. `runs/v2/generation-progress.json`
   is the single source of truth for job state; check it before every submit.
3. Use per-account credentials explicitly, do not rely on an ambient login:
   `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables set per subprocess, or
   `~/.kaggle/kaggle.json` swapped by an explicit wrapper.
4. Record the owning account in each job dir (add `"account": "A"|"B"` to `job.json`).
   Today the job dirs do not record it — add it before the next generation wave.
5. Assume 30 GPU-hours/week/account. Track hours consumed per account; do not let one
   account starve the other.
6. Collection is per-account too: `kaggle kernels output` must run with the credentials of
   the account that owns the kernel.

Expected effect for a 24-shard, ~2 h/shard wave:
- 1 account → ~48 h wall clock.
- 2 accounts → ~24 h wall clock.
Throughput is bounded by shards/account, not by anything local.

### 4.3 Kaggle transport traps (all hit in practice)
| Trap | Symptom | Fix |
|---|---|---|
| Notebook body size | `SaveKernel` HTTP 400 above ~1 MB (0.93 MB ok, 1.4 MB fails) | ship inputs through the private HF dataset `ASTRALK/aya-baselines-20261111`; kernel re-verifies SHA-256 before use |
| Title/slug mismatch | `SaveKernel` 409 Conflict | `kernel-metadata.json` `title` must be the slug of `id`; Kaggle slugifies `_`→`-` |
| Status lookup | "Kernel not found" | poll with the hyphen slug (`launch_v2.kernel_status` resolves and caches this) |
| No cancel in CLI | `kaggle` 2.2.4 has no `kernels cancel` (list/status/logs/output/delete only) | let a discarded shard finish, or delete the kernel (unverified as a cancel); never rely on cancelling to save quota |
| Secret leakage | token inside notebook source | inject at push time only (`v2_compute_jobs.submit` rewrites the cell), never commit it |

### 4.4 In-kernel throughput
- `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, FP16, greedy, one pass, pinned
  native chat template, no reflection prompt.
- Batch generation; record `finish_reason` per row (`eos` vs `length`) so truncation is
  disclosed rather than hidden.
- Write output incrementally (`responses.jsonl` + `progress.json`) so a killed kernel
  still leaves usable rows; collection is gated on a `collected.json` receipt.

### 4.5 Small models: serve, don't loop (vLLM / TGI)
For Tiny Aya the current shard runner does HF `generate()` in batches, one shard per
kernel. That leaves the GPU idle between batches and re-processes the identical system
prompt on every row. For the next wave:

1. **One engine per GPU.** Use vLLM (`AsyncLLMEngine`) or TGI with continuous batching
   inside each kernel: `max_num_seqs=64`, `gpu_memory_utilization=0.90`,
   `dtype=float16`, `enforce_eager=True` (T4 has no benefit from CUDA graphs here and
   compile time is wasted), `max_model_len` set to the longest prompt in the shard.
2. **Several replicas = several GPUs, not several processes on one GPU.**
   If the kernel shape exposes 2×T4, run one replica per GPU (`CUDA_VISIBLE_DEVICES=0|1`)
   and split the shard's rows across them. On a single GPU, replicas only fight for VRAM;
   continuous batching already gives the throughput.
3. **Two accounts = two models in parallel.** Put tiny on account A and 8b on account B
   instead of serialising them; each account keeps one GPU session, so total wall clock
   halves without violating any Kaggle rule.
4. **Prefix caching** for the shared system prompt (`enable_prefix_caching=True`) — on a
   7,549-row shard this removes a large fraction of prefill work.
5. **Determinism budget.** vLLM/TGI must be pinned by version and revision, FP16, greedy,
   same chat template hash. Before trusting a migrated engine, replay 20 saved prompts and
   require identical normalised text (or an explicitly disclosed difference) against the
   HF outputs; record `engine`, `engine_version`, `max_num_seqs` in `job.json`.

Expected effect: continuous batching plus prefix caching is typically **5–20×** the naive
per-batch HF loop for short chat answers; the exact factor must be measured on 100 rows
before scaling, and disclosed in the generation contract rather than assumed.

### 4.6 Multi-shard supervision
- One supervisor process owns the queue and the two account credentials; workers never
  submit. This is what makes "never duplicate a job" enforceable.
- On a worker failure, requeue only if `collected.json` is absent and the kernel is
  confirmed absent/failed — never re-push a kernel whose output might exist.
- Track per-account running kernels and GPU-hours in the progress file; a second kernel
  pushed to the same account just queues behind the first and looks "slow".


---

## 5. Stage 2 — GLM grading (the real bottleneck)

### 5.1 Measured facts (use these for planning)
| Metric | Value |
|---|---|
| Throughput at `--concurrency 1` | **~15 labels/min** (~4 s/label) |
| Cost per PreciseWikiQA label | **$0.000136** (~890 prompt+completion tokens) |
| Cost per label, longer prompts (SimpleQA) | ~$0.00018–0.0002 |
| Replay cost of an already-graded label | **~3 s** (journal decode + strict re-parse) |
| 2,000 labels, serial | ~2.2 h |
| 1,899 cached labels replayed | ~87 min (pure waste) |
| Observed failure at concurrency 8 | `Lock busy/stuck` on `scoring/budget.json` |

### 5.2 Why it is serial today
`JobStore.evaluate()` runs a `ThreadPoolExecutor(max_workers=concurrency)`, but every
attempt takes the **same global file lock** to read-modify-write `scoring/budget.json`
(which then contained thousands of attempt records) plus fsync. At concurrency 8 the
lock waits exceeded the timeout and the run halted on a *local* error — not a model or
network error. The pinned workaround is `--concurrency 1`.

### 5.3 How to parallelise it properly (blueprint)

Two layers: **structural** (remove the shared write hotspot) and **adaptive** (run 8, drop
to 4, then 2, then 1 when the system complains — with jitter — instead of hard-coding 1).

#### 5.3a Structural fixes (do these first)
1. **Append-only ledger.** Replace the single `budget.json` map with
   `scoring/ledger/attempts/<attempt_id>.json` written once (immutable) + a periodic
   compactor that produces the aggregate `budget.json` view. Workers never contend;
   only the compactor and the reporting reader need a lock.
   *Files:* `v2_glm_budget.py` (`Ledger`, `file_lock`), `v2_glm_scoring.JobStore`.
2. **Per-attempt reservation, single-writer settle.** Reserve with an atomic
   `mkdir scoring/ledger/reservations/<attempt_id>` (no JSON rewrite). A single writer
   thread drains settled attempts and computes totals. The ceiling check becomes
   `reserved_so_far + max_reserve_per_call <= 18` at reservation time.
   Today each attempt rewrites the whole ledger under one global lock — that hold time is
   what made 8 workers queue past the 10 s lock timeout.
3. **Shard by cache key, not by task.** `hash(cache_key) % N == worker_index` gives N
   independent worker processes with disjoint journal directories, so even the existing
   per-journal locks never cross. Run N local processes; each keeps its own run lock
   (`run-<i>.lock`).
4. **Verified-cache index.** Maintain `cache_key → {journal_sha256, label}` in a small
   append-only index. On replay, trust the index and skip re-decoding the journal body
   (keep a periodic full audit). This removes the 3 s/replay tax that made a full
   re-score cost 7 h.

#### 5.3b Adaptive concurrency governor (safe to add before the structural work)
The HTTP side is already proven: the continuation pilot passed at concurrency **1, 2, 4
and 8**. The failures were local lock holds, never the provider. So hard-coding 1 is
leaving throughput on the table; the right answer is a governor that backs off instead of
aborting.

```
limit = 8  (or 4 on small machines)
window = last 60 s of outcomes
loop:
    dispatch up to (limit - inflight) jobs from the queue
    await completions
    classify failures (table below)
    if local_lock_failures >= 2 in window:      # contention, not the provider
        limit = max(1, limit // 2)              # 8 -> 4 -> 2 -> 1
        sleep(backoff(limit) + jitter(0..1s))   # 1s, 2s, 4s
        requeue those jobs (they never dispatched, so this is free)
    elif successes_since_change >= 50 and limit < max_limit:
        limit += 1                              # climb back up, one step at a time
    elif provider_429_or_5xx:                   # the provider, not us
        limit = max(1, limit // 2)
        sleep(Retry-After or 2^attempt * jitter, cap 60s)
        requeue only if the attempt was never dispatched
```

**Error taxonomy — what may be retried without risking money**

| Failure | Dispatched? | Reservation? | Retry? |
|---|---|---|---|
| `Lock busy/stuck` (local journal/ledger lock) | no | no | **yes, free**, after lock check/clear |
| Local parse/IO error before HTTP write | no | no | yes |
| Provider 429 / 5xx / connection reset after request sent | maybe | yes | **no automatic retry**; mark `unknown`, reconcile from journal |
| HTTP 200 with invalid label | yes | yes | only via explicit `--retry-invalid <cache_key>` |
| Timeout of unknown state | maybe | yes | **never**; treat as `unknown` reserve |

The rule that keeps the ledger honest: **retry the local failure, never the ambiguous
call.** A governor that "just retries everything" would double-spend on exactly the cases
the journal/ledger design exists to prevent.

Operational guard: the governor must not resurrect a lock held by a live process. It
retries with backoff; if the lock persists beyond `3 × timeout`, it halts with the same
actionable error and `scripts/v2_clear_stale_locks.py inspect` decides whether the lock is
stale.

Expected throughput with (1)+(2)+(3) plus the governor, same 4 s API latency:

| Concurrency | Labels/min | 2,000 labels |
|---|---|---|
| 1 (today) | 15 | ~2.2 h |
| 4 | ~55 | ~36 min |
| 8 | ~105 | ~19 min |
| 16 | ~200 (provider permitting) | ~10 min |

Measure the provider's real ceiling on 50 rows before committing to >8; DeepInfra
throttles before the local writer does.


### 5.4 Scoping rule that saves hours
Only ever grade **missing** cache keys:
```powershell
python -X utf8 -B scripts/v2_glm_scoring.py score --run-id <run> `
  --models 8b --tasks hallulens_precisewikiqa --concurrency 1 --execute
```
Grading all tasks "to be safe" re-verifies thousands of cached jobs at ~3 s each. A
scoped run that costs 2 h instead of 7 h is not a micro-optimisation, it is the difference
between one and three working sessions.

### 5.5 TC/DR of the halt path
- Empty directory locks (`scoring/budget.json.lock`,
  `scoring/v2-glm/results/<run>/run.lock`, `scoring/v2-glm/request-journals/<key>.lock`)
  are left behind by any killed scorer and will halt the next run with
  `Lock busy/stuck`.
- Recovery: `python -X utf8 -B scripts/v2_clear_stale_locks.py inspect` → confirm
  `busy_baseline: []` → `... clear --confirm` (writes a receipt under `recovery/`).
- A halt is **not** a spent dollar: the ledger keeps `unknown` reserves separate
  (`unknown_reserves_usd`), and a pre-dispatch failure has no reservation at all.

---

## 6. Stage 3 — assembling the report

- Snapshots are immutable: `scoring/v2-glm/results/<run>/snapshots/<hash>.json`.
  A new run always writes a new file; `halted=True` snapshots are rejected.
- **Merging is allowed only under these rules** (implemented in `build_scoped`):
  every packet must be `halted=False`, share the same `v2_inputs` binding, contain no
  duplicate identity, and every label is re-verified against its raw journal by
  `journal_label()`. Report merged provenance by listing each snapshot path + SHA-256.
- Scoped reports (subset of benchmarks):
  ```powershell
  python -X utf8 -B scripts/v2_benchmark_report.py `
    --fixed <snapA.json> <snapB.json> `
    --selfaware runs/v2/compute/selfaware-*/output/selfaware-v2/grades.jsonl `
    --tasks simpleqa_verified abstention_bench hallulens_precisewikiqa --write
  ```
- The full six-benchmark path (`build`) still exists and is unchanged; it requires the
  FActScore extraction/retrieval artifacts as well.
- Publication is idempotent: `publish()` refuses to overwrite a report that does not match
  the verified result, so re-running is safe.

### 6.1 Metric semantics you must not get wrong
| Task | GLM stages per row | Aggregation |
|---|---|---|
| `simpleqa_verified` | 1 (`simpleqa`, enum A/B/C) | yield / coverage / accuracy-when-attempted |
| `abstention_bench` | 1 (`abstention`) + conditional `correctness` only when `should_abstain=False` and the model answered | joint success = correct or appropriately abstained |
| `hallulens_precisewikiqa` | 2 for **every** row (`abstention` + `correctness`) | INCORRECT and UNVERIFIABLE both count as hallucination |
| `hallulens_nonexistentrefusal` | 1 (`does_believe`) | false acceptance |
| `selfaware` | none (phrase rules + SimCSE >0.75) | unknown-label F1 + answerable reference accuracy |

SelfAware never touches GLM. Its only compute is a pinned
`princeton-nlp/sup-simcse-roberta-large` embedding job on Kaggle.

---

## 7. Cost accounting — two numbers, do not mix them

| Field | Meaning |
|---|---|
| `reported_cost_usd` | legacy actually-reported spend (~$2.30) |
| v2 `token_priced_usd` | GLM token cost of this campaign |
| **actual spent** | sum of the two → the number the user cares about (~$3.75) |
| `reserved_upper_usd` | actual spent **+ worst-case legacy reservations** (~$3.57 buffer) |
| `unknown_reserves_usd` | money possibly spent on ambiguous attempts; must be 0 before publishing |

Rule: report **actual spent** to humans, and keep `reserved_upper` as the safety gate for
new spend (`actual + reserve <= 18`). Never present `reserved_upper` as spend.

Measured end-to-end: full four-benchmark campaign ≈ **$3.8** including legacy.

---

## 8. Failure catalogue (fast lookup)

| Symptom | Cause | Action |
|---|---|---|
| `Lock busy/stuck ...` | stale empty lock dir from a killed process | `v2_clear_stale_locks.py clear --confirm` |
| `halted: true`, `state: unknown` | reserved attempt without raw response | do **not** replay; investigate journal/ledger pair |
| `Missing fixed result: (...)` | scoped report fed a packet lacking a task | scope the scorer first, then the report |
| `Extra or ineligible fixed-benchmark judgments` | packet contains tasks outside `--tasks` | restrict the packet or widen `--tasks` |
| `SelfAware result lacks a collected remote output receipt` | grades file outside `runs/v2/compute/<job>/output/` | collect through `collect_compute` |
| Kernel 409 on push | metadata title ≠ slug | fix title |
| Kernel body 400 | notebook > ~1 MB | move payload to HF dataset |
| Report refuses to overwrite | different existing results | never force; investigate which run is authoritative |

---

## 9. Runbook (copy-paste)

```powershell
# 0. preflight
python -X utf8 -B scripts/v2_clear_stale_locks.py inspect        # busy_baseline must be []
python -X utf8 -B -c "import json;print(json.load(open('scoring/budget.json'))['reserved_upper_usd'])"

# 1. scoped grading (only missing keys)
python -X utf8 -B scripts/v2_glm_scoring.py score --run-id baseline-v2-full `
  --models 8b --tasks hallulens_precisewikiqa --concurrency 1 --execute

# 2. state check (never poll in a tight loop)
python -X utf8 -B -c "import json;l=json.load(open('scoring/budget.json'));print(len(l['v2_glm']['attempts']), l['settled_requests'])"

# 3. verify the new snapshot then publish the scoped report
python -X utf8 -B scripts/v2_benchmark_report.py `
  --fixed scoring/v2-glm/results/baseline-v2-full/snapshots/<old>.json `
          scoring/v2-glm/results/baseline-v2-full/snapshots/<new>.json `
  --selfaware runs/v2/compute/selfaware-5426dad9c4a4/output/selfaware-v2/grades.jsonl `
  --tasks simpleqa_verified abstention_bench hallulens_precisewikiqa --write

# 4. tests before any change to reviewed modules
python -X utf8 -B -m unittest discover -s scripts -t scripts -p "test_v2*.py"
```

---

## 10. Checklist before spending again

- [ ] `v2_clear_stale_locks.py inspect` shows no busy baseline and no non-empty lock
- [ ] `unknown_reserves_usd == 0`
- [ ] `actual_spent + worst_case_next_run <= 18`
- [ ] The run is scoped to missing cache keys only (`--models`, `--tasks`)
- [ ] Concurrency is 1 unless the ledger rewrite (§5.3) has landed and been reviewed
- [ ] Kaggle jobs are not duplicated across the two accounts; `generation-progress.json` agrees
- [ ] A single 25-minute reminder is armed; no other scorer runs concurrently
- [ ] After completion: snapshot `halted=False`, zero errors, counts match `COUNTS`
- [ ] Report re-verifies every journal; published files match the verified result exactly

---

## 11. What I would change first, in order

1. **Ledger concurrency** (§5.3 items 1–2): unlocks 8× on the paid path and removes the
   `Lock busy/stuck` class of failure entirely.
2. **Verified-cache index** (§5.3 item 4): removes the ~3 s/cached-label replay tax, which
   is the difference between a 30-minute report and a 7-hour one.
3. **Account tagging in job metadata** (§4.2 rule 4): makes the two-account split auditable.
4. **Automatic stale-lock sweep at startup** with the same receipt format, so a killed run
   never costs another hour.
5. **Batch ledger flushes every N attempts** instead of per attempt (fewer fsyncs, same
   guarantees, since attempts are immutable and the aggregate is derived).

Everything else in the pipeline is already correct-by-construction and should not be
"improved" casually — the immutability and journal re-verification are what make the
published numbers defensible.

---

## 12. When to stop, and how to stop safely

Do not grind on a broken run. Stop, save state, report. A half-finished benchmark with an
intact ledger is worth more than a "fixed" one with an unverifiable number.

### 12.1 Abort triggers (any one is enough)

| Trigger | Threshold |
|---|---|
| Same failure class after the documented mitigation | 3 consecutive halts |
| Lock cannot be cleared | a live baseline python process still holds it |
| `unknown_reserves_usd > 0` | any non-zero value — reconcile before spending again |
| Projected overspend | `actual + reserve` would exceed $18 |
| Provider outage | repeated 429/5xx for > 15 min at concurrency 1 |
| Progress stall | attempt count unchanged for > 15 min while `run.lock` exists |

### 12.2 Safe-stop procedure

1. **Stop the supervisor/queue first, not individual workers.** A worker killed between
   reserve and settle leaves a lock; a queued worker leaves nothing.
2. Prefer letting the in-flight attempt settle (≤ 1 call, ~4 s) before terminating.
3. Record a receipt at `scoring/v2-glm/results/<run>/recovery/stop-<UTC>.json`:
   attempt counts by state, `actual spent`, `reserved_upper`, `unknown_reserves`, the newest
   snapshot path + SHA-256, the halt error text, and the list of remaining cache keys
   (or a count per task/model if the list is large).
4. **Preserve everything**: journals, snapshots, partial `output/` dirs, `progress.json`,
   ledgers, failed receipts. No deletions, no rollbacks, no "clean" re-runs.
5. Leave empty locks in place unless `v2_clear_stale_locks.py inspect` shows
   `busy_baseline: []`; then clear with `--confirm` so the removal is receipted.
6. Report: what is complete, what is missing (exact counts), what was spent, what blocked
   it, and the single next command that would resume safely.

### 12.3 Resuming after a stop

- The resume is always a *scoped* run over missing cache keys (§5.4); cached labels are
  reused by exact cache identity and cost nothing.
- Never resume by re-running the full task list "to be safe" — that is the 3 s/replay tax
  that turns a 30-minute resume into a 7-hour one.
- If the stop happened with `unknown_reserves_usd > 0`, reconcile those attempts against
  their journals first and write the reconciliation receipt before dispatching anything.

