# Aya baseline benchmarks

**Complete:** four benchmarks, 6,347 evaluated rows per model (4,369 new answers + 1,978 reused).

| Benchmark | Metric | Tiny Aya Global | Aya Expanse 8B | Rows/model |
|---|---|---:|---:|---:|
| SimpleQA Verified | Correct-answer yield ↑ | 3.70% | 5.70% | 1,000 |
| SimpleQA Verified | Answer coverage | 95.20% | 95.30% | 1,000 |
| SimpleQA Verified | Accuracy when attempted ↑ | 3.89% | 5.98% | 1,000 |
| HalluLens PreciseWikiQA | Hallucination when not abstained ↓ | 91.94% | 84.64% | 1,000 |
| HalluLens PreciseWikiQA | False refusal ↓ | 3.20% | 4.30% | 1,000 |
| HalluLens PreciseWikiQA | Correct-answer rate ↑ | 7.80% | 14.70% | 1,000 |
| AbstentionBench | Behavior accuracy ↑ | 68.30% | 72.70% | 978 |
| AbstentionBench | Joint task success ↑ | 51.23% | 60.22% | 978 |
| SelfAware | Unknown-label F1 ↑ | 34.07% | 43.70% | 3,369 |
| SelfAware | Answerable reference accuracy ↑ | 32.04% | 39.01% | 2,338 |

## Denominators and counts

| Count | Tiny | 8B |
|---|---:|---:|
| SimpleQA correct / incorrect / not attempted | 37 / 915 / 48 | 57 / 896 / 47 |
| PreciseWikiQA non-abstained / hallucinated | 968 / 890 | 957 / 810 |
| AbstentionBench behavior successes / joint successes | 668 / 501 | 711 / 589 |
| SelfAware TP / FP / FN | 238 / 128 / 793 | 328 / 142 / 703 |

All required answers and judgments reconciled; no missing or failed judgments are silently graded.

## Methods and limitations

- GLM-5.3-Flash via HF/DeepInfra is the sole factual judge. Greedy judge decoding, low reasoning effort, 2,048-token limit; strict final-label parsing.
- Aya uses pinned native chat templates, FP16, greedy one-pass decoding, no reflection instruction. All model inference and SelfAware embedding computation ran remotely.
- These are disclosed local variants, not leaderboard reproductions. No new human or assistant factual adjudication; published judge-validation numbers do not transfer to GLM.
- HalluLens uses pinned third-party, sample-validated data. Keys are not individually human-verified; screened entities are presumed nonexistent. PreciseWikiQA repairs only the contradictory final Yes/No correctness instruction; incorrect and unverifiable both count as hallucination among non-abstained answers.
- AbstentionBench retains the balanced 489 answerable / 489 should-abstain subset. Correctness is judged only for answerable, non-abstained responses; joint success requires correct answers or appropriate abstention.
- SelfAware retains all 3,369 released rows, including the ID3149 flag discrepancy (2,338 answerable / 1,031 unanswerable). Official phrase-first rules, five-word windows, pinned SimCSE FP32 pooler output, max length 128, strict cosine >0.75; answerable accuracy uses the official case-insensitive reference substring rule.
- Scope: this report covers four benchmarks only. HalluLens NonExistentRefusal and FActScore biographies were dropped after partial generation/evidence work; their preserved artifacts are neither scored nor reported here. TruthfulQA was dropped earlier; LongFact++ remains probe-training data. F/C remains deferred.

## Cost and provenance

- GLM token-priced estimate: **$1.48640490**; provider estimates: **$1.170480420000000052057** (not invoices).
- Actual GLM spend: **$3.889081230 of $18**, leaving **$14.110918770** unused. That is legacy reported cost $2.303337630 plus v2 token-priced cost $1.585743600 across 9,494 settled judge calls. It includes the 691 judgments paid for during the abandoned NonExistentRefusal attempt, which were never used. Provider self-estimates sum to $3.573156750; they are not invoices.
- GLM attempts: 9,496 reserved, 9,494 settled; the two unsettled attempts carry at most a fraction of a cent each at observed rates. Reservations are internal safety machinery, not spend: each call reserved $0.1583104 (a full 1M-token input at list rates) against an observed average of roughly $0.000167. No further GLM charge will be made. Remote Kaggle compute is separate quota, not a paid GLM charge.
- Machine-readable metrics and full artifact hashes: `scoring/v2-glm/benchmark-results.json`.

- `scoring/v2-glm/results/baseline-v2-full/snapshots/8f05c2e977d847448d4cc4b9d31647cd.json` — SHA-256 `92f56240a946748bb8f521caec44a95b50ceaaa01fa192441fae0847687756c9`
- `scoring/v2-glm/results/baseline-v2-full/snapshots/7e401c4b4392458a8493a5d3b28d6581.json` — SHA-256 `c263f5bdc7471109c773c4de302a9f002dfb3c9af8403c2d07269b61225a0df9`
- `scoring/v2-glm/results/baseline-v2-full/snapshots/2d6a928ac7a94d19831c87be1c647e73.json` — SHA-256 `bdbc4d9299a868a7e32d4a20395e6718edba4dcd351fc854e609da23a91b55d9`
- `runs/v2/compute/selfaware-5426dad9c4a4/output/selfaware-v2/grades.jsonl` — SHA-256 `50c65780a13cee692a5712bf2de53c551dd48deb6f169dcdc4ecc6105b30b3ee`
- `runs/v2/compute/selfaware-5426dad9c4a4/collected.json` — SHA-256 `305ede842b43db0be6c4f041c82b1e3a2dd743b40b12bc2610cb6be0ccdc8538`
- `runs/v2/tiny/responses.jsonl` — SHA-256 `aa7a20232e7652d857433cc93efaf4dcfbc9f212ca28bb3ab6c6775444d9077f`
- `runs/v2/8b/responses.jsonl` — SHA-256 `10cec296f90e9f77be709a7c7643532acf5750786b16047b2e4d9ba1c22175a8`

### Generation truncations (retained, not regenerated)

| Model | EOS | Token limit |
|---|---:|---:|
| tiny | 2589 | 1780 |
| 8b | 1979 | 2390 |
