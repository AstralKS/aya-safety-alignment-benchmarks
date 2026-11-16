# 04 — SafeTune: changing safety behavior without confusing the methods

**Role:** safety-specific training, weight repair and inference interventions. **Repo:** [Lexsi-Labs/SafeTune](https://github.com/Lexsi-Labs/SafeTune). **Python:** ≥3.12. **License:** LSAL v1.2.

## 1. The problem it solves

Fine-tuning for a useful task can also alter unrelated behavior—for example, weakening harmful-request refusal. This change is called **safety drift**. A model can also be too cautious, refusing legitimate analysis.

[AuditKIT](03-auditkit.md) measures those failures. SafeTune offers methods to change the model or its inference process and test the safety–utility trade-off. It is a **library of alternatives**, not an automatic pipeline that applies every method.

The names describe *when and where* you intervene, not guarantees that the model becomes safe.

## 2. How it works

| Family | When it acts | What changes | Example purpose |
|---|---|---|---|
| **Hardening** | During fine-tuning | The training updates | Learn the new task while constraining safety drift |
| **Recovery** | After a checkpoint has drifted | Stored model weights | Restore useful safety-related changes from compatible references |
| **Steering** | During generation | Internal activations or the decoding process | Nudge a frozen model toward a target behavior |

**Hardening** replaces or modifies an ordinary training step; it need not follow a separate AlignTune run. Methods such as SafeGrad use safety information to adjust gradients—the updates training would apply—and can require an additional frozen reference model.

**Recovery** includes weight arithmetic. A reference comparison estimates changes associated with alignment and applies a selected correction to a drifted checkpoint. ReSta is one implementation; its drifted/base/aligned model objects must be compatible. It does not obtain evidence about a claim merely by restoring weights.

**Steering** can leave stored weights untouched. An **activation** is a numerical internal state computed for an input. A hook intercepts such a state and changes it while the model runs. Other steering methods modify token selection or use additional models instead.

Unlearning is a fourth family: training or editing to suppress a specified capability while retaining others. It is not automatically a cure for hallucination, and weight unlearning is different from deleting a database record.

### What a “refusal direction” actually means

At a chosen layer and token position, compare activations from harmful and harmless calibration prompts. A difference between their average vectors is a candidate direction `d`. A simple steering intervention adds it to the current hidden state:

```text
new hidden state = original hidden state + strength × direction
```

SafeTune exposes `extract_refusal_direction` and `RefusalDirectionModel(mode="steer")`, plus a higher-level refusal-direction trainer. Extraction, layer selection and strength calibration require real model runs.

The direction is a **hypothesis about a useful intervention**, not a semantic label stored in one neuron. It may encode topic, language or formatting differences. Adding more of it can increase refusal on both harmful and benign prompts.

## 3. One Aya example

Suppose validated Hindi/Hinglish requests receive more harmful assistance than intent-matched English requests. Also include benign tasks asking Aya to summarize or attribute sensitive content, including the invoice allegation from `01`–`03`.

Test one bounded steering experiment:

1. Generate unsteered responses under neutral and strong-system-prompt baselines.
2. Extract a candidate direction from separate, reviewed calibration groups.
3. Select layer and strength on development data, including benign controls.
4. Lock the intervention and run held-out harmful **and benign** tasks through the same live wrapper.
5. Compare harmful assistance, benign over-refusal, task quality and language fidelity.

“Fewer harmful answers, but all allegation extraction is refused” is a trade-off, not selective repair. A **Pareto plot** shows safety versus utility across settings; choose a development-set operating point, not whichever test result looks best.

For claim-grounded abstention, a refusal direction is only a candidate baseline. It is not automatically a **source-support direction** or a trained claim detector.

## 4. When to use or skip it

**Use steering** when you have a demonstrated failure and want a bounded inference intervention before committing to training. It still costs compute for calibration and evaluation.

**Use hardening** when safety retention during actual fine-tuning is the question. **Use recovery** when a real drifted checkpoint and compatible references already exist.

**Skip it** for a pure data/evaluation study or an external support classifier that does not need Aya-side intervention. Choose one family for the question; chaining every repair method obscures what caused the change.

## 5. Important limitations

- **The reviewed main WildGuard path scores refusal, not response harmfulness:** it reads the refusal field and ignores the harmfulness field. Its reported ASR/refusal complement is a refusal proxy; independently score harmful content.
- **Benchmark names do not select native graders:** the main suite applies one selected judge across loaded prompts. A safety score on GSM8K is not math accuracy; selecting AdvBench does not remove the default judge.
- **Score the actual wrapper:** the reviewed live-steering utility helper evaluates the unsteered base model. Run utility probes through the live intervention yourself; unchanged weights do not imply unchanged utility.
- **Persistence differs:** recovery `apply()` can mutate the loaded model but does not save it; call `save_checkpoint()` and verify reload. Activation hooks must be reinstalled after reload—they are not encoded by saving unchanged weights.
- **Memory includes more than the target:** references, contrastive models and the default roughly 7.25B WildGuard judge can dominate the budget. Separate generation and scoring where possible. No Kaggle fit has been demonstrated here.
- **No universal layer or strength:** calibrate on the actual checkpoint, template and languages. Include the full chat preamble in token budgets; truncation can remove supervised answer tokens.
- **Implementation faithfulness is not effectiveness:** upstream “Faithful/Variant/Simplified” badges describe method implementations, not independently reproduced Aya safety. Name the method and its deviations accurately.
- **Recovery is not tamper resistance:** repairing one attacked checkpoint does not prove safety survives a subsequent weight attack. Clearly label drifted artifacts and follow the license's responsible-use terms.

**Next:** [05 — CircuitKIT](05-circuitkit.md) asks whether a localized intervention has a causal explanation. Detailed runner/judge findings remain in [15 — Audit](15-missing-components.md).
