# 05 — CircuitKIT: testing which model components matter

**Role:** circuit discovery, causal checks and targeted interventions. **Repo:** [Lexsi-Labs/CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT). **Python:** ≥3.10. **License:** LSAL v1.2.

## 1. The problem it solves

[SafeTune](04-safetune.md) can change behavior by steering or editing a model. But improved scores alone do not explain **which components mattered**, or whether a proposed localization is better than a random intervention.

CircuitKIT helps discover candidate components, test them and use the result for targeted editing, fine-tuning or compression. Its central question is causal: **does changing this part of the computation change the declared behavior?**

## 2. How it works

### The minimum transformer vocabulary

A transformer layer maintains a vector representation called the **residual stream**. Attention combines information from token positions; an **attention head** is one such channel. An **MLP** is the layer's feed-forward network, which transforms representations.

A **circuit** is a selected set of components or connections hypothesized to contribute to one task. It is not a physical brain region, and there need not be one unique or minimal set.

**Attribution** ranks components by their estimated contribution to a chosen output score. CircuitKIT's default EAP-IG—edge attribution patching with integrated gradients—uses gradient-based estimates across a clean/corrupted contrast. This is a discovery method, not the final causal proof.

### Discovery needs a precise target

```text
defined behavior + controlled contrast → attribution ranking → candidate circuit
                                         ↓
                            patching/ablation + controls
                                         ↓
                           independently scored intervention
```

Specify the prompt distribution, token position and score first. Many tasks use a next-token score; that is narrower than the correctness of a whole generated answer. Multi-token claims require an appropriate answer-span metric or a carefully validated proxy.

**Patching** replaces selected activations in a corrupted run with those from a clean run. **Ablation** removes or substitutes components to test what remains. Both are interventions; a colorful activation heatmap is not.

### What the six faithfulness checks ask

Here, **faithfulness** means how well the selected circuit accounts for the defined behavior—not whether the answer is truthful or safe.

| Check | Question |
|---|---|
| Patching | Does restoring the selected clean computation recover the behavior? |
| Ablation | Does the selected set retain behavior when outside components are ablated? |
| Stability | Are similar components selected across seeds? |
| Robustness | Does the result survive controlled input changes? |
| Baselines | Does it beat random or magnitude-ranked sets of matched size? |
| Generalization | Does it explain a declared related task? |

Start with patching, ablation and baselines; broaden only if resources permit. Report which checks ran. A sufficient set can be redundant, and stable rankings can still fail causal tests.

## 3. One Aya example

Define a narrow source-support task: “Does the source state that the committee approved the proposal? Reply YES or NO.”

- **Clean source:** directly records approval.
- **Corrupted source:** discusses the same proposal but records no decision.
- **Target:** the model's score for the correct support decision, with language/template controlled.

Check tokenization, answer positions and length confounds before interpreting attribution. Discover on development groups, then patch/ablate on held-out groups and compare matched random components. Separately test whether a targeted intervention improves real claim extraction without causing blanket abstention.

The `Pipeline` API organizes discovery, faithfulness evaluation and interventions. `Pipeline.from_custom_data()` specifically requires CSV with clean/corrupt prompt and answer columns; a CuratorKIT JSONL export needs an explicit conversion or a different loader.

For compatibility plumbing, Tiny Aya's documented smoke task is `greater_than`, whose operands are checked against its tokenizer. Passing that task says nothing about a safety or source-support circuit.

## 4. When to use or skip it

**Use it** when a mechanistic explanation or localized intervention is central enough to justify model-port checks and additional experiments.

**Keep it optional** when the main result is behavioral evaluation, abstention or steering. A good held-out result does not need a circuit graph, and failed parity should not block the entire project.

Circuit-informed scores can guide SafeTune targeting or selective compression. That is a useful connection only if the circuit was validated for the **relevant behavior**; an artifact field named `safety_units` does not certify safety.

## 5. Important limitations

- **Cohere needs a parity gate:** Tiny Aya is `cohere2`; Aya Expanse 8B is `cohere`/`CohereForCausalLM`. Both use parallel attention/MLP blocks, so interventions cannot assume a sequential block layout. CircuitKIT uses a compatibility port; import `circuitkit` before `transformer_lens`.
- **Parity means matching the original computation:** compare the port's output distribution with Hugging Face under the exact settings used downstream. The documented Tiny Aya gate uses next-token KL divergence below `1e-4`, including sliding-window coverage. No real-weight gate has been run here.
- **Inference sharding does not transfer:** the reviewed discovery path has no supported multi-GPU layer-sharding setup. The notebooks' `device_map="balanced"` is not a CircuitKIT discovery configuration; gradients and cached activations add memory.
- **Pruning is zero-masking:** the reviewed path zero-masks weights while retaining the architecture. Do not claim fewer stored parameters or faster inference without measuring the actual artifact.
- **Quantization backends differ:** `quanto` does not honor the requested `bits` setting in the reviewed path; `llmcompressor` has a different contract. Inspect the resulting precision, not just the config string.
- **Hooks and checkpoints are different artifacts:** live activation hooks need reinstalling; persistent weight edits need export/reload checks. Preserve model revision, task, template and score definitions with circuit artifacts.
- **Attribution is not safety evidence:** re-score harmful assistance or source support and utility independently. Random, size/layer/norm-matched controls are needed before crediting a localization with selective benefit.

**Next:** [07 — Project ideas](07-project-idea.md). You now have the tool vocabulary needed to compare the candidate projects without combining every repository.
