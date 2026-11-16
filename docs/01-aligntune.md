# 01 — AlignTune: teaching a model new behavior

**Role:** general-purpose LLM post-training. **Repo:** [Lexsi-Labs/aligntune](https://github.com/Lexsi-Labs/aligntune). **Python:** ≥3.11.

**Reading order:** 01 → 02 → 03 → 04 → 05 → 07 → 11 → 12 → 13 → 14. These first five guides explain the tools before asking you to choose a project.

## 1. The problem it solves

A system prompt tells a model how to behave **in this conversation**. Fine-tuning changes its learned parameters, called **weights**, so the behavior can persist across conversations.

AlignTune supplies the training machinery: loading a model and data, choosing a training method, configuring memory use, running updates and saving the result. It does not decide what good behavior means; your examples or rewards do that.

For us, the distinction is simple: exploring Aya with prompts needs no AlignTune. Teaching Aya a better answer/attribute/abstain policy may need it.

## 2. How it works

Choose the method from the supervision you actually have:

| Method | What you supply | What the model learns |
|---|---|---|
| **SFT — supervised fine-tuning** | Prompts with desired responses | Produce responses like the demonstrations |
| **DPO — direct preference optimization** | The same prompt with a preferred and rejected answer | Prefer one response over the other; no separate reward model required |
| **GRPO — group-relative policy optimization** | Prompts and a function that scores generated responses | Improve against rewards relative to other responses in a sampled group |
| **Distillation** | A teacher's outputs or probability distributions | Approximate the teacher; it can inherit the teacher's errors |

Start with SFT if you can write good demonstrations. DPO is useful when the distinction between two plausible answers matters. GRPO adds generation and reward-design complexity; a reward for saying “I cannot answer” can teach blanket abstention rather than good judgment.

### Full weights versus adapters

**LoRA** trains small additional matrices while freezing the original weights. This reduces trainable parameters and optimizer memory, but still requires the base model in memory. The saved adapter is normally used **with** that base model, not as a standalone model.

**QLoRA** additionally quantizes the frozen base. LoRA does not imply four-bit loading: training an adapter against an unquantized base is a separate choice.

AlignTune routes training through a **backend**, meaning the implementation that performs the updates. TRL is the explicit starting choice; Unsloth is another supported route with architecture-dependent optimizations. Neither a backend label nor an upstream speed claim guarantees our Aya run will work.

## 3. One Aya example

Suppose the source says: “Meera alleges the supplier inflated the invoice; no independent verification is provided.” We want:

- **Extract the allegation:** retain “Meera alleges,” rather than assert that inflation happened.
- **Verify the allegation:** say the supplied evidence cannot establish whether it is true.
- **Answer supported questions:** do not refuse all invoice-related analysis.

An SFT dataset contains demonstrations of all three moves, including benign controls. A DPO version pairs faithful attribution with an unsupported factual assertion. The training format must match the chosen method.

This **source-checked SFT template** requires your own chat dataset at `output/repair/`, with the named `sft_messages` configuration. It has not been executed on Aya; the settings are a smoke-test starting point, not tuned hyperparameters.

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="CohereLabs/tiny-aya-global",
    dataset_name="output/repair/",
    config_name="sft_messages",
    backend="trl",
    use_peft=True,
    lora_r=8,
    num_epochs=1,
    max_samples=32,
    max_seq_length=1024,
    batch_size=1,
    learning_rate=2e-5,
    output_dir="./runs/aya-sft-smoke",
)
trainer.train()
```

The result is a trained artifact, not proof of improvement. Evaluate unseen source groups and reload the saved base-plus-adapter combination before making that claim.

## 4. When to use or skip it

**Use it** for an actual SFT, preference-learning or reward-based training arm on Tiny Aya Global or Aya Expanse 8B.

**Skip it** for prompt comparisons, evaluation alone, live activation steering, or fitting an external classifier on already-collected features. Those do not require changing Aya's weights.

You do not need to train simply because training is available. A strong prompt-only baseline tells you whether a learned intervention adds anything.

## 5. Important limitations

- **Enable LoRA explicitly:** the reviewed SFT factory defaults PEFT off. `lora_r` alone is insufficient; pass `use_peft=True`.
- **Inference fit is not training fit:** gradients, optimizer state, activations and sometimes reference models add memory. The notebooks' two-GPU inference sharding is not a distributed-training configuration.
- **Keep environments separate initially:** the playground pins `transformers==4.57.6`; reviewed Lexsi training dependencies require a newer stack, including `transformers>=5.15,<6`. Do not upgrade the working playground blindly.
- **Data and tokenizer compatibility matter:** Tiny Aya uses `cohere2`; Expanse uses `cohere`. Their token vocabularies differ, so direct token-distribution distillation between them needs an explicit alignment method; text-output distillation is different.
- **Built-in evaluation is not a release gate:** reviewed named benchmark bundles have resolver defects, and the audit callback logs diagnostics rather than enforcing a safety threshold. Use a working independent evaluator.
- **Licenses are layered:** AlignTune's own code is MIT, but its wheel bundles components with additional licenses, including AGPL/LGPL obligations. Review the shipped package before redistribution.

**Next:** [02 — CuratorKIT](02-curatorkit.md) explains where reliable training examples come from. Detailed implementation caveats remain in [15 — Audit](15-missing-components.md).
