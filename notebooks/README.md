# Aya model playgrounds for Kaggle

Two self-contained notebooks with the **same prompt bank and generation interface**:

| Notebook | Model |
|---|---|
| [tiny-aya-global-kaggle.ipynb](tiny-aya-global-kaggle.ipynb) | `CohereLabs/tiny-aya-global` — 3.35B |
| [aya-expanse-8b-kaggle.ipynb](aya-expanse-8b-kaggle.ipynb) | `CohereLabs/aya-expanse-8b` — 8B |

The full [prompt guide](PROMPTS.md) is also embedded as Markdown inside **both** notebooks. Each notebook works without uploading any other project files.

## Start on Kaggle

1. Accept the model's access conditions on Hugging Face while signed into your own HF account:
   - <https://huggingface.co/CohereLabs/tiny-aya-global>
   - <https://huggingface.co/CohereLabs/aya-expanse-8b>
2. Create a Kaggle notebook and import the corresponding `.ipynb` file.
3. Set **Accelerator: GPU T4 x2**, and turn **Internet on**.
4. In **Add-ons → Secrets**, add `HF_TOKEN` with permission to read the gated models and enable notebook access to it. A hidden `getpass` fallback is provided.
5. Run the setup, authentication/loading and helper cells in order. If dependency setup requests a kernel restart, restart and run from the top.
6. In **YOUR PLAYGROUND**, edit `SYSTEM_PROMPT` and `USER_MESSAGE`, then rerun the generation cell. Model loading only happens once per session.
7. Try the named prompt cases and system variants. Comparisons and activation capture are opt-in.

The loader downloads the official weights and defaults to **unquantized 16-bit loading**. Four-bit quantization is opt-in; if enabled, it happens in memory and does not reduce the original download size. The cache is under `/kaggle/temp`; result logs go under `/kaggle/working/aya-playground`.

## Hardware and comparison settings

- Both notebooks default to **unquantized 16-bit weights** (`LOAD_IN_4BIT=False`): **FP16 on T4**, automatically **BF16 on compatible Ampere-or-newer GPUs**. T4 does not provide native BF16 acceleration.
- With two GPUs, `DEVICE_MAP="balanced"` dispatches model layers across them. This is inference model sharding; it is not distributed fine-tuning and does not automatically double generation speed.
- Each T4 has its own memory pool. The loader reserves GPU headroom and reports the actual device map and peak memory.
- To try single-GPU inference, use `DEVICE_MAP="cuda:0"`; four-bit inference may be faster that way because it avoids cross-GPU transfers. Available memory and prompt length still matter.
- For a quantization comparison or lower-memory run, change `LOAD_IN_4BIT=True` in a fresh session to enable NF4 with double quantization. Keep prompts, history, seeds, decoding and precision matched when comparing the two models.
- Approximate 16-bit weight storage is 6.7 GB for Tiny Aya Global and 16 GB for Aya Expanse 8B, before runtime overhead. Balanced sharding across T4 x2 is the starting configuration; actual peak memory depends on the prompt and generation settings.
- Input is never silently truncated. The helper raises when the rendered prompt plus requested output exceeds its configured context budget.

## Compute budget

Current resources: **2 Kaggle accounts, $20 HF credit, $50 Modal credit**. Use Kaggle for this exploration; use HF alongside it where the specific credited service and hardware fit the task; keep Modal as reserve. Actual GPU availability and quotas determine usable compute. HF credit coverage and hardware determine whether an HF training run is feasible or faster.

These notebooks call **local Transformers generation on Kaggle**, not a paid HF inference endpoint. HF is used for model downloads and authentication.

## What the experiments establish

The cases cover multilingual/code-mixed safety and over-refusal, source-grounded claim extraction, hallucination/abstention, evidence injection, multi-turn pressure, and precision-sensitive tasks. Each card explains what to inspect.

Prompt comparisons supply behavioral baselines for later steering. Optional activation capture exposes real layer-output vectors at the final prompt token; it is not a trained hallucination detector, R/S lens, neuron explanation, or causal safety circuit. Compression comparisons require separate precision runs. Tamper resistance requires a subsequent training attack, not a persuasive system prompt.

The exploration uses Transformers. A later submission can integrate **CuratorKIT + AuditKIT**, with SafeTune/CircuitKIT for the selected intervention and analysis; running these notebooks alone does not demonstrate the two-Lexsi-tool requirement.

## Outputs

Each session creates a model-specific timestamped folder containing:

- `manifest.json`: model revision, precision, library versions and hardware.
- `runs.jsonl`: exact system/user/history, rendered prompt, decoding settings, response, timing, token counts, finish reason and execution status.
- `reviews.jsonl`: optional manual labels linked to a run ID.
- `PROMPTS.md`: the embedded prompt guide.
- Optional activation tensors and their prompt/model metadata.

The export cell creates a ZIP for download through Kaggle's output browser. It exports results, not model weights or HF credentials. Compare the same case IDs between the two notebooks. Errors and length-limited responses remain visible; empty outputs are not scored as safe or correct.

## Common issues

| Symptom | Action |
|---|---|
| 401/403 while loading | Accept access conditions for that exact model; confirm the attached `HF_TOKEN` can read it. |
| CUDA unavailable | Enable GPU and start a fresh GPU session. |
| Import errors after installing | Restart the kernel. These notebooks pin Transformers 4.57.6 and compatible supporting packages while retaining Kaggle's PyTorch. |
| An existing `torchvision`/`librosa` import fails | These text notebooks do not need those optional packages; in the disposable Kaggle session remove the conflicting optional package and restart, or start a clean image. |
| Out of memory | Shorten the prompt/output and use a fresh session if a previous model remains loaded. If needed, opt into four-bit loading as a separately labeled precision condition. |
| Abruptly cut-off JSON/answer | Check `finish_reason`; increase `max_new_tokens` within the context budget. |
| System prompt seems ignored | Inspect the rendered chat-template cell. The helper refuses to silently discard a supplied system prompt. |
| Quantized outputs differ from FP16 | This is an experimental result; retain the precision labels and compare matched cases. |

## Verification status

Both notebooks pass schema validation, and all **28 code cells compile**. Their **24 cases and 8 system variants match**. Focused CPU-only fixtures passed for chat assembly, context limits, output decoding, error logging and activation-hook cleanup. **No authenticated Kaggle GPU/model run is claimed.** The prompt rubrics are authored expectations, not measured model results.
