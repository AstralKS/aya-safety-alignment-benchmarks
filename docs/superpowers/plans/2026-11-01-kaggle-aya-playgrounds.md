# Kaggle Aya Playgrounds Implementation Plan

**Goal:** Deliver two standalone Kaggle notebooks for exploratory, editable-prompt inference on Tiny Aya Global and Aya Expanse 8B.

**User-specified design:** Load one model per notebook; expose the system prompt, user message and generation settings; include demanding Markdown prompt cards covering multilingual/code-mixed safety, hallucination/abstention, and baseline questions relevant to steering, localization, compression and tamper resistance.

**Execution:** Inline in the current workspace. This workspace is not a Git repository. Existing research repositories and starter implementation are outside this change.

**Hardware and budget:** Kaggle T4 x2 first; HF $20 used alongside Kaggle when an eligible service/hardware suits the task; Modal $50 reserve. Downloads use HF authentication, while notebook generation runs locally on Kaggle. No paid service is invoked by the notebooks.

**Architecture:** Share notebook construction outside the workspace, but embed every helper and prompt in each output notebook. Per the user's precision preference, default both models to unquantized 16-bit loading: FP16 on T4, BF16 on compatible newer GPUs, and balanced inference dispatch across available GPUs. NF4 four-bit loading is opt-in. Log actual configuration, prompts and responses. Prompt-only pressure is distinguished from training-time tampering; activation inspection is distinguished from learned probes and causal circuits.

**Deliverables:**
- `notebooks/tiny-aya-global-kaggle.ipynb`
- `notebooks/aya-expanse-8b-kaggle.ipynb`
- `notebooks/PROMPTS.md`
- `notebooks/README.md`

## Tasks

- [x] Establish compatible dependency pins, Kaggle setup instructions, gated-model authentication and T4-aware loading.
- [x] Add native chat-template rendering, editable single-turn/multi-turn prompts, bounded generation, per-run logging and manual review fields.
- [x] Embed a shared bank of challenging prompts, system-prompt variants, expected behaviors and track-specific interpretation notes.
- [x] Add opt-in comparisons, lightweight last-prompt-token activation capture and result export.
- [x] Validate notebook structure, compile code cells and exercise CPU-only helper behavior; document that gated GPU execution remains unperformed.

## Validation boundaries

Use notebook-schema validation and focused local checks of prompt assembly, token-budget enforcement, continuation decoding, logging and hook cleanup. These checks do not establish Kaggle installation success, model-loading success, model capabilities, training results or GPU memory fit. Actual model execution is left to the user's authenticated Kaggle sessions.

**Completed validation:** both 61-cell notebooks pass notebook-schema validation; all 28 code cells compile. Both contain identical 24-case/8-system libraries and the same guide as `notebooks/PROMPTS.md`. Focused CPU fixtures passed for explicit system/history preservation, rejection of dropped system text, context-budget enforcement, continuation-only decoding, EOS/length/empty-output handling, failed-run logging, activation output shape and hook cleanup after both success and failure. No model weights were downloaded and no GPU execution occurred.
