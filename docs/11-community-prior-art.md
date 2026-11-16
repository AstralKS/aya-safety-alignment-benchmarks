# 11 — Relevant prior art: what to learn and where to differentiate

Read after [07 — Project ideas](07-project-idea.md). This is a **bounded source review**, not a literature census: five public projects were inspected at their Nov 1, 2026 pins. Reported model results were **not reproduced** here; a public plan or dry run is not a completed model experiment.

## 1. The closest work to our shortlist

### Refusal Parity — multilingual steering already has close prior art

[abdullahabtahi/refusal-parity](https://github.com/abdullahabtahi/refusal-parity) studies whether refusal repair treats languages fairly on Tiny Aya. It combines cleaned harmful/benign prompts, gated activation steering, weight recovery and evaluation.

The reviewed project reports language-specific calibration trade-offs: less benign refusal can come with weaker harmful refusal in some languages. It also reports sensitivity to a benign greeting prefix, poor non-English coverage from English keyword refusal heuristics, and a negative cross-language CircuitKIT comparison.

**Why it matters:** “multilingual refusal + steering” is not an unexplored contribution. Borrow the controls—benign tasks, calibrated thresholds, independent paired tests and negative results—not its scores as predictions for Global or Expanse.

**A narrower question for us:** do script/code-switch changes affect substantive harmful assistance or source-grounded answer/attribute/abstain decisions, and can a targeted intervention preserve legitimate extraction? That remains a hypothesis, not a novelty claim.

### PidginGuard — language selection alone is not enough

[Esimuda/pidginguard](https://github.com/Esimuda/pidginguard) proposes multilingual Nigerian safety evaluation and repair using CuratorKIT, SafeTune and AuditKIT, with optional circuit analysis. It includes harmful prompts and benign controls across Nigerian languages/registers.

At the reviewed pin, the visible evidence was scaffolding and offline harness work; live Tiny Aya execution awaited gated access.

**Why it matters:** a multilingual pipeline is a useful shape, but merely changing the language list is weak differentiation. Specify the failure, validate equivalent intents and measure the utility cost.

### lexsi_redteam — useful controls, limited outcome scope

[Zhiyilang/lexsi_redteam](https://github.com/Zhiyilang/lexsi_redteam) has prompt-level abuse cases, benign-but-sensitive controls and dry-run measurement plumbing. No model-backed robustness result was verified in this review.

**Why it matters:** legitimate “edgy” tasks help expose blanket refusal. Threat-sounding prompts or a successful mock run do not independently establish harmful real-model behavior; use content-level labels for the response actually produced.

## 2. A useful experimental template: curation controls

[TirthNandha/medcurate-pubmedqa](https://github.com/TirthNandha/medcurate-pubmedqa), from a different event, compares raw and CuratorKIT-filtered synthetic data before AlignTune training and AuditKIT evaluation.

Its useful feature is the **size-matched raw control**. Comparing 800 raw examples with 400 curated examples changes both quality and training exposure. Adding a 400-example raw arm helps separate those effects. The reviewed report also notes label-distribution changes after filtering and inconclusive metrics, rather than treating non-significance as equivalence.

For an Aya trained-repair experiment, keep these controls:

- Same starting checkpoint and comparable training settings.
- Raw versus curated data at matched sample count; report token exposure too.
- Class/language balance and the CuratorKIT rejection funnel.
- Held-out source groups, paired analysis and independently scored utility.

The reported PubMedQA gains and serving speeds are **not** Aya performance estimates.

## 3. What a defensible contribution could be

| Candidate from `07` | Beyond the closest reviewed work | Evidence needed |
|---|---|---|
| Claim-grounded selective abstention | Separate unsupported assertion, faithful attribution and unnecessary refusal | Reviewed source spans/labels; risk–coverage and supported-task quality; strong prompt baseline |
| Code-mixed safety with targeted steering | Validated script/code-switch contrasts and a selective intervention | Harmful-content labels, benign utility, language fidelity and held-out intents |
| Optional localization | Test whether a selected mechanism beats matched random interventions | Cohere parity, patching/ablation and independent behavioral outcomes |
| Optional compression | Re-test a real intervention at the deployed precision | Matched unquantized/quantized conditions and exact-artifact measurements |

None of these rows establishes that no similar paper exists. Before claiming novelty, search the primary literature for the **chosen narrow question**, not every toolkit name.

A useful thesis sentence is: “On these two Aya checkpoints, under this controlled language/evidence shift, this locked intervention changes this error rate at this measured utility cost.” Fill it with results only after execution.

## 4. Practices worth borrowing

1. **Freeze the decision rules:** document metrics, splits and method-selection rules before final testing; log deviations.
2. **Report the price of safety:** more refusal is not automatically less harmful assistance, and less answering is not automatically better grounding.
3. **Show difficult controls:** benign sensitive tasks, source-supported allegations, unfamiliar scripts and simple formatting changes.
4. **Publish evidence per condition:** baseline, prompt intervention and learned/steered condition should have distinct configurations, responses and artifact identities.
5. **Keep failures:** port failures, judge disagreement and negative or inconclusive effects are findings, not rows to discard.
6. **Credit and check reuse:** cite borrowed code/data and verify event-period, model and license eligibility separately. Public availability is not blanket permission.

**Next:** [12 — Representation research](12-neuroai-angles.md) connects this shortlist to probes, geometry and causal interventions. Exact reviewed pins remain in the later source ledger, `13`.
