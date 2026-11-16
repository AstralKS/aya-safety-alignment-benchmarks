# 14 — Relevant papers: what to read and what to ask

Read after `13`. Linked paper metadata and abstract-level claims were checked on **Nov 1, 2026**; full methods were not audited and experiments were not reproduced. This is a selected reading guide, not a catalog of every Lexsi-related paper.

## 1. Start with papers that answer our actual questions

| Paper | Why read it? | Question to bring to the full paper |
|---|---|---|
| **[Tiny Aya: Bridging Scale and Multilingual Depth](https://arxiv.org/abs/2603.11510)** | Understand the 3.35B multilingual subject and Global/regional variants | Which languages/tasks support the capability claims, and which safety behaviors remain untested? |
| **[Provenance-Grounded Gating and Adaptive Recovery](https://arxiv.org/abs/2606.11127)** | Learn why grounding, quality gates and recovery can affect different examples | What gets rejected or recovered, and what happens to difficult labels and downstream utility? |
| **[SafeTune](https://arxiv.org/abs/2609.22153)** | Compare hardening, recovery, unlearning and steering as different interventions | Are starting checkpoints, scorers and utility costs comparable across methods? |
| **[Faithfulness to Refusal: A Causal Audit of Neuron Selectors](https://arxiv.org/abs/2607.05355)** | Understand why a stable importance ranking is not enough | Do selected components beat layer-matched random controls, and what capability is lost? |

These four are the useful starting set for `07`'s shortlist and `12`'s representation questions. The Tiny Aya paper does not establish Expanse 8B compatibility or our models' safety.

## 2. Add these only for the corresponding experiment

- **Actual training:** [AlignTune](https://arxiv.org/abs/2602.09621) explains its common training/backend boundary. Read the underlying SFT/DPO/GRPO method papers for the algorithm itself; using the factory is not a new training method.
- **A curation contribution:** [CuratorKIT](https://arxiv.org/abs/2606.21631) explains the library pipeline, exports and provenance. Pair it with the gating paper above to distinguish implementation from experimental evidence.
- **Localized weight repair:** [C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal](https://arxiv.org/abs/2602.04521) reports circuit-restricted updates, typically below 5% of parameters in its experiments, producing a persistent checkpoint without inference hooks. Ask how the selected parameters were validated and whether the effect transfers to our Aya task; do not rename a reproduction as a new method.
- **Compression or unlearning:** [Forgetting That Sticks](https://arxiv.org/abs/2605.15138), the MANSU paper, studies whether unlearning survives quantization. Small weight updates can be lost when values are rounded into quantization bins. Ask whether the exact exported artifact retains the target behavior and utility; this is not a theorem that every safety repair fails under NF4.

CircuitKIT's repository is the implementation reference for circuit APIs. C-ΔΘ is a method paper, not independent validation of the library or its Cohere port. AuditKIT's API and reviewed defects likewise require repository evidence, not a neighboring toolkit paper.

## 3. Three ideas worth taking from the papers

**Curation is a selection process.** Gates can remove different populations and sometimes good examples. Report per-gate rejection overlap, recovered yield and label/language balance; use a matched raw-data control if training gains are the claim.

**Localization can be redundant.** Different, even disjoint component sets can retain a behavior. Faithfulness tests and utility controls support a specific intervention claim; neither a stable selector nor one successful ablation proves the unique mechanism.

**The deployment artifact is another experimental condition.** A repaired full-precision checkpoint, quantized export and live-steered wrapper are not automatically equivalent. Re-test the actual condition you release. Adding extra precision arms is optional; verifying the chosen artifact is not.

Together, these papers motivate experiments; they do **not** validate an end-to-end Aya pipeline, establish source-support labels or guarantee multilingual detector transfer. Connect each cited claim to the model, task and controls it actually studied.

## 4. Citation record

The checked citation block below is preserved from the earlier review. These entries cite arXiv preprints, not verified conference acceptance. Re-check revisions and venue metadata when submitting, and cite exact repository revisions for implementation behavior.

<details>
<summary>BibTeX for the eight selected references</summary>

```bibtex
@article{aligntune2026,
  title   = {AlignTune: Modular Toolkit for Post-Training Alignment of Large Language Models},
  author  = {Lyngkhoi, R E Zera Marveen and Chawla, Chirag and Seth, Pratinav and Avaiya, Utsav
             and Bhattacharjee, Soham and Khandoga, Mykola and Yuan, Rui and Sankarapu, Vinay Kumar},
  journal = {arXiv preprint arXiv:2602.09621}, year = {2026},
  url = {https://arxiv.org/abs/2602.09621}}

@article{curatorkit2026,
  title   = {CuratorKIT: Data Curation and Synthetic Data Generation for LLM Post-Training},
  author  = {Bhattacharjee, Soham and Sharma, Karun and Sankarapu, Vinay Kumar and Seth, Pratinav},
  journal = {arXiv preprint arXiv:2606.21631}, year = {2026},
  url = {https://arxiv.org/abs/2606.21631}}

@article{provenancegating2026,
  title   = {Provenance-Grounded Gating and Adaptive Recovery in Synthetic Post-Training Data Curation},
  author  = {Bhattacharjee, Soham and Sharma, Karun and Sankarapu, Vinay Kumar and Seth, Pratinav},
  journal = {arXiv preprint arXiv:2606.11127}, year = {2026},
  url = {https://arxiv.org/abs/2606.11127}}

@article{safetune2026,
  title   = {SafeTune: A Unified Faithful Library for Auditing and Repairing Safety Drift in Fine-Tuned LLMs},
  author  = {Seth, Pratinav and Sadhu, Saisab and Kaushal, Anshul and Sankarapu, Vinay Kumar},
  journal = {arXiv preprint arXiv:2609.22153}, year = {2026},
  url = {https://arxiv.org/abs/2609.22153}}

@article{cdelta2026,
  title   = {{C}-{$\Delta\Theta$}: Circuit-Restricted Weight Arithmetic for Selective Refusal},
  author  = {Kasliwal, Aditya and Seth, Pratinav and Sankarapu, Vinay Kumar},
  journal = {arXiv preprint arXiv:2602.04521}, year = {2026},
  url = {https://arxiv.org/abs/2602.04521}}

@article{mansu2026,
  title   = {Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution},
  author  = {Sadhu, Saisab and Seth, Pratinav and Sankarapu, Vinay Kumar},
  journal = {arXiv preprint arXiv:2605.15138}, year = {2026},
  url = {https://arxiv.org/abs/2605.15138}}

@article{faithfulnessrefusal2026,
  title   = {Faithfulness to Refusal: A Causal Audit of Neuron Selectors},
  author  = {Eswar, Ananth and Seth, Pratinav and Avaiya, Utsav and Sankarapu, Vinay Kumar},
  journal = {arXiv preprint arXiv:2607.05355}, year = {2026},
  url = {https://arxiv.org/abs/2607.05355}}

@article{tinyaya2026,
  title   = {Tiny Aya: Bridging Scale and Multilingual Depth},
  author  = {Salamanca, Alejandro R. and Abagyan, Diana and D'Souza, Daniel and Khairi, Ammar
             and Mora, David and Dash, Saurabh and Aryabumi, Viraat and Rajaee, Sara
             and Mofakhami, Mehrnaz and Sahu, Ananya and Euyang, Thomas and Prince, Brittawnya
             and Smith, Madeline and Lin, Hangyu and Locatelli, Acyr and Hooker, Sara
             and Kocmi, Tom and Gomez, Aidan and Zhang, Ivan and Blunsom, Phil and Frosst, Nick
             and Pineau, Joelle and Ermis, Beyza and {Üstün}, Ahmet and Kreutzer, Julia
             and Fadaee, Marzieh},
  journal = {arXiv preprint arXiv:2603.11510}, year = {2026},
  url = {https://arxiv.org/abs/2603.11510}}
```

</details>

For project choices return to [07](07-project-idea.md); for probe/intervention controls return to [12](12-neuroai-angles.md). [13](13-research-notes.md) remains the historical source ledger, not a fresh execution report.
