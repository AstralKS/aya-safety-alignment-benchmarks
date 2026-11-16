# 06 — AgentTune

> **Status (Nov 8, 2026):** AgentTune is **not part of the current shortlist** (`07`). This guide is
> retained as a tool reference; its closing Track 3/agent-project notes are historical.

> **One-line:** build tool-using agents, then train, evaluate, distill, and self-heal them through
> one trajectory schema (`EventLog`) — GRPO/PPO/DPO/RLOO/BCO on TRL, plus agent design, memory,
> harness, RAG, DECIDE workflows, and a closed-loop failure→retrain→gate pipeline.
>
> **Repo:** `github.com/Lexsi-Labs/AgentTune` · **PyPI:** `agenttune` · **License:** LSAL v1.1 ·
> **Python:** ≥3.12 · **Version at review:** 1.1.0 · **Pin:** `torch==2.11.0`,
> `transformers>=5.15,<6`, `trl==1.7.1`.

---

## 1. The two halves

1. **Agentic post-training on TRL** — `create_agentic_trainer(algorithm=...)` exposes GRPO, PPO,
   DPO, RLOO and BCO. Configured online paths actually execute tools; DPO/BCO also accept offline
   preference data, so selecting an algorithm alone does not establish live tool rollouts.
2. **Agentic spine** — one normalized trajectory schema moves one agent artifact through
   `build → collect → evaluate → train → distill → heal`.

Strategies, memory, harnesses, DECIDE and lifecycle wiring share `EventLog`. Many paths can be
tested with injected fakes and no GPU; real training, embeddings, judges and external tools need
their respective models/services. **GPU-free does not mean dependency-free:** the base package
includes LangGraph, OpenEnv and a substantial ML stack (`pyproject.toml:79–192`).

---

## 2. EventLog — the shared contract

`agenttune/agentic/events.py`:

| Tier | Source | Carries | Usable for |
|---|---|---|---|
| `full` | `collect_rollout()` / native rollout projection | `token_span`, `logprobs` | SFT rows, GRPO masking, distillation (trainable) |
| `light` | observational (`DictToolHarness`, DECIDE `PipelineState`, eval dicts) | events only | UI / eval / heal (read-only) |

A `light` log can be evaluated and scanned for failures but **cannot be trained on**: `as_dataset_rows` /
`masked_tokens` raise if tier ≠ `full`. Projection naming: `as_*` = cast of the log's own events;
`to_*` = new representation for a consumer (`to_eval_dict`, `to_audit_records`).

`EventKind` includes action/tool/observation/reward/thought/finish-style events; `Event` carries
`payload`, optional `token_span`, `logprobs`, and a `scope` for reward events
(`step|turn|episode`).

`Project.infer()` and `collect()` produce **light** logs (`project.py:178–195`).
`collect_rollout()` produces full logs (`:138–176`); `train()`/`distill()` require a supplied
trainer factory, and `train(fmt="grpo")` requires an on-policy rollout engine. The lifecycle API
does not turn an observational log into valid training evidence automatically.

---

## 3. The spine pillars

### Agent design (`strategy.py`, `strategy_advanced.py`, `strategy_memory.py`, `strategy_tree.py`)

Every strategy reduces to `init → propose → observe → is_done`, with model access behind an
injected `policy` callable (testable with no model):

- `ReActStrategy` — reference ReAct (arXiv:2210.03629)
- `PlanExecuteStrategy` — Plan-and-Solve (arXiv:2305.04091)
- `ReflexionStrategy` + `run_reflexion` — retry with verbal self-reflections (arXiv:2303.11366)
- `MemoryReActStrategy` — ReAct with memory recall/write
- `TreeOfThoughtsStrategy` — ToT/LATS (arXiv:2305.10601, 2310.04406); greedy beam plus
  best-first `search()` with backtracking

### Memory (`memory.py`, `memory_vector.py`, `memory_graph.py`)

- `MemoryKind` (working / episodic / semantic / procedural / entity — CoALA), `Scope`,
  `MemoryItem`
- `InContextMemory` — recent-k scratchpad
- `VectorMemory` — cosine semantic recall (injected embedder)
- `GraphMemory` — typed directed edges, `link`/`neighbors` BFS, logical-clock temporal decay,
  `forget_older_than` (Graphiti/A-MEM inspired)
- `TrajectoryStore` — episodic replay of `full` logs; emits distillation datasets

Trainable `consolidate`/`forget` policies follow the Memory-R1 pattern; `snapshot`/`restore`
supports rollout reset/replay.

### Harness (`harness.py`, `harness_openenv.py`)

Gym-like `reset`/`step`/`action_space`, capability flags with a DRIFT conformance bench, and
`replay()`:

- `DictToolHarness` — pure-Python dict of callables; produces `light` logs
- `OpenEnvHarness` — adapter to a caller-supplied local/remote OpenEnv environment
- `run_conformance` → `ConformanceReport`; `replay(harness, log)` re-executes a log's tool calls

### Tools

Built-ins (`agenttune/agentic/tools/builtin/`): `ReadFileTool`, `WriteFileTool`, `ListDirTool`,
`GrepTool`, `RunPythonTool`, `RunBashTool`, `SQLDatabaseTool`, `WebSearchTool`, `HttpGetTool`,
`HttpPostTool`, `GitHubTool`, `SlackTool`, `PlaywrightTool`, `FileIngestionTool`, `OpenEnvTool`,
plus a `ToolRegistry` and `ToolExecutor`.

`IsolatedTool` uses `ProcessPoolExecutor`, a timeout and a picklability check. Its default
`require_isolation=False` allows **in-process fallback** (`tool_isolation.py:59–64,89–137`).
A child process still has the host user's filesystem/network authority; this is not a hostile-code
sandbox. OpenEnv's security boundary depends on the supplied deployment, credentials, mounts and
egress policy, not merely on constructing `OpenEnvHarness`.

Registration makes tools discoverable, not automatically granted to every agent. If explicitly
exposed, `RunBashTool` runs model-supplied commands with `shell=True`, and `RunPythonTool` runs
arbitrary code on the host (`code_tools.py:23–45,62–80`). For AgentGuard use narrow **mock tools**,
default-deny capabilities and no ambient credentials; see audit finding 7.

### RL training (`create_agentic_trainer`)

```python
def create_agentic_trainer(algorithm: str, backend: str = "auto", **kwargs) -> Any
```

- `algorithm`: `"grpo" | "dpo" | "ppo" | "rloo" | "bco"`
- `backend`: `"auto"` (Unsloth if importable, else TRL) | `"trl"` | `"unsloth"` (unofficial,
  install yourself)
- `**kwargs` forwarded to the trainer: `model`, `train_dataset`, `tools`, `reward_funcs`,
  `output_dir`, `max_steps`, batch/generation settings, `max_steps_per_turn`, `use_vllm`,
  `vllm_mode`, `peft_config`, `log_completions`, `report_to`, `push_to_hub`, `hub_model_id`
- All five have **agentic tool-calling paths** when configured. DPO/BCO also accept offline
  complete datasets without live rollouts; a reward function alone does not enable their tool
  loop (`agentic_dpo.py:305–334,422–445`; `agentic_bco.py:419–447,536–556`).
- Rollout engines: `transformers` (local), `vllm` (fast batched, Linux+CUDA), `api` (any LiteLLM
  provider); `create_rollout_engine` / `create_rollout_fn`

Use **`backend="trl"` explicitly** in reproducible runs. `auto` can change backend when Unsloth
becomes importable (`core/backend_factory.py:163–178`); it is not an invariant TRL default.

### Rewards

`REWARD_REGISTRY` has **26 named functions**; `combine_rewards(...)` builds weighted composites.
Groups: SQL (`correctness_reward`, `structure_reward`, `query_reward`, `sql_format_reward`),
generic use-case (`reward_correct_answer`, `reward_tool_used`, `reward_concise_answer`,
`format_reward`), RAG (`search_grounding_reward`, `message_id_citation_reward`,
`answer_format_reward`, `answer_correctness_reward`), FinQA/calculation
(`placeholder_coverage_reward`, `template_structure_reward`, `computation_reward`), ingestion
(`numerical_match_reward`, `exploration_reward`), PRM (`hybrid_prm_reward`,
`distilled_judge_reward`), process (`tool_chain_reward`, `summary_written_reward`), and six
`finqa_*` variants.

`LLMJudge` supports hosted API, local transformers, or local vLLM judge backends.

Reward-hacking defenses (real code, verified):
- `RuleGuardCombinator` — deterministic score clamping (`min(score, 0.2)`) when a completion fails
  a cheap grounding-overlap check; **only fires if `retrieved_chunks` is threaded through**
- relative (batch) scoring and judge-prompt rotation in the judge path

### RAG and synthesis

- `agenttune/rag` — `SQLiteFTSBackend` / `ChromaBackend`, `SearchCorpusTool`, GRPO training
  against a retrieval reward
- `create_rag_trainer(...)` — one call builds/indexes the backend, wires the search tool +
  reward defaults, then delegates to `create_agentic_trainer`
- `agenttune.rag.synthesis` — 6-stage docs→multi-hop-QA pipeline: typed entity graph → path
  sampling → answer-first generation (MHTS) → closed-loop verification (retrieval necessity +
  chain dependency, 1 retry) → difficulty labelling (GRADE power-mean) → gold-chunk-disjoint GRPO-ready
  split. Metrics: answerability, faithfulness, recall@k, **multi-hop necessity**, lexical/
  structural/semantic diversity, coverage, difficulty distribution. Its README **reports**
  end-to-end validation on real models; this was not reproduced here. Gold-chunk isolation is
  real (`rag/synthesis/split.py`), but not a general benchmark-decontamination guarantee. Its
  separate "contamination gate" drops already-solved questions, not leaked benchmark prompts.

### Distillation

`create_distill_trainer(...)` — collect teacher trajectories (hand-authored `demonstrations=`,
live `teacher_engine=`+`tasks=`, or pre-built `trajectories=`), then behaviour-clone into a small
student with a built-in LoRA+TRL `SFTTrainer` factory (or your own `trainer_factory`). This is
**behaviour cloning**, not weight-KD.

### DECIDE (decision workflows as YAML)

YAML template → `GraphRunner` → `langgraph.StateGraph` with stage types `llm_call`, `router`,
`llm_judge`, `rules`, `tool_call`, plus audit logging, step-limit enforcement, and
`DestinationRouter`. DECIDE is independent of the training stack and shares only `EventLog` at the
boundary. Warning: `StageWiseRunner` silently ignores conditional routing; use `GraphRunner` for
branching templates.

### Self-healing closed loop

`detect → classify → generate → retrain → gate → deploy`:

- `FailureDetector` / `FailureClassifier` (e.g. `loop_collapse`, `tool_crash`)
- `TrainingExampleGenerator` → preference pairs via `derive_preference_from_completions`
- `RetrainingTrigger` → real DPO retrain through `create_agentic_trainer`
- `DeploymentGate` — historical task/optional trajectory-regression gate, not independent security verification
- `SelfHealLoop(classifier, generator, trainer_factory=None)` drives the whole loop; classify and
  generate stages are injectable, so detection is model-free

`DeploymentGate.build_test_set()` reconstructs **successful historical runs**, and
`score_model()` compares verdicts, not unauthorized final-state effects
(`deployment_gate.py:91–190,196–231`). `evaluate_decision()` skips the trajectory check when
either side lacks results and converts an empty task score to `0.0`; with no test cases and no
trajectory results it approves "no regression" without model calls (`:329–356`; CPU controls in
audit Appendix A.11). Require nonempty, independent
validation evidence before applying a decision. `Project.heal()` itself is failure **detection**,
not automatic retrain/deploy (`project.py:292–327`).

### Multi-agent orchestration

`AgentTuneGraph` (LangGraph) composes rollout nodes and LLM judges, compiles to a GRPO
`rollout_func` or `reward_func`.

---

## 4. Evaluation

`agenttune/eval`:

- `RLEvaluator`, `BaseEvaluator`, `run_eval` (use-case-specific tool-use agent scoring for
  `email_search`, `finqa`, `file_ingestion`, `generic`), `LMEvalRunner` (lm-eval CLI)
- Metrics: accuracy, perplexity, math accuracy, pass@k, ROUGE, BLEU, KL divergence, policy
  entropy, reward accuracy
- **Trajectory metrics** (programmatic, no LLM): TAC (tool-argument correctness via JSON Schema),
  ARR (API redundancy ratio), SCSR (success rate), RAD (reasoning-action divergence), LCF (loop
  count factor), PAS (plan adherence), PMED (path minimum edit distance), ASE (action-sequence
  efficiency). `None` vs `0.0` is deliberate for PAS/PMED/ASE ("not applicable" vs "scored zero").
- TAC only measures anything if `tool_schemas` is supplied to the evaluator constructor;
  with the default `{}`, dict-shaped calls count as unknown tools and TAC silently degrades.

These are diagnostics, not verified safety or task ground truth. For example,
`Project.evaluate()` defaults to substring matching in tool/text events (`project.py:67–76`),
and `evaluate_agentic()` exposes six programmatic metrics without injecting your tool schemas
(`:208–222`). Use an independently configured evaluator for schema-sensitive metrics and
AuditKIT oracles over genuine environment evidence for the AgentGuard headline.

---

## 5. Service, CLI, notebooks

- CLI: `agenttune train --algorithm ... --model ... --dataset ... [--reward-funcs ...]`,
  `agenttune pipeline --template ... --input ...`, `agenttune version`. **No `--tools` flag** —
  tool-calling training is Python-API only.
- FastAPI service: `uvicorn agenttune.agentic.service.app:create_app --factory` exposes the
  GPU-free spine path + lifecycle event stream (REST/WebSocket); GRPO/full self-heal report
  `wired-runs-on-gpu` instead of executing in-process.
- The README **reports 43 notebooks** (28 local + 15 use-case), plus **19 separate scripted
  walkthroughs**, and Colab mirrors of nine (`README.md:373`). These are not 62 notebooks or
  independently executed runs in this review; some older upstream pages still say 30.

---

## 6. Cohere / Tiny Aya specifics (source/docs checked; model behavior not run here)

- **Upstream reports no tool-rendering template for its reviewed Cohere models.** Tiny Aya has only a
  `default` template and renders `tool` turns empty; Aya Expanse/Aya Vision ignore `tools`; North
  Micro Vision's template drops tools/tool results.
- AgentTune detects this and falls back: it adds tool schemas to the system prompt
  (`tools_fallback_prompt`, asking for `[{"tool_name": ..., "parameters": {...}}]`) and passes
  tool results back as user turns (`tool_result_format`). It still parses Cohere's native
  `<|START_ACTION|>[...]<|END_ACTION|>` format (Command R7B) if emitted.
- Pass `tools_fallback_prompt=None` to raise instead, or set a tool-capable `chat_template`.
- Tiny Aya is **gated**; accept the license + `hf auth login`. The following is a model-backed
  training template, **not an executed CPU/Mac performance claim**:

```python
from datasets import Dataset
from agenttune.core.backend_factory import create_agentic_trainer

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

train_dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "What is 2+3?"}]}] * 8)
trainer = create_agentic_trainer(
    algorithm="grpo",
    backend="trl",
    model="CohereLabs/tiny-aya-global",
    train_dataset=train_dataset,
    tools=[add],
    reward_funcs=[lambda completions, **kw: [1.0 if "5" in str(c) else 0.0 for c in completions]],
    output_dir="./output/tiny_aya_tools",
    max_steps=10, per_device_train_batch_size=2, gradient_accumulation_steps=1,
    num_generations=2, max_steps_per_turn=3, use_vllm=False,
    peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    report_to="none",
)
trainer.train()
```

The substring reward above can be earned by answering "5" without using `add`. It illustrates
the API, not a tool-use or safety reward. Replace it with independently observed action/state
checks for the project, and measure actual tool-call parse/execution rates.

- **North Micro Vision full fine-tune**: pass `beta=0` (loading a second copy as the reference
  model fails in transformers); LoRA is unaffected.
- Output directory contract for AuditKIT: `trajectories.jsonl` (one line per rollout: `task`,
  `steps`, `final_response`, `metadata.conversation`; tool calls as
  `{id, name, arguments}`; `reward: null`), the model/adapter folder, and
  `lexsi_provenance.json` (with embedded CuratorKIT provenance when the dataset folder has it).
  This contract is documented for GRPO/RLOO tool runs; inspect the selected producer instead of
  assuming every algorithm emits identical files. The raw logs lack final-state snapshots and
  often stable case IDs. A real sidecar join is required for verified outcomes (see §10).

---

## 7. Known issues (from `docs/community/known-issues.md` and `docs/features.md`)

| Issue | Impact | Workaround |
|---|---|---|
| `ToolRegistry.get("read_file")` triggers `auto_register_builtins()`, which imports every builtin tool incl. `langchain_community`-dependent ones | ImportError for unrelated tools | install `langchain_community`, or construct tools directly |
| `DistilledJudge` silently returns all-zero reward if transformers/checkpoint load fails | silent zero training signal | check logs; avoid for critical rewards |
| `RougeMetric`/`BleuMetric` silently return `0.0` if `rouge_score`/`nltk` missing | meaningless eval | use trajectory metrics / AuditKIT |
| `OutputStage._route_to_destination` is a dead no-op | none (real routing elsewhere) | — |
| `StageWiseRunner` ignores all conditional routing | wrong pipeline topology | use `GraphRunner` |
| `AuditReader.extract_dpo_pairs` / `DecideToTrainerBridge.extract_bco_labels` look for fields `AuditWriter` never emits | empty training pairs | use `CollectRunner` from in-memory `PipelineState` |
| CLI cannot do tool-calling training (`--tools` missing) | feature gap | Python API |
| Registry-fetched `SQLDatabaseTool()` has no `db_uri` | tool fails cleanly with an error | construct with `SQLDatabaseTool(db_uri=...)` |
| Hybrid PRM default reward scores well-formed, malformed, and missing tool calls identically | reward signal gap | document/fix as a project contribution |
| GitHub/Slack/Playwright tools need credentials/packages not exercised | demos may break | avoid in the critical path |
| `RuleGuardCombinator` no-ops unless `retrieved_chunks` is passed | silent lack of grounding clamp | thread the chunks through |
| Qwen3-style reasoning traces need `enable_thinking=False` in RAG synthesis (`OpenAICompatLLMClient` auto-handles) | broken generation/solver with thinking on | verify |
| `DeploymentGate.evaluate_decision()` can approve an empty task set with absent trajectory evidence | "approved" without evaluation | require nonempty independent cases, complete coverage and a separate safety gate |
| Host code tools + `IsolatedTool` in-process fallback | host authority if exposed to untrusted actions | mock tools for the benchmark; explicit grants, stronger isolation and `require_isolation=True` where applicable |

---

## 8. Smoke tests

```bash
git clone https://github.com/Lexsi-Labs/AgentTune.git && cd AgentTune
pip install -e .
python -c "import agenttune; print(agenttune.__version__)"
pytest          # after pip install -e '.[dev]'
```

The following source-checked templates were not run with the full AgentTune installation here.

```python
# GPU-free spine quickstart with an installed package (no model, no network)
from agenttune.agentic import Project, DictToolHarness, ReActStrategy

harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=4)
strategy = ReActStrategy(lambda s: {"name": "add", "arguments": {"a": 2, "b": 3}}
                         if s.step == 0 else {"name": "finish", "arguments": {"answer": "5"}})
proj = Project(strategy=strategy, harness=harness)
proj.infer("what is 2+3?")
print(proj.evaluate([{"task": "what is 2+3?", "expected": "5"}]))
print(proj.evaluate_agentic(["what is 2+3?"]))   # TAC/ARR/SCSR/RAD/LCF...
```

```python
# Model-backed RAG training template; Qwen is a smoke subject, not a permitted final subject.
from agenttune import create_rag_trainer
trainer = create_rag_trainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    backend="trl",
    corpus=[{"doc_id": "policy", "title": "Founding",
             "text": "Northwind Robotics was founded in 2015 by Elena Cho."}],
    train_dataset=[{"prompt": "Who founded Northwind Robotics?", "answer": "Elena Cho"}],
    output_dir="./out_rag", max_steps=20,
)
trainer.train()
```

---

## 9. Special things to keep in mind

1. **The GPU-free spine is the reliable scripted demo surface.** `Project.infer/evaluate/evaluate_agentic`,
   `DemoRolloutEngine.collect_rollout`, `sft_dataset()`, and `heal()` detection run with no model
    and no network with suitable fakes. Label such a fallback scripted, not a real Tiny Aya result.
2. **`light` logs cannot train.** If you want to train/diagnose on a trajectory, collect it
   through a rollout engine, not `DictToolHarness`.
3. **AgentTune has a broad integration surface.** A long issue list is disclosure, not a measured
   maturity ranking; AlignTune/CircuitKIT have more Python files at these pins. Time-box wiring
   and use AuditKIT to verify supplied state/artifacts independently—an importer alone is not truth.
4. **The shipped Tiny Aya template needs the fallback prompt path unless replaced.** Budget
   schema tokens and measure parse/execution rate early; the existence of a fallback does not
   establish that the policy follows it.
5. **Unsloth backend is unofficial** here: install `unsloth==2026.8.19 --no-deps` +
   `unsloth_zoo>=2026.8.13 --no-deps` yourself; default to TRL.
6. **vLLM is Linux+CUDA only.** On Kaggle/Colab use `use_vllm=False` or the `api` rollout engine;
   on Modal L4/A10 with Linux, vLLM is the fast path.
7. **`max_steps_per_turn` bounds tool loops.** Without it, a bad policy can burn the whole
   generation budget in one rollout.
8. **Check the tier before projecting.** `EventLog` tier is set by the producer; a
   "trainable" claim on a light-tier log is a bug in your integration, not a library error.
9. **Distillation is behaviour cloning.** No logit matching; the student learns from the teacher's
   token spans/logprobs where full-tier logs exist.
10. **Provenance propagates**: if the training dataset is a CuratorKIT folder with
    `lexsi_provenance.json`, it is embedded under `inputs[].provenance` in AgentTune's own
     provenance record. This is nested metadata, not validated byte/hash binding; the writer's
     `git_sha` is `None`. Record revisions and artifact digests yourself.

---

## 10. Integration with the rest of the stack

| Hand-off | How |
|---|---|
| CuratorKIT → AgentTune | explicitly load/adapt the training-only messages/GRPO export to the selected trainer's schema; incompatible exports may be empty; preserve source provenance |
| AlignTune → AgentTune | load the full/merged SFT checkpoint or deliberately supported adapter path as the base policy for agentic RL |
| AgentTune → AuditKIT | import episodes, join genuine state/artifact sidecars by stable case/trial IDs, then `AgentEvalRunner().run(AgentEvalSpec(cases=..., episodes=...))`; score safety and utility separately |
| AgentTune → SafeTune | optional policy diagnostics/intervention; caller must implement live-wrapper/tool-authorization wiring and test it; no built-in AgentTune guardrail integration found |
| AgentTune → CircuitKIT | optional analysis on a supported full/merged policy checkpoint after Tiny Aya parity; causal safety claims require independent outcome checks |
| AgentTune → HF Hub | model/adapter + `trajectories.jsonl` + provenance; publish as a dataset too |

Recorded episodes use `episodes=`; deployed evaluation uses `agent=`, not `model=`. Put the
oracle on `AgentCase.outcome`. Raw rollout imports mark final state unavailable; do not fabricate
empty state, pair by row position, or promote source reward to verified success. The cookbook
in [`09-cross-tool-integration.md`](09-cross-tool-integration.md) covers the join and separate
`AgentEvalResult` persistence contract.

---

## 11. Verdict

**Use AgentTune when** your project is about making tool calls accurate, efficient, or reliable:
it is the only stack tool with agentic rollouts, trajectory metrics, RAG-agent training,
distillation, and a self-healing loop. The EventLog story is genuinely differentiating.

**Avoid unnecessary AgentTune scope when** the thesis is only text-level safety drift rather than
tool behavior. For this agentic project, keep a bounded tool world, a scripted demo fallback and
one working training path instead of implementing every lifecycle feature.

**For our registered Track 3 AgentGuard submission:** the thesis is not simply "train an agent"—it is
"**red-team and repair an agent's tool use**": find tool-call failures (wrong arguments,
redundant calls, unsafe tool choices, prompt-injection-driven tool use), generate corrective
trajectories, train a policy repair and test whether frozen task-level V-ASR falls while benign
utility survives. Lead with agentic fine-tuning; red-teaming supplies the threat model. Track 3 is
excepted from the two-track-repository minimum, but AgentTune + AuditKIT remains the recommended
pair. Do not promise improved results before the model-backed experiment runs.

> **Correction 2026-11-15 — the Track 3 framing above is void.** No Track 3 / AgentGuard submission
> exists and none is planned; no AgentTune work was ever run, and the Track 3 registration
> assumption is unconfirmed. The work actually completed is a **Track 2a-style abstention and
> hallucination baseline** (`part 1/baselines/benchmark.md`), which still needs a track tool chain
> (Rules §2) and a before-and-after arm (Rules §5). See `AGENTS.md` FINAL STATE.
