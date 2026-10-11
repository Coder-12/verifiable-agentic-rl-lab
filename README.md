# Verifiable Agentic RL Lab

**Research engineering for verifiable reasoning, reliable feedback, and agent post-training.** The implemented Countdown study examines what a small language model can learn from exact rewards; the broader roadmap targets execution-verified coding and interactive agent environments.

> **Current evidence:** Countdown RLVR Milestone 1 is **closed**. The training and verification pipeline was exercised end to end, but the final controlled experiments **did not establish a reliable improvement in held-out exact correctness**. Subsequent milestones are research directions, not implemented results.

## At a glance

| Area | Verified status |
| --- | --- |
| Exact symbolic verification | Implemented: restricted Python AST, exact rational arithmetic, multiset number-use checks |
| Dataset generation and evaluation | Implemented: frozen Countdown tasks, greedy evaluation, verifier-selected best-of-N, failure analysis |
| RLVR post-training | Implemented and exercised: Qwen2.5-1.5B-Instruct, LoRA, TRL GRPO trainer (DAPO loss configuration) |
| Experimental controls | Implemented: seeded evaluations, rollout diagnostics, two-seed closure protocol, saved reports |
| Code repair / tool-use / agent RL | **Planned or separately scoped; not implemented in this repository** |

**Core principle:** an increase in training reward, output validity, or test-suite pass rate is not automatically an increase in task-solving ability. Claims are determined by controlled, independent evaluation.

## Milestone 1 — Verifiable Symbolic Reasoning & RLVR (closed)

**Task.** Given a set of integers and a target, generate an arithmetic expression that uses each provided number exactly once and evaluates to the target. The verifier uses a restricted AST interpreter with `fractions.Fraction`; it does not execute model output with raw `eval`.

**Experimental setup.** The primary policy was `Qwen/Qwen2.5-1.5B-Instruct`, evaluated on a frozen set of **200 tasks**. The study compared the untouched policy, stochastic sampling with verifier reranking, shaped-reward GRPO experiments, a bounded curriculum intervention, and a predeclared correctness-only reward ablation. Training used LoRA with TRL's GRPO trainer configured with DAPO loss.

### Measured results

| Evaluation | Correct / 200 | Exact success | Interpretation |
| --- | ---: | ---: | --- |
| Untouched policy, greedy | 9 | **4.5%** | Single-output baseline |
| Untouched policy, verifier-selected best-of-8 | 46 | **23.0%** | Inference-time selection from eight sampled attempts; **not** a trained-model gain |
| Shaped GRPO, 200 steps (seeds 42 / 123 / 777) | 11 / 11 / 9 | 5.5% / 5.5% / 4.5% | Exploratory changes; no seed-robust accuracy gain |
| Balanced `k=3/k=4` curriculum GRPO, seed 42 | 8 | 4.0% | Did not improve held-out correctness |
| Correctness-only GRPO closure, seeds 42 / 777 | 9 / 8 | **4.5% / 4.0%** | Neither run met its predeclared success gate |

**Final classification: mixed/unstable; negative primary correctness result.**

The untouched model sometimes discovered valid solutions under sampling, but this did not translate into reliable greedy correctness improvement under the tested RLVR configurations. Shaped rewards more consistently improved formatting, parseability, and number-use compliance than final arithmetic accuracy. The binary-reward closure had relatively few informative successful rollout groups.

These findings apply to the tested model, task distribution, verifier, training setup, seeds, and compute horizon. They do **not** show that GRPO generally fails, or identify a unique cause of the negative result.

**Primary evidence**

- [Full experimental record](reports/countdown_rlvr_week1.md) — protocol, measurements, failure analysis, configurations, and limitations.
- [Predeclared closure protocol](reports/countdown_milestone1_closure_plan.md) — primary endpoint, decision rules, and stopping criteria.
- [Paired closure analysis](reports/countdown_milestone1_closure_analysis.json) — task-level gains, losses, and diagnostics.
- [Frozen evaluation set](data/countdown_eval_200.json) — the 200-task held-out reference used for this milestone.

## Implemented research pipeline

```text
Countdown task generator
    ↓
Prompt formatting → model completions
    ↓
Safe expression extraction + AST/Fraction verifier
    ↓
Structured failure codes + scalar reward
    ↓
Greedy / best-of-N evaluation  OR  LoRA GRPO training
    ↓
Held-out verification, per-task failure analysis, artifact reporting
```

The verifier checks permitted syntax, exact use of the supplied numbers (including duplicates), safe arithmetic, and equality with the target. Invalid syntax, disallowed operations, incorrect number usage, wrong value, and division by zero are distinct failure types.

Key implemented components:

| Location | Responsibility |
| --- | --- |
| `src/tasks/countdown/` | Task generation, prompts, verifier, and reward presets |
| `scripts/eval_base_model.py` | Greedy baseline evaluation |
| `scripts/eval_best_of_n.py` | Sample-and-verify best-of-N evaluation |
| `scripts/eval_base_policy_capability.py` | Training-sampler rollout audit |
| `scripts/train_grpo_countdown.py` | LoRA GRPO experiment runner |
| `scripts/eval_grpo_model.py` | Post-training evaluation |
| `tests/` | Verifier and reward checks |
| `reports/` | Preserved measurements and closure artifacts |

## Reproducing the foundations

Requires **Python 3.10+**. GPU-backed model evaluation and training require compatible PyTorch / CUDA or other supported acceleration; exact historical run environments are documented in the experiment record.

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Evaluate the original 1.5B policy on the frozen set:

```bash
python scripts/eval_base_model.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset data/countdown_eval_200.json \
  --batch-size 1 \
  --max-new-tokens 256 \
  --report-path reports/local_Qwen2.5-1.5B-Instruct_pass1_results.json
```

Sample eight completions per task with verifier-based selection:

Run this example in a disposable checkout or preserve the existing report
first: the script writes to the fixed path
`reports/eval_Qwen2.5-1.5B-Instruct_best_of_8_results.json`, replacing that file.

```bash
python scripts/eval_best_of_n.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset data/countdown_eval_200.json \
  --batch-size 1 --n 8 --seed 42
```

For training commands, pinned model revisions, historical package versions, seeds, dataset hashes, reward presets, and checkpoint provenance, use the [experimental record](reports/countdown_rlvr_week1.md) and saved run configurations rather than assuming a new local run is byte-identical to the original.

### Reproducibility limits

The main held-out sample contains 200 tasks. The final correctness-only closure used two predeclared seeds; historical shaped-run datasets were not all retained, and a contemporaneous base rerun specified in the closure plan was not completed. These caveats are documented, not silently resolved. Model checkpoints and large run artifacts may be held outside Git, with recorded identities where available.

## Research roadmap — completed work and future directions

The original seven-stage structure remains, with titles standardized around research capabilities rather than individual benchmarks or training algorithms. **Only Milestone 1 has verified results in this repository.** Future stages are directions, not claims of implemented systems, and execution order is determined by evidence and feasibility.

| Stage | Research focus | Status |
| --- | --- | --- |
| **1. Verifiable Symbolic Reasoning & RLVR** | Exact arithmetic verification, rollout analysis, controlled LoRA GRPO evaluation | **Closed — negative primary correctness result** |
| **2. Outcome-Verified Mathematical Reasoning** | Natural-language mathematical reasoning and exact outcome evaluation | Deferred; not implemented |
| **3. Execution-Verified Code Repair & Agent Training** | Real Python defects, isolated execution, independent tests, SFT and conditional RL | **Prioritized as a separate project; feasibility stage** |
| **4. Tool-Use Policy Learning & Trajectory Evaluation** | Constrained tool interaction, behavioral adaptation, trajectory evaluation | Proposed |
| **5. Evidence-Grounded Retrieval Agent Training** | Retrieval actions, evidence sufficiency, citation and answer verification | Proposed |
| **6. Long-Horizon Repository Repair Agents** | Multi-step, repository-level debugging and repair beyond bounded patch tasks | Proposed |
| **7. Process-Level Credit Assignment & Agentic Verification** | Intermediate feedback, reliable verifiers, and long-horizon attribution | Proposed |

### Prioritized next project — Execution-Verified Code Repair & Agent Training

This separately scoped project advances **the intent of original Milestone 3** without claiming that this repository already supports real-repository execution or agent training.

Its planned progression is:

```text
Authentic reproducible Python bugs
→ isolated patch execution + independent acceptance tests
→ untouched-model and test-feedback baselines
→ verified-demonstration SFT
→ conditional RLVR only if reward signal and budget justify it
→ held-out functional repair evaluation + reproducible report
```

This is an **engineering and research plan, not a performance claim**. The initial gate is to validate real bug sources, reference repairs, test quality, environment costs, and execution security. Any code-execution harness belongs in a separately validated implementation repository; it will be linked here once it exists and is ready for review. The long-horizon repository repair milestone (Stage 6) remains a later, more autonomous objective.

## Research and safety standards

- **Verifier first:** evaluate reward correctness and failure modes before optimization.
- **Baselines before training:** compare untouched-policy and inference-time alternatives before attributing value to SFT or RL.
- **Independent evaluation:** protect frozen held-out sets from training and prompt selection.
- **Execution isolation:** do not run model-generated code in an untrusted host environment. A virtual environment alone is not a security boundary.
- **Budget-aware research:** small pilot runs, explicit stopping rules, and cost reporting before scaling.
- **Honest claims:** report failed runs, seed variation, negative findings, and limitations; separate engineering completion from research novelty.

## Artifacts and license

Source, tests, small frozen evaluation sets, experiment metadata, and technical reports belong in version control. Model weights, large checkpoints, caches, credentials, and private infrastructure secrets do not. Never commit service-account keys or tokens.

Licensed under **MIT** as declared by the project metadata.

---

**Project philosophy:** Build reliable verifiers and experiments first; advance to harder reasoning and agent tasks only when the evidence justifies the cost and complexity.
