# Verifiable Agentic RL Lab

An end-to-end Countdown RLVR study using **Qwen2.5-1.5B-Instruct, LoRA, and
TRL GRPO with DAPO loss**, evaluated on a frozen 200-task held-out set.

- **Exact verification:** a safe AST-based symbolic verifier checks number
  usage and target equality using rational arithmetic.
- **Measured sampled capability:** the untouched base model solved **46/200
  tasks with verifier-selected best-of-8 (23.0%) versus 9/200 with greedy
  decoding (4.5%)** on the same frozen set. Best-of-8 uses eight sampled
  attempts per task and is an inference-time result.
- **Controlled training experiments:** shaped rewards, a balanced `k=3/k=4`
  curriculum, rollout-signal diagnostics, and a correctness-only ablation with
  two predeclared seeds and fixed decision gates.
- **Documented outcome:** the final treatment achieved **4.5% and 4.0% greedy
  pass@1**, establishing no reliable correctness improvement. Milestone 1
  closed as **mixed/unstable, with a negative primary correctness result**.

See the [experimental record](reports/countdown_rlvr_week1.md),
[frozen closure protocol](reports/countdown_milestone1_closure_plan.md), and
[paired task analysis](reports/countdown_milestone1_closure_analysis.json)
for configurations, raw evidence, limitations, and the stopping decision.

## Broader Project Scope

A small-budget research-engineering project for studying **verifiable RL fine-tuning** of language models, starting from simple exact-reward reasoning tasks and gradually scaling toward code, tool-use, RAG agents, and mini software-engineering agents.

The goal is to build the same core loop used in modern LLM post-training research:

```text
policy model
→ sample completions / trajectories
→ verify outputs or steps
→ convert verification into reward
→ optimize with SFT / DPO / RAFT / GRPO / RLFT
→ evaluate on held-out tasks
→ inspect failures
→ improve verifier, reward, data, and environment
```

The completed study uses **Countdown arithmetic RLVR**, with an exact symbolic
verifier and a small training budget. Natural-language math, code execution,
tool use, and agentic tasks are proposed later milestones; their implementation
and results are outside the completed study.

---

## Project Motivation

Modern reasoning and agentic LLMs are increasingly improved through **post-training** methods such as:

- supervised fine-tuning on high-quality trajectories,
- preference optimization,
- reward-ranked fine-tuning,
- reinforcement learning from human or AI feedback,
- reinforcement learning from verifiable rewards,
- process rewards for long-horizon reasoning,
- tool/environment feedback for agentic tasks.

For budget-constrained research engineering, the most practical way to understand this stack is not to start with a large agent benchmark. Instead, the project begins with small verifiable environments where the full training loop can be implemented, tested, evaluated, and debugged end to end.

This project is designed around the following principle:

```text
simple task, correct loop
→ harder task, same loop
→ richer verifier
→ longer trajectory
→ agent environment
→ process reward
→ agentic verifier
```

---

## Research Objective

The central research question is:

> How do verifiable rewards and verifier-guided RLFT improve small language models as we scale from simple reasoning tasks to long-horizon agentic tasks?

The project gradually investigates:

1. Whether a small model can improve on exact-reward reasoning tasks.
2. Whether best-of-N verifier reranking already solves much of the task.
3. Whether GRPO/RLFT improves pass@1 beyond sampling and reranking.
4. How reward hacking appears in simple verifiable tasks.
5. How natural-language reasoning differs from symbolic reasoning.
6. How execution feedback can train code models.
7. How tool-use trajectories can be trained and verified.
8. How RAG agents can be rewarded for grounded, cited answers.
9. How sparse outcome rewards eventually require denser process rewards.
10. How agentic verifiers can support long-horizon agent training.

---

## Completed Milestone

The completed milestone is:

```text
Milestone 1: Countdown RLVR Foundation
```

This milestone implements the following verifiable training and evaluation loop:

```text
Countdown task
→ safe symbolic verifier
→ scalar reward function
→ base model evaluation
→ best-of-N evaluation
→ GRPO smoke training
→ held-out evaluation
→ failure analysis
```

The **Countdown verifier** is the correctness authority throughout evaluation
and training. It uses restricted AST evaluation and exact rational arithmetic;
generated answers must satisfy both number-use constraints and target equality.

---

## Milestone 1 Status

**Milestone 1: Countdown RLVR is closed.** Its final classification is
**mixed/unstable, with a negative primary correctness result**.

Official evaluation artifacts:

```text
reports/countdown_rlvr_week1.md
reports/eval_Qwen2.5-0.5B-Instruct_pass1_results.json
reports/eval_Qwen2.5-0.5B-Instruct_best_of_4_results.json
reports/eval_Qwen2.5-1.5B-Instruct_pass1_results.json
reports/eval_Qwen2.5-1.5B-Instruct_best_of_8_results.json
reports/countdown_milestone1_closure_plan.md
reports/countdown_milestone1_closure_analysis.json
```

Results on the frozen `data/countdown_eval_200.json` set:

| Evaluation | Correct tasks | Success rate | Interpretation |
| --- | ---: | ---: | --- |
| Untouched base, greedy | 9 | 4.5% | Single-output reference |
| Untouched base, verifier-selected best-of-8 | 46 | 23.0% | Eight sampled attempts per task |
| Shaped GRPO, 200 steps, seeds 42 / 123 / 777 | 11 / 11 / 9 | 5.5% / 5.5% / 4.5% | Exploratory gains did not hold across all seeds |
| Balanced `k=3/k=4` GRPO, 200 steps, seed 42 | 8 | 4.0% | Curriculum treatment did not improve correctness |
| Correctness-only GRPO, 200 steps, seeds 42 / 777 | 9 / 8 | 4.5% / 4.0% | Neither seed met the predeclared paired-success gate |

The study measures a gap between capability recoverable by sampling and
capability this training setup reliably converts into greedy accuracy. The
200-task set and two closure seeds support a bounded conclusion. Historical
training-data identity could not be verified, and the planned contemporaneous
Kaggle base rerun was skipped; the report records these reproducibility limits.

Shaped training improved structural compliance more consistently than exact
correctness. `k=5` remained unsolved in the tested greedy GRPO evaluations. The
[experimental record](reports/countdown_rlvr_week1.md) contains per-difficulty
results, rollout diagnostics, and the stopping rationale.

---

## Countdown Task

The Countdown task is defined as follows:

> Given a list of numbers and a target value, generate an arithmetic expression that uses each provided number exactly once and evaluates to the target.

Example task:

```text
numbers = [1, 3, 4, 6]
target = 24
```

A valid expression is:

```text
6 / (1 - 3/4)
```

This expression evaluates to:

```text
6 / 0.25 = 24
```

It uses each number exactly once:

```text
1, 3, 4, 6
```

Allowed operators:

```text
+
-
*
/
()
```

Model completions use `<reasoning>` and `<answer>` tags. The evaluation path
extracts the answer expression before applying the strict verifier and records
format, syntax, number-use, and arithmetic failures separately.

---

## Verifier Design

The Countdown verifier must check:

1. The expression is non-empty.
2. The expression parses successfully.
3. The expression uses only safe arithmetic syntax.
4. The expression contains only integer literals from the provided numbers.
5. Each provided number is used exactly once.
6. No extra numbers are used.
7. Duplicate input numbers are handled correctly.
8. The expression evaluates exactly to the target.
9. Unsafe Python constructs are rejected.
10. Division by zero is handled safely.

The verifier must not use raw `eval`.

Instead, it should use Python `ast` parsing and a custom evaluator with exact rational arithmetic using `fractions.Fraction`.

Allowed AST nodes include:

```text
ast.Expression
ast.BinOp
ast.UnaryOp
ast.Constant
ast.Add
ast.Sub
ast.Mult
ast.Div
ast.UAdd
ast.USub
```

Rejected constructs include:

```text
function calls
variable names
attribute access
imports
lists
dicts
strings
comparisons
assignments
subscripts
```

Number usage should be checked with multiset comparison, not simple set comparison.

For example:

```text
numbers = [2, 2, 5, 10]
expression = (10 - 5) * (2 + 2)
target = 20
```

This is valid because two separate `2`s are provided and two `2`s are used.

---

## Reward Design

The default `shaped` preset assigns different rewards to verifier outcomes:

```text
1.0 = correct expression
0.3 = exact required numbers but wrong final value
0.05 = parseable expression with wrong number usage
0.0 = invalid, unsafe, empty, parse error, or division by zero
```

The earlier Phase A-C baseline used a simpler dense reward where wrong value
and wrong number usage both received `0.2`. During Phase D, that mapping was
changed after smoke training improved structural compliance without improving
exact task success. This observation motivated the change; it did not isolate
the original reward mapping as the cause.

The final `correctness_only` training preset gives `1.0` for correct expressions
and `0.0` for every incorrect outcome. It removes all partial credit and tests
that ablation, rather than isolating the wrong-value reward alone. Evaluation
reports retain shaped reward as a diagnostic; verifier correctness remains the
primary outcome.

`pass@1` and verifier status counts remain comparable across reports. Mean
reward should not be compared directly across runs that use different reward
mappings.

---

## Project Structure

```text
verifiable-agentic-rl-lab/
  pyproject.toml
  README.md
  data/        frozen Countdown eval sets
  models/      run configs and training reward diagnostics (no weights)
  reports/     evaluation results and the Milestone 1 closure protocol/analysis
  scripts/     dataset generation, base / best-of-N / GRPO evaluation,
               base-policy capability audit, GRPO training, closure analysis
  src/tasks/countdown/   verifier, reward, dataset, prompts, eval utilities
  tests/       verifier, reward, dataset, prompts, eval and training-script tests
```

---

## Development Setup

Install the project in editable mode:

```bash
pip install -e ".[dev]"
```

Install optional GCS artifact helpers:

```bash
pip install -e ".[gcs]"
```

Run tests:

```bash
pytest
```

Run only Countdown verifier tests:

```bash
pytest tests/test_countdown_verifier.py
```

Run only Countdown reward tests:

```bash
pytest tests/test_countdown_reward.py
```

Format and lint checks can be added with:

```bash
ruff check .
```

Type checking can be added with:

```bash
mypy src
```

---

## Implemented Components

### Step 1: Countdown Verifier

Implementation:

```text
src/tasks/countdown/verifier.py
```

This file provides:

```text
VerificationResult
verify_countdown_expression
safe AST validation
integer literal extraction
Fraction-based expression evaluation
number-usage checking
structured failure reasons
```

Expected verifier reasons:

```text
correct
empty_expression
parse_error
unsafe_expression
wrong_numbers
wrong_value
division_by_zero
```

### Step 2: Countdown Reward

Implementation:

```text
src/tasks/countdown/reward.py
```

This file converts verifier results into scalar rewards and exposes named
shaped and correctness-only presets.

### Step 3: Unit Tests

Implementation:

```text
tests/test_countdown_verifier.py
tests/test_countdown_reward.py
```

Tests cover:

```text
correct expression
wrong value
missing number
reused number
extra number
unsafe expression
variable name
function call
division by zero
duplicate input numbers
reward mapping
```

### Step 4: Baseline Evaluation

Base-model evaluation scripts:

```text
scripts/eval_base_model.py
scripts/eval_best_of_n.py
```

Metrics:

```text
pass@1
pass@4
pass@8
invalid format rate
average output length
failure categories
```

### Step 5: GRPO Smoke Training

Completed GRPO runs validated the training loop and tested held-out correctness,
structural behavior, reward treatments, and seed/checkpoint stability.

Diagnostics examined during the experiments:

```text
Does reward increase during training?
Does held-out pass@1 improve?
Does the model learn invalid shortcuts?
Does output length inflate?
Does train reward diverge from eval reward?
```

---

## Evaluation Philosophy

This project prioritizes honest evaluation over headline numbers.

Future studies may extend the comparisons with:

```text
base model
best-of-N with verifier
SFT or reward-ranked baseline when applicable
RLFT / GRPO model
RLFT / GRPO + best-of-N
```

Important metrics:

```text
pass@1
pass@k
invalid format rate
average output length
train reward
held-out reward
train/eval gap
verifier false positives
verifier false negatives
manual failure categories
```

Failure analysis is required for every milestone.

A run where reward increases but held-out accuracy does not improve is still useful if the failure mode is clearly diagnosed.

---

## Implemented and Proposed Comparison Methods

Milestone 1 implemented untouched-base greedy evaluation, verifier-selected
best-of-N, and GRPO trained-model greedy evaluation. RAFT, DPO, and trained-model
best-of-N below are proposed comparisons for future studies, with no completed
results claimed here.

### Base Model Evaluation

Evaluate the original model without fine-tuning.

Purpose:

```text
measure the starting capability of the model
```

### Best-of-N with Verifier Reranking

Sample multiple outputs from the model and use the verifier to select the best one.

Purpose:

```text
measure whether the model already has latent capability that can be extracted through sampling
```

This is an important baseline before RLFT because if best-of-N works well, the model may already know how to solve the task sometimes.

### Reward-Ranked Fine-Tuning / RAFT (Proposed)

Sample multiple outputs, score them with a verifier or reward function, keep the best outputs, and fine-tune on those selected outputs.

Purpose:

```text
learn from high-reward trajectories without running online RL
```

### Direct Preference Optimization / DPO (Proposed)

Use chosen and rejected outputs to train the model with a preference objective.

Purpose:

```text
provide a cheaper and more stable preference-learning baseline before RLFT
```

### GRPO / RLFT

Sample groups of outputs from the current policy, score them with a verifier or reward function, and update the model to make better-than-average outputs more likely.

Purpose:

```text
train the policy directly from verifiable rewards
```

### RLFT + Best-of-N (Proposed)

Evaluate whether the RL-trained model also benefits from verifier-based test-time sampling.

Purpose:

```text
separate policy improvement from inference-time scaling
```

For every major milestone, the preferred comparison is:

```text
base model
→ best-of-N
→ reward-ranked fine-tuning or DPO when applicable
→ GRPO / RLFT
→ GRPO / RLFT + best-of-N
```

---

## Foundation to Frontier Roadmap

Milestone 1 is complete. Milestones 2-7 describe proposed directions rather than
implemented systems or committed experiments.

### Milestone 1: Countdown RLVR

Exact symbolic arithmetic verifier.

Purpose:

```text
learn the core RLVR loop in a safe, cheap, controlled environment
```

### Milestone 2: GSM8K-Mini RLVR

Natural-language math with exact final-answer verification.

Purpose:

```text
move from symbolic expressions to language-based reasoning
```

### Milestone 3: Code RL with Unit Tests

Use execution feedback as reward.

Purpose:

```text
learn verifiable training for practical coding tasks
```

### Milestone 4: Tool-Use SFT and RL

Train trajectories involving tools such as:

```text
calculator
search_docs
read_doc
```

Purpose:

```text
teach tool-use behavior before long-horizon agent RL
```

### Milestone 5: Agentic RAG RL

Use a local document corpus and tools such as:

```text
search(query)
read(doc_id)
cite(doc_id)
answer
```

Rewards may include:

```text
answer correctness
citation correctness
evidence sufficiency
groundedness
penalty for unsupported answers
penalty for unnecessary tool spam
```

Purpose:

```text
connect verifiable RLFT to practical RAG and agentic AI systems
```

### Milestone 6: Mini SWE Agent RL

Use a small software repository with failing tests.

The agent can:

```text
inspect files
edit files
run tests
submit final patch
```

Reward:

```text
tests pass
```

Purpose:

```text
move toward realistic software-engineering agents without starting from full SWE-bench scale
```

### Milestone 7: Dense Process Rewards and Agentic Verifiers

Add rewards for intermediate actions, not only final success.

Examples:

```text
good search query
correct document read
valid tool argument
useful observation
evidence-supported final answer
test progress
final success
```

Purpose:

```text
study long-horizon credit assignment and frontier-style agentic RL
```

---

## What This Project Is Not

This project is not trying to reproduce large-lab reasoning models at full scale.

It does not start with:

```text
large models
huge datasets
full SWE-bench
complex multi-agent frameworks
learned reward models
process reward models
expensive API-generated data
```

Those may become useful later, but they are not the first milestone.

The project starts small because the goal is to understand and control the full RLFT loop.

---

## Research Principles

The project follows these principles:

1. **Verifier first.**
   Do not train until the verifier is trustworthy.

2. **Baselines before RL.**
   Always compare against base model and best-of-N verifier reranking.

3. **Small runs before large runs.**
   Every training setup should start with a smoke run.

4. **Held-out evaluation.**
   Training reward alone is not enough.

5. **Manual failure analysis.**
   Inspect failures and classify them.

6. **Reward hacking is expected.**
   Treat reward hacking as a signal to improve the verifier or reward.

7. **One reusable loop.**
   Each new task should reuse the same conceptual pipeline.

8. **Foundation to frontier.**
   Increase task complexity only after the current stage is understood.

---

## Status

Milestone 1 is closed as **mixed/unstable, with a negative primary correctness
result**. Both predeclared closure runs and their diagnostic checkpoints have
been evaluated. The frozen protocol's stopping conditions have been met.

Further training interventions would be a separate study. See the
[canonical experimental record](reports/countdown_rlvr_week1.md) for the evidence
and [closure protocol](reports/countdown_milestone1_closure_plan.md) for the
predeclared gates and post-execution annotation.

The verifier, dataset generator, reward presets, evaluation scripts, GRPO
training, and closure analysis are implemented. Milestone 2 (GSM8K-mini) and
Milestone 3 (code execution rewards) remain proposed and have not started in
this repository.

---

## Artifact and Secret Policy

Tracked:

```text
source code
tests
small frozen eval datasets
JSON/Markdown evaluation reports
project configuration
```

Not tracked:

```text
model checkpoints
LoRA adapter weights
virtual environments
Python caches
cloud service-account keys
local logs and scratch files
large compressed archives
```

Model artifacts should be stored externally, such as in GCS, and restored with
the checksum-verified helper scripts when needed.

---

## License

MIT License.
