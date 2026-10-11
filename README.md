# Verifiable Agentic RL Lab

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

This project starts with a controlled **Countdown arithmetic RLVR** setup because it has an exact symbolic verifier, low compute requirements, and clear failure modes. The same framework will later extend to natural-language math, code execution rewards, tool-use agents, agentic RAG, and mini SWE-style tasks.

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

## Current Milestone

The current milestone is:

```text
Milestone 1: Countdown RLVR Foundation
```

This milestone builds the smallest complete verifiable RLFT loop:

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

The first implementation step is the **Countdown verifier**.

Before training any model, the project first ensures that the verifier is safe, strict, and well-tested. In RLVR, the verifier is the reward oracle. A weak verifier produces fake progress.

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

Main `1.5B` baseline results on the frozen `data/countdown_eval_200.json` set:

```text
Qwen/Qwen2.5-1.5B-Instruct greedy pass@1:     4.50%
Qwen/Qwen2.5-1.5B-Instruct best-of-8:        23.00%
```

Historical shaped-GRPO smoke result:

```text
Qwen2.5-1.5B GRPO standalone step200 pass@1: 5.50%
```

Final interpretation:

```text
best-of-8 shows nontrivial sampled latent capability
shaped GRPO improved structural verifier compliance more reliably than exact correctness
the balanced k=3/k=4 curriculum reached 4.00% greedy pass@1
the predeclared two-seed correctness-only ablation reached 4.50% and 4.00%
neither closure seed met its paired held-out correctness gate
k=5 remained unsolved in the tested greedy GRPO evaluations
```

The detailed evidence, decision gates, limitations, and stopping rationale are
maintained in:

```text
reports/countdown_rlvr_week1.md
```

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

The first version expects clean expression strings. Handling messy model outputs such as `"The answer is ..."` will be added during model evaluation.

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

The current Countdown reward function is intentionally strict but shaped enough
to avoid treating different failure modes as equally useful:

```text
1.0 = correct expression
0.3 = exact required numbers but wrong final value
0.05 = parseable expression with wrong number usage
0.0 = invalid, unsafe, empty, parse error, or division by zero
```

The earlier Phase A-C baseline used a simpler dense reward where wrong value
and wrong number usage both received `0.2`. During Phase D, that mapping was
changed after smoke training showed that rewarding wrong number usage too much
encouraged partial compliance without improving exact task success.

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

## Initial Implementation Plan

### Step 1: Countdown Verifier

Implement:

```text
src/tasks/countdown/verifier.py
```

This file should provide:

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

Implement:

```text
src/tasks/countdown/reward.py
```

This file should convert verifier results into scalar rewards.

### Step 3: Unit Tests

Implement:

```text
tests/test_countdown_verifier.py
tests/test_countdown_reward.py
```

Tests should cover:

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

After the verifier is tested, add scripts to evaluate a base model:

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

After baseline evaluation works, run a small GRPO training experiment.

The first training objective is not to achieve state-of-the-art performance. It is to verify that the full RLFT loop runs correctly.

Questions to answer:

```text
Does reward increase during training?
Does held-out pass@1 improve?
Does best-of-N still outperform the trained policy?
Does the model learn invalid shortcuts?
Does output length inflate?
Does train reward diverge from eval reward?
```

---

## Evaluation Philosophy

This project prioritizes honest evaluation over headline numbers.

Every milestone should compare:

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

## Baseline Methods Covered

This project evaluates reinforcement learning against progressively stronger non-RL baselines. Online RLFT is introduced only after those baselines are established, and its gains are measured relative to them.

The main baseline methods are:

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

### Reward-Ranked Fine-Tuning / RAFT

Sample multiple outputs, score them with a verifier or reward function, keep the best outputs, and fine-tune on those selected outputs.

Purpose:

```text
learn from high-reward trajectories without running online RL
```

### Direct Preference Optimization / DPO

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

### RLFT + Best-of-N

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

This project follows a staged roadmap.

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

```text
Milestone 1 (Countdown RLVR): closed, mixed/unstable with a negative primary correctness result.
Verifier, reward, dataset generator, base/best-of-N/GRPO evaluation and closure analysis: implemented and tested.
Next: Milestone 2 (GSM8K-mini) and Milestone 3 (code execution rewards). Not yet started in this repository.
```

Findings carried forward into the next milestones:

```text
best-of-N (23.0%) far exceeds greedy pass@1 (4.5%), so latent capability exists but is not being converted by this GRPO setup
shaped rewards raised structural compliance more than exact correctness
any new reward shaping is tested against a pre-declared protocol and frozen held-out set
```

For the full evidence trail and decision logic, read:

```text
reports/countdown_rlvr_week1.md
reports/countdown_milestone1_closure_plan.md
```

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
