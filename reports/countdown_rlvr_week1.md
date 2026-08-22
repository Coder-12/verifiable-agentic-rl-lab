# Countdown RLVR Milestone 1 Experimental Record

## Scope

This record documents the completed Milestone 1 Countdown RLVR study, from
baseline construction through the final bounded closure experiment.

Covered phases:

- Phase A: `0.5B` prompt hardening and strict `pass@1`
- Phase B: `0.5B` verifier-reranked best-of-N debugging and latent-capability check
- Phase C: `1.5B` main baseline evaluation on the frozen `200`-example set
- Phase D: initial `1.5B` GRPO smoke training and checkpoint selection
- Base-policy trainability audit under the exact GRPO rollout configuration
- Balanced `k=3/k=4` curriculum follow-up
- Frozen two-seed correctness-only closure experiment and stopping decision

This record is the canonical research and reproducibility reference for the
completed milestone.

## Official Artifacts

Use these files as the baseline record:

- `reports/eval_Qwen2.5-0.5B-Instruct_pass1_results.json`
- `reports/eval_Qwen2.5-0.5B-Instruct_best_of_4_results.json`
- `reports/eval_Qwen2.5-1.5B-Instruct_pass1_results.json`
- `reports/eval_Qwen2.5-1.5B-Instruct_best_of_8_results.json`
- `reports/eval_Qwen2.5-1.5B-Instruct_base_policy_capability_tasks200_g4_seed42.json`
- `reports/eval_Qwen2.5-1.5B-GRPO-k3k4-step200-seed42_pass1_results.json`
- `reports/eval_Qwen2.5-1.5B-GRPO-correctness-only-step200-seed42_pass1_results.json`
- `reports/eval_Qwen2.5-1.5B-GRPO-correctness-only-step200-seed777_pass1_results.json`
- `reports/countdown_milestone1_closure_analysis.json`
- `reports/countdown_milestone1_closure_plan.md`

Additional Phase D artifacts:

- `models/Qwen2.5-1.5B-GRPO-Smoke-step200/`
- `reports/eval_Qwen2.5-1.5B-GRPO-Smoke-step200_pass1_results.json`
- the saved training trace for the `step200` run
- additional standalone `step200` seed outputs captured from Kaggle notebook logs

Task definition:

- exact Countdown arithmetic verification
- strict rule-based verifier
- baseline reward mapping was fixed during Phase A-C
- Phase D uses revised reward shaping after the first GRPO smoke exposed an invalid-answer reward issue
- no answer-key matching against `expected_solution`

## Model Roles

### Debug model

`Qwen/Qwen2.5-0.5B-Instruct`

Use:

- prompt debugging
- evaluator debugging
- verifier integration checks
- best-of-N infrastructure checks
- quick failure analysis

### Main Milestone 1 model

`Qwen/Qwen2.5-1.5B-Instruct`

Use:

- main baseline evaluation
- meaningful best-of-N measurement
- Phase D training target

## Frozen Evaluation Sets

### Debug set

- `data/countdown_eval_50.json`

### Main set

- `data/countdown_eval_200.json`

These eval sets are treated as frozen baseline artifacts.

## Commands Used

### `0.5B` pass@1

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_base_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/countdown_eval_50.json \
  --batch-size 1
```

### `0.5B` best-of-4

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_best_of_n.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/countdown_eval_50.json \
  --batch-size 4 \
  --n 4 \
  --seed 42
```

### `1.5B` pass@1

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_base_model.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset data/countdown_eval_200.json \
  --batch-size 1
```

### `1.5B` best-of-8

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_best_of_n.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset data/countdown_eval_200.json \
  --batch-size 1 \
  --n 8 \
  --seed 42
```

### `1.5B` base-policy capability audit

```bash
python scripts/eval_base_policy_capability.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 200 \
  --num-generations 4 \
  --max-new-tokens 128 \
  --batch-size 4 \
  --dataset-seed 42 \
  --generation-seed 42
```

## Phase A and B Summary: `0.5B`

### `0.5B` pass@1

Source:

- `reports/eval_Qwen2.5-0.5B-Instruct_pass1_results.json`

Metrics:

- `pass@1 = 4.00%`
- `format_error_rate = 0.00%`
- `safe_parseable_rate = 100.00%`
- `exact_number_usage_rate = 26.00%`
- `mean_reward = 0.232`

Interpretation:

- prompt hardening fixed the dominant format problem
- the pipeline became trustworthy as a debug baseline
- the main remaining failure mode was still incorrect number usage

### `0.5B` best-of-4

Source:

- `reports/eval_Qwen2.5-0.5B-Instruct_best_of_4_results.json`

Configuration:

- `n = 4`
- `seed = 42`

Metrics:

- `oracle_pass@4 = 12.00%`
- `reranked_success@4 = 12.00%`
- `raw_completion_success_rate = 3.50%`
- `format_error_rate = 15.00%`
- `safe_parseable_rate = 81.00%`
- `exact_number_usage_rate = 22.50%`
- `mean_reward = 0.189`
- `selected_mean_reward = 0.296`

Interpretation:

- best-of-4 materially improved over greedy `pass@1`
- the `0.5B` model showed nontrivial latent capability under sampling
- verifier-guided reranking was aligned enough to recover correct samples when they appeared

Research conclusion for `0.5B`:

- good debug model
- weak direct solver
- useful for verifying the RLVR loop, not for judging final project success

## Phase C Summary: `1.5B`

### `1.5B` pass@1

Source:

- `reports/eval_Qwen2.5-1.5B-Instruct_pass1_results.json`

Metrics:

- `pass@1 = 4.50%`
- `format_error_rate = 9.50%`
- `safe_parseable_rate = 84.50%`
- `exact_number_usage_rate = 40.50%`
- `mean_reward = 0.205`

Status counts:

- `correct = 9`
- `wrong_numbers = 88`
- `wrong_value = 72`
- `format_error = 19`
- `parse_error = 2`
- `unsafe_expression = 10`

Interpretation:

- greedy performance is still weak
- constraint awareness is stronger than the `0.5B` debug baseline
- the model is clearly not solved by prompting alone
- this is a valid pre-training baseline for RLVR

### `1.5B` best-of-8

Source:

- `reports/eval_Qwen2.5-1.5B-Instruct_best_of_8_results.json`

Configuration:

- `n = 8`
- `seed = 42`

Metrics:

- `oracle_pass@8 = 23.00%`
- `reranked_success@8 = 23.00%`
- `raw_completion_success_rate = 4.3125%`
- `format_error_rate = 25.8125%`
- `safe_parseable_rate = 61.1875%`
- `exact_number_usage_rate = 27.00%`
- `mean_reward = 0.1569`
- `selected_mean_reward = 0.3830`

Selected-output status counts:

- `correct = 46`
- `wrong_value = 113`
- `wrong_numbers = 40`
- `parse_error = 1`

Global completion counts across all `1600` sampled completions:

- `correct = 69`
- `wrong_numbers = 547`
- `wrong_value = 363`
- `format_error = 413`
- `parse_error = 163`
- `unsafe_expression = 45`

Interpretation:

- best-of-8 improves over greedy `pass@1` from `4.5%` to `23.0%`
- this is `46/200` solved tasks versus `9/200` under greedy decoding
- the uplift is large enough to demonstrate real latent Countdown capability
- raw sampled completions are noisy, but verifier reranking is strong enough to select cleaner outputs
- on this task, oracle and reranked metrics are equal because any correct sample receives the top verifier reward and is selected

### Difficulty Breakdown

Observed best-of-8 solved tasks:

- `k=3`: `38` solved
- `k=4`: `8` solved
- `k=5`: `0` solved

Interpretation:

- the current model can often recover solutions on easier tasks under sampling
- the `k=5` setting remains the main difficulty frontier
- this is the clearest target for improvement during training

## Metric Definitions

### Greedy `pass@1`

Question answered:

- can the model solve the task in one deterministic attempt?

Meaning:

- measures direct baseline capability
- most important metric for post-training improvement

### Oracle `pass@N`

Question answered:

- did any of the `N` sampled completions solve the task?

Meaning:

- measures latent capability under sampling
- asks whether the model knows how to solve the task at all, even inconsistently

### Verifier-reranked `success@N`

Question answered:

- after scoring the `N` samples with the verifier-derived reward and selecting one final output, is that selected output correct?

Meaning:

- measures whether verifier-guided selection can recover good samples reliably
- especially important because Phase D training will optimize against the same verifier logic

### Base-policy rollout audit metrics

Question answered:

- under the exact GRPO rollout configuration, does the base policy expose enough reward-diverse and occasionally correct trajectories for RL to learn from?

Important distinction:

- this is not greedy `pass@1`
- this is not the official frozen-set best-of-N baseline
- this is an inference-only trainability diagnostic on fresh training-distribution tasks

Key metrics:

- `sample_1_success_rate`: correctness of the first sampled rollout in each group; useful as a one-sample stochastic reference, but not decisive for GRPO
- `correct_rollout_rate`: correct completions divided by all sampled completions
- `group_success_rate`: groups with at least one correct completion among the sampled rollouts
- `success_variance_group_rate`: groups containing a correct sample and at least one lower-reward sample
- `structural_only_variance_group_rate`: groups with reward variation but no correct sample

For GRPO trainability, `group_success_rate` and `success_variance_group_rate` are more important than `sample_1_success_rate`, because GRPO learns from within-group relative rewards.

## Main Research Conclusion Before Training

At the end of Phase C:

- the `1.5B` model is still weak in direct greedy solving
- the `1.5B` model has clear latent Countdown capability under sampling
- verifier-guided reranking is already aligned enough to extract that capability
- the dominant remaining limitation is model weakness, not evaluator failure
- the hardest unresolved region is `k=5`, not verifier correctness

This is exactly the kind of baseline picture needed before RLVR:

- the task is not already solved
- the verifier is useful
- there is recoverable capability for RL to sharpen

## Reproducibility Notes

- Treat the seeded reports as the official best-of-N baselines.
- Do not cite earlier unseeded exploratory results as official baselines.
- `pass@1` is deterministic for the current evaluation path.
- Best-of-N results should always be reported with the exact `seed` or seed list.
- The eval code was later updated to store fuller run configuration directly in report JSON payloads. That change improves future artifact hygiene but does not change the baseline numbers documented here.

### Historical reproducibility note: `0.5B` seed sweep

During the `0.5B` best-of-4 analysis, a later multi-seed sweep produced materially lower values than an earlier single seeded report. That discrepancy led to a reproducibility audit.

Key point:

- the discrepancy was treated as an engineering reproducibility issue, not as a model-improvement result

What was learned:

- MPS sampling behavior required more careful seeding discipline
- the evaluation code was then hardened around seeding behavior and future artifact logging

What remains official:

- the originally cited seeded `0.5B` best-of-4 report remains the baseline artifact for that phase
- the later discrepancy is important historical context, but it does not alter the role of `0.5B` as a debug and infrastructure model

Why this does not block Phase D:

- the main training-facing baseline is the `1.5B` model on the frozen `200`-example set
- the `0.5B` model was used primarily to validate evaluator, prompt, verifier, and best-of-N infrastructure

### Runtime and environment note for Phase C

The main `1.5B` Phase C best-of-8 baseline was practically infeasible on local Apple Silicon due to runtime cost, so the final run was moved to Kaggle on a `T4` GPU using reduced precision inference.

Operationally important context:

- local MPS was sufficient for verifier work, prompt debugging, and smaller evaluations
- local MPS was too slow for the full `1.5B` best-of-8 baseline on the frozen `200`-example set
- Kaggle `T4` made the run feasible while preserving the same evaluation code path and baseline definition

This matters for reproducibility because:

- the baseline numbers are scientifically comparable only if model, dataset, verifier, reward mapping, and decoding setup are preserved
- changing hardware for feasibility is acceptable here because the task definition and evaluation logic remained fixed

### Raw completion success versus oracle task success

The `1.5B` best-of-8 report gives:

- `raw_completion_success_rate = 4.3125%`
- `oracle_pass@8 = 23.0%`

These numbers answer different questions.

`raw_completion_success_rate` asks:

- out of all `1600` sampled completions, how many individual completions are correct?

`oracle_pass@8` asks:

- out of `200` tasks, how many tasks had at least one correct sample among the `8` sampled completions?

Important interpretation:

- `4.3125% * 8 = 34.5%` is not the oracle success rate
- that multiplication would only act as an upper bound if every correct sampled completion came from a different task
- in the actual run, `69` correct completions were concentrated inside `46` solved tasks

Independence interpretation:

- if each of the `8` samples behaved like an independent draw with per-sample success probability `0.043125`, the expected oracle success would be `1 - (1 - 0.043125)^8 ≈ 29.7%`
- the observed `23.0%` is below that independence expectation

Research meaning:

- the model clearly has latent capability
- sampled attempts are correlated within a task
- many tasks remain hard even when multiple samples are drawn
- this is consistent with a real but limited pre-training capability that RL may sharpen

### What the baseline does and does not prove

What it does prove:

- the verifier pipeline is usable
- the `1.5B` model is not already strong enough to make training unnecessary
- the model contains recoverable capability under sampling
- verifier-based reranking is aligned enough to recover correct samples when they exist

What it does not prove:

- that the first RL run will beat best-of-8 immediately
- that `k=5` difficulty will disappear after smoke training
- that every best-of-N uplift should be interpreted as independent-sample capability

This distinction is important. The current weakness profile should be understood primarily as model weakness, which is exactly the motivation for Phase D training.

## Base-Policy Trainability Audit

After the first GRPO smoke runs, an additional inference-only audit was run on the untouched `Qwen/Qwen2.5-1.5B-Instruct` base policy.

Purpose:

- diagnose whether direct GRPO has enough successful and reward-diverse rollouts to exploit
- avoid guessing whether weak GRPO improvement was caused by model capability, exploration, optimization, or task difficulty
- measure the starting policy under the same rollout settings used during GRPO training

Source:

- `reports/eval_Qwen2.5-1.5B-Instruct_base_policy_capability_tasks200_g4_seed42.json`

Configuration:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- dataset: fresh training-distribution Countdown tasks
- prompt: current locked Countdown prompt
- verifier: strict symbolic verifier
- reward mapping: current GRPO reward mapping
- prompt groups: `200`
- rollouts per prompt: `4`
- total completions: `800`
- decoding: `temperature = 0.7`, `top_p = 0.95`, `max_new_tokens = 128`
- dataset seed: `42`
- generation seed: `42`
- device used: local `mps`

### Overall Audit Result

| Metric | Value |
| --- | ---: |
| Sample-1 success rate | `1.50%` |
| Correct rollout rate | `3.12%` |
| Group success rate | `10.50%` |
| Verifier-reranked group success | `10.50%` |
| Nonzero reward-variance group rate | `91.50%` |
| Success-variance group rate | `10.50%` |
| Structural-only variance group rate | `81.00%` |
| All-zero group rate | `8.00%` |

Reward distribution across `800` sampled completions:

| Reward | Count | Meaning |
| ---: | ---: | --- |
| `1.0` | `25` | correct expression |
| `0.3` | `140` | exact numbers, wrong value |
| `0.05` | `221` | wrong number usage |
| `0.0` | `414` | invalid, unsafe, parse/format failure, or zero-reward failure |

Status counts across all sampled completions:

| Status | Count |
| --- | ---: |
| `format_error` | `367` |
| `wrong_numbers` | `221` |
| `wrong_value` | `140` |
| `parse_error` | `27` |
| `unsafe_expression` | `20` |
| `correct` | `25` |

Interpretation:

- the model exposes real correct trajectories under GRPO-style sampling, but they are sparse
- reward variance is abundant, but most of it is structural rather than success-based
- direct GRPO therefore has enough signal to improve compliance, but limited signal to reliably improve final correctness
- this explains why smoke GRPO improved formatting, parseability, and number usage more reliably than `pass@1`

### Audit Result by Puzzle Size

| Puzzle size | Groups | Correct rollout rate | Group success | Success variance | Structural-only variance | All-zero groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `k=3` | `67` | `7.46%` | `23.88%` | `23.88%` | `73.13%` | `2.99%` |
| `k=4` | `67` | `1.49%` | `5.97%` | `5.97%` | `82.09%` | `10.45%` |
| `k=5` | `66` | `0.38%` | `1.52%` | `1.52%` | `87.88%` | `10.61%` |

Additional by-size diagnostics:

| Puzzle size | Safe parseable | Exact number usage | Mean reward | Selected mean reward |
| --- | ---: | ---: | ---: | ---: |
| `k=3` | `61.94%` | `36.94%` | `0.176` | `0.425` |
| `k=4` | `45.52%` | `19.03%` | `0.081` | `0.232` |
| `k=5` | `37.12%` | `5.68%` | `0.035` | `0.108` |

Interpretation by difficulty:

- `k=3` is directly trainable with GRPO under the current rollout settings
- `k=4` has sparse but real success discovery and is a reasonable curriculum target
- `k=5` has almost no success discovery under `G=4`, making direct sparse GRPO unlikely to improve it reliably at this stage

### Relationship to Best-of-N and GRPO Smoke Results

The audit complements the earlier frozen-set best-of-8 baseline.

Best-of-8 showed:

- the base model has latent Countdown capability
- correct samples can be found for `k=3` and `k=4`
- `k=5` remained unsolved in the official best-of-8 baseline

The base-policy audit shows:

- under the actual GRPO rollout setting of `G=4`, success signal is much weaker than the best-of-8 headline might suggest
- `k=3` provides useful success-variance signal
- `k=4` provides limited but nonzero success-variance signal
- `k=5` provides almost no success-variance signal

This explains the observed GRPO smoke behavior:

- GRPO learned structural compliance because structural reward variance was common
- GRPO did not reliably improve `pass@1` because success-variance signal was sparse
- `k=5` did not improve because successful `k=5` rollouts were almost never discovered

Important caveat:

- length-limit diagnostics in this audit should be treated as approximate, because batched generation and padding can make length-limit flags conservative
- the correctness, reward, verifier status, and by-size capability metrics are the primary decision evidence

### Audit Conclusion at the Time

The current `1.5B` instruction-tuned model is a valid small-model starting point for Milestone 1, but direct full-distribution GRPO is not the highest-ROI next step.

Evidence-supported conclusion:

- direct GRPO is supported for `k=3`
- direct GRPO is weakly supported for `k=4`
- direct GRPO is not yet supported for `k=5` under the current rollout budget

At this point, the evidence supported a single balanced `k=3/k=4` curriculum
test while retaining the unchanged frozen 200-task evaluation set. The outcome
of that test and the subsequent closure experiment are recorded below.

## Phase D GRPO Smoke Training

### Training setup

Model:

- `Qwen/Qwen2.5-1.5B-Instruct`

Training method:

- GRPO through `trl.GRPOTrainer`
- LoRA adapter training
- single Kaggle `T4` GPU
- FP16 on T4-class hardware
- stochastic rollout generation during training
- strict rule-based Countdown verifier as the reward source

Training objective:

- improve greedy `pass@1` on the frozen `data/countdown_eval_200.json`
- preserve strict verifier behavior
- use smoke runs to validate whether the RL loop can improve the actual Countdown objective before scaling

### Reward shaping update

The first GRPO smoke run used the original dense reward mapping:

- `1.0`: correct
- `0.2`: wrong value or wrong number usage
- `0.0`: invalid, unsafe, parse error, empty expression, or division by zero

The `50`-step adapter exposed a reward-shaping weakness:

- `pass@1` regressed from `4.50%` to `3.00%`
- format and parseability improved
- exact number usage improved
- too-few-literal failures increased sharply

Interpretation:

- the model learned to produce cleaner verifier-compatible outputs
- however, it was still rewarded too similarly for invalid number usage and wrong arithmetic
- this encouraged partial compliance without improving exact task success

The Phase D reward mapping was then changed to:

- `1.0`: correct
- `0.3`: exact required numbers, wrong final value
- `0.05`: wrong number usage
- `0.0`: invalid, unsafe, parse error, empty expression, or division by zero

Rationale:

- reward exact number usage more than invalid number usage
- keep a small dense signal for parseable arithmetic
- avoid treating omitted or invented literals as nearly as useful as valid-but-wrong arithmetic

Important comparability note:

- `pass@1`, `correct`, and failure counts remain comparable across all reports
- `mean_reward` before and after the reward-shaping change is not directly comparable

### Standalone smoke run results

All trained checkpoints below were evaluated with greedy pass@1 on the frozen `200`-example evaluation set.

| Checkpoint | pass@1 | Correct | Format error | Safe parseable | Exact usage | Wrong value | Wrong numbers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `1.5B` | `4.50%` | `9` | `9.50%` | `84.50%` | `40.50%` | `72` | `88` |
| GRPO `step100` | `4.50%` | `9` | `2.50%` | `96.50%` | `51.00%` | `93` | `91` |
| GRPO `step200`, seed `42` | `5.50%` | `11` | `0.00%` | `99.00%` | `65.50%` | `120` | `67` |
| GRPO `step200`, seed `123` | `5.50%` | `11` | `0.00%` | `99.00%` | `61.50%` | `112` | `75` |
| GRPO `step200`, seed `777` | `4.50%` | `9` | `0.50%` | `98.00%` | `60.00%` | `111` | `76` |
| GRPO `step300` | `3.50%` | `7` | `0.50%` | `98.00%` | `61.50%` | `116` | `73` |

Additional literal diagnostics:

| Checkpoint | Too few literals | Forbidden literals |
| --- | ---: | ---: |
| Baseline `1.5B` | `41` | `54` |
| GRPO `step100` | `84` | `7` |
| GRPO `step200`, seed `42` | `61` | `6` |
| GRPO `step200`, seed `123` | `70` | `5` |
| GRPO `step200`, seed `777` | `72` | `4` |
| GRPO `step300` | `68` | `5` |

Puzzle-size success:

| Checkpoint | `k=3` correct | `k=4` correct | `k=5` correct |
| --- | ---: | ---: | ---: |
| Baseline `1.5B` | `7` | `2` | `0` |
| GRPO `step100` | `7` | `2` | `0` |
| GRPO `step200`, seed `42` | `8` | `3` | `0` |
| GRPO `step200`, seed `123` | `9` | `2` | `0` |
| GRPO `step200`, seed `777` | `7` | `2` | `0` |
| GRPO `step300` | `5` | `2` | `0` |

The seed `123` standalone `step200` JSON report was not preserved locally, but the full metric output was captured from the Kaggle notebook output and is documented here. The seed `777` standalone `step200` result was also captured from notebook output. This is sufficient for smoke-run research notes, while future confirmatory runs should preserve the JSON report artifact.

### Checkpoint sweep result

After the first two standalone `step200` runs improved over baseline, a checkpoint-sweep run was executed with:

- `max_steps = 300`
- `seed = 123`
- checkpoint saved every `50` steps
- checkpoints evaluated at `50`, `100`, `150`, `200`, `250`, and `300`

This sweep is not directly comparable to a standalone `max_steps=200` run at checkpoint `200`, because the cosine learning-rate schedule depends on the total configured `max_steps`.

In a standalone `200`-step run:

- step `200` is the end of the schedule
- the learning rate has decayed almost to zero

In a `300`-step sweep:

- checkpoint `200` is only two-thirds through the schedule
- the learning rate is still materially active

Sweep evaluation results:

| Sweep checkpoint | pass@1 | Correct | Format error | Safe parseable | Exact usage | Wrong value | Wrong numbers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-50` | `2.00%` | `4` | `4.00%` | `95.00%` | `45.50%` | `87` | `99` |
| `checkpoint-100` | `3.50%` | `7` | `0.50%` | `98.50%` | `50.50%` | `94` | `96` |
| `checkpoint-150` | `3.00%` | `6` | `0.00%` | `98.50%` | `51.00%` | `96` | `95` |
| `checkpoint-200` | `3.50%` | `7` | `0.00%` | `98.50%` | `54.00%` | `101` | `89` |
| `checkpoint-250` | `4.00%` | `8` | `0.50%` | `97.50%` | `57.50%` | `107` | `80` |
| `checkpoint-300` | `4.00%` | `8` | `0.50%` | `97.50%` | `57.50%` | `107` | `80` |

Sweep interpretation:

- no checkpoint in the `300`-step schedule beat the `4.50%` baseline
- the best sweep checkpoints reached only `4.00%`
- structural compliance still improved versus baseline
- longer training under this schedule did not convert more outputs into correct answers
- the standalone `max_steps=200` recipe remains better than the `300`-step schedule

Important conclusion:

- `checkpoint-200` inside a `300`-step run should not be used to refute the standalone `step200` result
- they are different optimization schedules, not identical training states

### Training trace interpretation

The standalone runs and the checkpoint sweep showed that the GRPO loop is operational:

- reward standard deviation was usually nonzero
- `frac_reward_zero_std` was often well below `1.0`
- LoRA gradients were nonzero in meaningful portions of training
- output length shortened after early rollouts
- clipping from `max_completion_length=128` became rare after the initial windows

Meaning:

- the verifier is producing usable within-group preference signal
- the policy is receiving update pressure
- the run is not merely executing infrastructure without learning signal

The training trace alone is not sufficient to judge success. The decisive metric remains frozen-set greedy `pass@1`.

### Main Phase D findings

The revised reward shaping improved structural compliance across standalone `step200` seeds:

- format errors dropped strongly
- safe parseability improved strongly
- exact number usage improved from the `40.50%` baseline to `60.00%`-`65.50%`
- forbidden literal introduction dropped from `54` baseline cases to `4`-`6`

The remaining dominant failure mode shifted toward arithmetic:

- standalone `step200` seeds had `111`-`120` wrong-value failures
- this means the model often obeyed the input-number constraint but still failed to compute the target expression

The best training recipe so far remains standalone `max_steps=200`, but the pass@1 improvement is not yet robust:

- `step100` matched baseline at `4.50%`
- standalone `step200` improved to `5.50%` for seeds `42` and `123`
- standalone `step200` matched the `4.50%` baseline for seed `777`
- `step300` regressed to `3.50%`
- the `300`-step checkpoint sweep did not beat baseline at any saved checkpoint

Research interpretation:

- Phase D has shown a real training signal
- GRPO can improve verifier compliance reliably
- there is smoke-scale evidence that standalone `max_steps=200` can improve greedy pass@1, but the gain is not seed-robust yet
- the `+2/200` gain appeared in `2/3` standalone seeds and disappeared in `1/3`
- this is promising engineering evidence, not a final training result
- training longer with the same schedule is not automatically better

The `k=5` frontier remains unsolved:

- baseline: `0` correct
- step100: `0` correct
- standalone step200 seed `42`: `0` correct
- standalone step200 seed `123`: `0` correct
- standalone step200 seed `777`: `0` correct
- step300: `0` correct
- 300-step sweep checkpoints: `0` correct

This confirms that the first GRPO runs mostly affect easier `k=3` and `k=4` tasks.

### Historical smoke checkpoint decision

Preserve the standalone `step200` artifacts as the most informative GRPO smoke artifacts.

Required preserved artifacts:

- full standalone `step200` model directories
- standalone `step200` pass@1 eval reports when available
- `run_config.json`
- generated training dataset JSONL
- training trace or notebook output
- exact code snapshot used for the run

The seed `42` and seed `123` standalone `step200` runs are the best observed checkpoints by pass@1. The seed `777` run is equally important scientifically because it matched baseline while still improving structural compliance. Do not treat standalone `step300` or the `300`-step sweep checkpoints as improved checkpoints. They improved structural compliance but did not improve the primary metric.

## Curriculum Follow-Up: Balanced `k=3/k=4`

### Question and controlled change

The rollout audit showed the strongest successful-rollout signal at `k=3`, a
sparse signal at `k=4`, and almost no signal at `k=5`. The next controlled
intervention therefore changed only the training-task distribution: 2,000
balanced `k=3/k=4` tasks, excluding `k=5`. The model, prompt, verifier, shaped
reward, LoRA/GRPO setup, 200-step horizon, and full frozen evaluation remained
unchanged.

The run used seed `42`, FP16 on a Kaggle T4, `G=4`, temperature `0.7`, top-p
`0.95`, and rollout length `128`. It was evaluated greedily on the unchanged
200-task `k=3/4/5` evaluation set.

### Result

| Run | Correct | pass@1 | Format error | Safe parseable | Exact usage | Wrong value | Wrong numbers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen base | `9` | `4.50%` | `9.50%` | `84.50%` | `40.50%` | `72` | `88` |
| Balanced `k=3/k=4`, step 200, seed `42` | `8` | `4.00%` | `1.00%` | `98.50%` | `62.00%` | `116` | `73` |

The curriculum run improved structural validity but not the primary metric. Its
correct answers were `6/85` for `k=3`, `2/65` for `k=4`, and `0/50` for `k=5`.
The full evaluation retained `k=5` intentionally; its absence from training is
not evidence that the curriculum should have solved it.

### Interpretation and decision

Balanced `k=3/k=4` training was not a positive result. It reinforced the
existing pattern: improved format/number compliance can coexist with dominant
`wrong_value` failures and no reliable gain in exact arithmetic correctness.
This did not show that curricula are generally ineffective. It only ruled out
this one distribution-only intervention as a sufficient answer within the
fixed 200-step setup. Together with the preceding base-policy rollout audit,
it motivated one final bounded reward test rather than another open-ended
curriculum or training-length change.

Artifact: `reports/eval_Qwen2.5-1.5B-GRPO-k3k4-step200-seed42_pass1_results.json`.

## Final Closure Experiment: Correctness-Only Partial-Credit Ablation

### Predeclared protocol

`reports/countdown_milestone1_closure_plan.md` was frozen before these runs.
It defined a bounded final treatment rather than an open reward search. The
only training treatment change was the reward mapping:

| Verifier outcome | Training reward |
| --- | ---: |
| `correct` | `1.0` |
| Every incorrect or invalid outcome | `0.0` |

The test is therefore a **correctness-only partial-credit ablation**. It does
not isolate the historical `wrong_value=0.3` reward alone and cannot prove that
this component caused prior behavior.

Two seeds, `42` and `777`, were predeclared and run regardless of the first
result. Both used the untouched `Qwen/Qwen2.5-1.5B-Instruct` base model; 2,000
fresh full-distribution `k=3/4/5` training tasks; LoRA (`r=16`, alpha `32`);
AdamW with learning rate `5e-6`, cosine schedule, and `0.1` warmup; batch size
`1` with accumulation `4`; DAPO loss with `beta=0`; `G=4`; group reward
scaling; temperature `0.7`; top-p `0.95`; 128-token rollouts; and a 200-step
horizon. Training and evaluation used FP16 on a Kaggle Tesla T4.

The retained checkpoints were steps 100 and 200. Step 200 was the sole primary
endpoint. Step 100 was evaluated only because the primary results were not
positive; it was diagnostic and was never eligible for checkpoint selection.
All checkpoints used greedy, strict-verifier evaluation on the frozen 200-task
set with `max_new_tokens=256`.

The generated training distributions were:

| Seed | `k=3` | `k=4` | `k=5` | Dataset SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `42` | `855` | `639` | `506` | `e54c8751615f835cf134947c2caa6bc58750a7cdd851028e97672a679c321478` |
| `777` | `888` | `638` | `474` | `592b95f6330a03d24a74d465f989166b5355f970884402dc505bfff12d918a45` |

The closure configuration resolved the model revision to
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The recorded environment was
Python `3.12.13`, PyTorch `2.10.0+cu128`, Transformers `5.0.0`, TRL `1.10.0`,
PEFT `0.19.1`, Datasets `5.0.0`, and Accelerate `1.13.0` on a Tesla T4
(compute capability `7.5`, CUDA runtime `12.8`). The two source-hash and full
configuration records are `models/run_config_closure_exp_seed42.json` and
`models/run_config_closure_exp_seed777.json`.

### Primary endpoint and frozen decision gates

The historical frozen base report contained `9/200` correct (`4.50%`). The
same-seed historical shaped references were `11/200` for seed `42` and `9/200`
for seed `777`. The predeclared paired-success thresholds were therefore 12
correct for seed `42` and 10 correct for seed `777`.

| Primary step-200 result | Base | Historical shaped reference | Paired-success threshold | Treatment | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Seed `42` | `9` | `11` | `12` | `9` (`4.50%`) | Fail |
| Seed `777` | `9` | `9` | `10` | `8` (`4.00%`) | Fail |

Neither seed was a paired success. Seed `42` matched base pass@1 but did not
exceed its shaped reference; seed `777` was below both. The seed-777 run also
violated every predeclared structural tolerance relative to the frozen base:
format error was `+9.0` percentage points, safe parseability was `-9.5` points,
and exact-number usage was `-8.0` points. Seed `42` improved all three
structural metrics relative to base but did not improve exact correctness.

| Checkpoint | Correct | pass@1 | Format error | Safe parseable | Exact usage | Wrong value | Wrong numbers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | `9` | `4.50%` | `9.50%` | `84.50%` | `40.50%` | `72` | `88` |
| Seed `42`, step 100 | `9` | `4.50%` | `3.00%` | `95.50%` | `50.00%` | `91` | `91` |
| Seed `42`, step 200 | `9` | `4.50%` | `2.50%` | `96.00%` | `47.50%` | `86` | `97` |
| Seed `777`, step 100 | `8` | `4.00%` | `12.00%` | `82.00%` | `36.50%` | `65` | `91` |
| Seed `777`, step 200 | `8` | `4.00%` | `18.50%` | `75.00%` | `32.50%` | `57` | `85` |

The evaluation `mean_reward` fields retain the legacy shaped diagnostic mapping
for comparability with prior evaluation reports. They are not the
correctness-only training objective and are not used to judge the treatment.

### Rollout signal and checkpoint diagnostics

The correctness-only runs did discover correct rollouts, so the protocol's
exploration-limited terminal condition was not met. However, usable groups were
rare under group-normalized reward:

| Training diagnostic | Seed `42` | Seed `777` |
| --- | ---: | ---: |
| Correct rollouts | `41/800` (`5.12%`) | `40/800` (`5.00%`) |
| Successful-variance groups | `26/200` (`13.00%`) | `34/200` (`17.00%`) |
| Zero-variance groups | `174/200` (`87.00%`) | `166/200` (`83.00%`) |
| All-zero groups | `173/200` (`86.50%`) | `166/200` (`83.00%`) |

Correct rollouts came almost entirely from `k=3`: `39/41` for seed `42` and
`36/40` for seed `777`. The remaining correct rollouts were `k=4`; neither run
produced a correct `k=5` rollout. This identifies sparse relative-success
signal as a plausible contributor to the outcome, but does not uniquely
separate sparse credit assignment from optimization, capacity, or other
implementation-specific effects.

Step-100 evaluation does not reveal a missed early optimum. Seed `42` had the
same nine correct tasks at steps 100 and 200. Seed `777` stayed at eight correct
tasks, with two gains and two losses, while structural quality worsened from
step 100 to step 200: format error increased from `12.0%` to `18.5%`, safe
parseability fell from `82.0%` to `75.0%`, exact usage fell from `36.5%` to
`32.5%`, and average output length increased from 330 to 373 characters.

### Paired held-out task analysis

The closure analysis compares each step-200 treatment report to the frozen base
report on the same 200 tasks. It is descriptive, not a population-level
significance test.

| Treatment | Gained | Lost | Both correct | Both incorrect | Exact sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed `42` | `6` | `6` | `3` | `185` | `1.0` |
| Seed `777` | `3` | `4` | `5` | `188` | `1.0` |

There were no `k=5` gains or losses for either seed. Seed `42` had `5` gains and
`4` losses at `k=3`, plus `1` gain and `2` losses at `k=4`; seed `777` had equal
`k=3` gains and losses (`3` each) and one `k=4` loss. These transitions provide
no directional evidence that the treatment improved the frozen task set.

### Final classification and stopping decision

**Milestone 1 is classified as mixed/unstable, with a negative primary
correctness result.**

The primary correctness result is negative because neither predeclared seed
met its paired-success gate despite usable successful-variance rollout signal.
The formal mixed/unstable classification additionally follows from the material
structural regression in seed `777`, which triggered the protocol's mixed branch.
This is not an exploration-limited result: correct rollouts and
success-variance groups occurred in both runs.

The frozen protocol permits no further reward adjustment, seed, training
extension, curriculum, SFT rescue, verifier change, or hyperparameter search.
Stopping therefore follows the predeclared decision rules, not a post-hoc
judgment about effort or outcome.

## Milestone 1 Synthesis

### Objective and progression

Milestone 1 built and tested a small exact-reward Countdown RLVR loop:
safe symbolic verification, frozen greedy and multi-sample evaluation, rollout
diagnostics, GRPO/DAPO LoRA training, failure analysis, a balanced curriculum
test, and a two-seed predeclared closure ablation. The experimental
progression was:

```text
strict verifier and frozen evaluation
-> base greedy and best-of-N capability
-> GRPO smoke runs and shaped reward behavior
-> base-policy rollout audit
-> balanced k=3/k=4 curriculum test
-> frozen correctness-only closure protocol
-> two-seed primary evaluation and diagnostic checkpoint analysis
```

### What succeeded

- A strict AST/Fraction Countdown verifier and reward pipeline were exercised
  end to end in generation, evaluation, and GRPO training.
- The frozen base evaluation established modest greedy performance (`9/200`) and
  substantial conditional sampled capability: best-of-8 reached `23.00%` on the
  frozen evaluation, while the GRPO-style rollout audit found correct samples
  in `21/200` prompt groups.
- Shaped GRPO consistently demonstrated behavioral effects on formatting,
  parseability, and number-use compliance, even though exact correctness was
  not robust across seeds.
- The curriculum and correctness-only interventions were controlled, evaluated
  on the same held-out set, and retained even when nonpositive.
- The closure treatment used two predeclared seeds, saved step-100 diagnostics,
  recorded rollout-status/group-variance evidence, and used paired task
  transitions rather than only aggregate pass@1.

### What did not succeed

The project did not establish a reliable improvement in greedy held-out exact
Countdown correctness for the tested 1.5B GRPO/DAPO/LoRA configuration. The
correctness-only partial-credit ablation tied the frozen base in one seed and
fell below it in the other; it did not outperform the historical shaped
references. It also did not improve `k=5` correctness.

### Bounded interpretation

The evidence distinguishes sampled capability from reliably trainable greedy
capability in this setup. Best-of-N and rollout audits show that the base policy
can sometimes discover correct expressions, particularly for `k=3`. The tested
GRPO configurations more consistently changed structural/compliance behavior
than exact arithmetic success. The final sparse reward retained some successful
groups but left most groups all-zero, and did not yield stable held-out greedy
improvement.

This does not establish that GRPO, RLVR, correctness-only reward, or the model
generally fails for reasoning. It also does not prove that model capacity,
exploration, optimization, or a single reward component is the unique
bottleneck. The conclusions apply only to this model, Countdown distribution,
verifier, rollout budget, LoRA/TRL GRPO/DAPO implementation, two seeds, and
200-step budget.

### Reproducibility and limitations

- The frozen evaluation set contains 200 tasks; the paired uncertainty values
  are descriptive and do not support population-level significance claims.
- The historical shaped-run training datasets were not retained. Closure runs
  regenerated same-seed data and recorded dataset SHA-256 values, but their
  byte identity to historical shaped data is not verified.
- The closure plan requested a contemporaneous Kaggle base rerun. The final
  analysis instead uses the existing frozen base report (`9/200`); this is a
  documented protocol deviation. It does not alter the nonpositive decision
  because neither treatment reaches its historical shaped threshold.
- Closure run configs record the model revision, package versions, hardware,
  source hashes, and dataset hashes. They do not contain a Git commit/dirty
  state or resolved tokenizer revision; those values are null.
- The repository retains the closure run configs, reward diagnostics,
  step-100 checkpoints, evaluation JSONs, and paired analysis. Full step-200
  adapters, generated training JSONL files, and complete historical shaped
  artifacts must remain archived externally with their recorded hashes if
  exact reruns are required.

### What carries forward

Later work can reuse the task generator, strict verifier, prompt/evaluation
stack, reward-preset interface, reproducibility metadata, rollout diagnostics,
and paired-report analysis. Any curriculum redesign, reward search, SFT/RAFT
intervention, larger model, altered rollout budget, or longer training is a new
experimental program and is outside this completed milestone.
