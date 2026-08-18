# Countdown RLVR Milestone 1 Baseline Note

## Scope

This note documents the final pre-training baseline status for Milestone 1 of the Countdown RLVR project.

Covered phases:

- Phase A: `0.5B` prompt hardening and strict `pass@1`
- Phase B: `0.5B` verifier-reranked best-of-N debugging and latent-capability check
- Phase C: `1.5B` main baseline evaluation on the frozen `200`-example set
- Phase D: initial `1.5B` GRPO smoke training and checkpoint selection

This note is intended to be the professional reference point for Milestone 1
baseline and first-pass GRPO smoke results.

## Official Artifacts

Use these files as the baseline record:

- `reports/eval_Qwen2.5-0.5B-Instruct_pass1_results.json`
- `reports/eval_Qwen2.5-0.5B-Instruct_best_of_4_results.json`
- `reports/eval_Qwen2.5-1.5B-Instruct_pass1_results.json`
- `reports/eval_Qwen2.5-1.5B-Instruct_best_of_8_results.json`

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

### Current checkpoint decision

Preserve the standalone `step200` artifacts as the most informative GRPO smoke artifacts.

Required preserved artifacts:

- full standalone `step200` model directories
- standalone `step200` pass@1 eval reports when available
- `run_config.json`
- generated training dataset JSONL
- training trace or notebook output
- exact code snapshot used for the run

The seed `42` and seed `123` standalone `step200` runs are the best observed checkpoints by pass@1. The seed `777` run is equally important scientifically because it matched baseline while still improving structural compliance. Do not treat standalone `step300` or the `300`-step sweep checkpoints as improved checkpoints. They improved structural compliance but did not improve the primary metric.

### Next experiment decision

The immediate reproducibility question has been partially answered at smoke scale:

- standalone `200`-step GRPO reached `5.50%` on two seeds
- standalone `200`-step GRPO matched the `4.50%` baseline on a third seed
- therefore, the structural effect is reliable, but the pass@1 improvement is not yet reliable

The next experiment should target the remaining bottleneck rather than simply increasing steps or repeating the same seed sweep.

Recommended next direction:

- keep standalone `max_steps=200` as the default schedule
- do not use the `300`-step schedule as the next default
- modify one factor at a time to improve arithmetic correctness

Most likely next intervention:

- curriculum or data shaping toward easier `k=3` and `k=4` tasks first
- continue evaluating on the full frozen `200`-example set

Reason:

- all observed gains are in `k=3` and `k=4`
- `k=5` remains at `0` correct
- the dominant remaining failure mode is `wrong_value`, not format error or forbidden literals
- reward shaping is now doing its structural job, so the next bottleneck is arithmetic search/reasoning

## Current Milestone Status

Milestone 1 has now moved beyond baseline evaluation into first-pass GRPO smoke training.

What is established:

- baseline eval was completed on the intended main model
- latent capability was demonstrated with best-of-8
- verifier reranking behaved as expected
- GRPO smoke training runs successfully on Kaggle `T4`
- revised reward shaping improves structural verifier compliance
- standalone `max_steps=200` is the most informative training recipe so far
- standalone `step200` improved pass@1 in `2/3` seeds and matched baseline in `1/3`

What remains unresolved:

- whether additional training can convert `wrong_value` failures into correct solutions
- whether `k=5` can be improved by this setup or needs a stronger curriculum/model/training recipe
- whether pass@1 gains can be made robust across seeds

Current recommendation:

- keep standalone `step200` artifacts as the current most informative smoke checkpoints
- treat the `5.50%` result as promising but not yet robust
- treat the seed `777` result as evidence that current GRPO mostly teaches verifier compliance, not reliably better arithmetic
- next, test targeted recipe changes for arithmetic correctness rather than longer training alone
