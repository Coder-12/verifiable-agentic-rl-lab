# Milestone 1 Closure Protocol

## Status and Scope

This is the authoritative **pre-implementation** protocol for the final bounded
Milestone 1 experiment. It preserves the historical record in
`countdown_rlvr_week1.md` and must not be revised in response to the treatment
results. The goal is a fair test of one reward-design hypothesis, not a search
for a favourable result.

The experiment remains a small-scale, budget-constrained Countdown RLVR study:
an instruction-tuned `Qwen/Qwen2.5-1.5B-Instruct` base policy, strict symbolic
verification, LoRA, and TRL's `GRPOTrainer` using `loss_type="dapo"` and
`beta=0`.

## Evidence Leading to This Test

The frozen 200-task greedy base result was `9/200` correct (`4.50%`). Best-of-N
and the base-policy capability audit established genuine but uneven latent
capability: sampled correct trajectories are commonest for `k=3`, sparse for
`k=4`, and almost absent for `k=5` under the `G=4` rollout regime. The audit
found `25/800` correct rollouts, `21/200` successful groups, and `183/200`
groups with reward variation; most reward variation was structural rather than
success-based.

Earlier shaped-reward standalone GRPO runs at 200 steps produced `11/200`
correct for seed `42`, `11/200` for seed `123`, and `9/200` for seed `777`.
They improved structure more consistently than arithmetic correctness, with
`wrong_value` the dominant held-out failure category. A balanced `k=3/k=4`
curriculum run reached only `8/200`; it is not a positive training result.

This protocol therefore tests **correctness-only partial-credit ablation**. It
does not isolate `wrong_value=0.3` alone, nor prove that partial credit caused
the prior behavior.

## Frozen Treatment and Controls

Only the training reward mapping changes:

| Verifier outcome | Treatment reward |
| --- | ---: |
| `correct` | `1.0` |
| every incorrect or invalid outcome | `0.0` |

The strict verifier, prompt, answer extraction, generator rules, and frozen
evaluation semantics remain unchanged. The fixed conditions are:

- untouched base model: `Qwen/Qwen2.5-1.5B-Instruct`
- training data: 2,000 newly generated tasks, full `k=3/4/5` distribution,
  `min_numbers=3`, `max_numbers=5`, `balanced_by_k=false`
- LoRA: `r=16`, alpha `32`, standard seven Qwen projection modules
- optimizer/schedule: `adamw_torch`, learning rate `5e-6`, cosine schedule,
  warmup ratio `0.1`, batch size `1`, accumulation `4`
- GRPO rollout setup: `G=4`, temperature `0.7`, top-p `0.95`, maximum
  completion length `128`, `scale_rewards="group"`, DAPO loss, `beta=0`
- hardware: Kaggle T4 in FP16
- training horizon: 200 steps, with retained checkpoints at steps 100 and 200
- held-out evaluation: frozen `data/countdown_eval_200.json`, greedy decoding,
  batch size `1`, `max_new_tokens=256`, strict verifier.

The predeclared treatment seeds are `42` and `777`; both are required regardless
of the first result. Historical shaped runs are context only. Their original
`train_dataset.jsonl` artifacts are unavailable, so the comparisons are
same-seed reconstructed comparisons, not byte-verified historical pairings.

## Required Sequence

1. Re-evaluate the untouched base model once on the Kaggle T4 using the current
   greedy evaluator and a unique report path.
2. Run correctness-only GRPO seed `42`; save checkpoints at 100 and 200 steps;
   evaluate step 200.
3. Run the same treatment for seed `777`; save checkpoints at 100 and 200
   steps; evaluate step 200.
4. Evaluate both step-100 checkpoints only when the step-200 outcome is mixed,
   negative, or exploration-limited. Step 100 is diagnostic only and cannot
   replace the step-200 endpoint or justify an extension.
5. Run the analysis utility on the contemporaneous base and both treatment
   reports. It produces paired task transitions, per-k transitions, and a
   descriptive exact sign-test uncertainty value.

The base evaluation command is:

```bash
python scripts/eval_base_model.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset data/countdown_eval_200.json \
  --batch-size 1 \
  --max-new-tokens 256 \
  --report-path reports/eval_Qwen2.5-1.5B-Instruct_kaggle_t4_closure_base_pass1_results.json
```

For each seed, run:

```bash
python scripts/train_grpo_countdown.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-samples 2000 \
  --output-dir models/Qwen2.5-1.5B-GRPO-correctness-only-step200-seedSEED \
  --max-steps 200 \
  --seed SEED \
  --max-completion-length 128 \
  --reward-preset correctness_only \
  --save-steps 100 \
  --save-total-limit 2

python scripts/eval_grpo_model.py \
  --model models/Qwen2.5-1.5B-GRPO-correctness-only-step200-seedSEED \
  --dataset data/countdown_eval_200.json \
  --batch-size 1 \
  --max-new-tokens 256
```

If step-100 diagnostics are required by the decision rules, preserve both
checkpoint reports explicitly:

```bash
python scripts/eval_grpo_model.py \
  --model models/Qwen2.5-1.5B-GRPO-correctness-only-step200-seedSEED/checkpoint-100 \
  --dataset data/countdown_eval_200.json \
  --batch-size 1 \
  --max-new-tokens 256 \
  --report-path reports/eval_Qwen2.5-1.5B-GRPO-correctness-only-step100-seedSEED_pass1_results.json
```

After both primary evaluations, run:

```bash
python scripts/analyze_milestone1_closure.py \
  --base-report reports/eval_Qwen2.5-1.5B-Instruct_kaggle_t4_closure_base_pass1_results.json \
  --treatment-report seed42=reports/eval_Qwen2.5-1.5B-GRPO-correctness-only-step200-seed42_pass1_results.json \
  --treatment-report seed777=reports/eval_Qwen2.5-1.5B-GRPO-correctness-only-step200-seed777_pass1_results.json \
  --output reports/countdown_milestone1_closure_analysis.json
```

## Required Artifacts

Every treatment directory must contain `train_dataset.jsonl`, its SHA-256,
source hashes, observed k distribution, reward preset/mapping, resolved model
and tokenizer revisions when available, git state, package versions, T4 metadata,
full GRPO/LoRA settings, `training_reward_diagnostics.json`, trainer log history,
and both checkpoints. Rollout diagnostics must include verifier-status counts by
k, correct rollout count, successful-variance group count, zero-variance group
rate, and all-zero group rate.

## Predeclared Decision Rules

Let `B` be the contemporaneous base correct count on the frozen 200-task set.
Historical shaped reference counts are 11 for seed 42 and 9 for seed 777. A
treatment seed is a **paired success** only when it strictly exceeds both its
same-seed shaped reference and `B`:

- seed 42: at least `max(11, B) + 1` correct
- seed 777: at least `max(9, B) + 1` correct

A positive Milestone 1 outcome requires both paired successes at step 200,
usable successful-variance rollout signal in both runs, and no structural
regression relative to contemporaneous base: format errors no more than two
percentage points worse; safe-parseable rate and exact-number-use rate no more
than two percentage points lower.

The outcome is mixed/ambiguous if only one seed is a paired success, either run
has a structural regression, both beat base but fail a shaped reference, step
100 improves then step 200 regresses, or one run lacks usable successful-variance
signal. It is negative if neither seed is a paired success while both retain
usable successful-variance signal. It is exploration-limited if both lack
successful-variance groups, or one lacks them and step-200 results show no
consistent positive effect. Violated configuration, mismatched revision,
missing/corrupt artifacts, or failed preflight is invalid and may only be rerun
identically after correction.

No outcome authorizes a third seed, a new reward mapping, longer training, a
new curriculum, an SFT rescue, verifier weakening, or a hyperparameter search
within Milestone 1.

## Permissible Conclusions

- **Positive:** correctness-only reward produced a modest, two-seed held-out
  greedy improvement over the contemporaneous base and same-seed shaped
  references in this concrete small-model TRL GRPO/DAPO setup.
- **Mixed:** the treatment changed behavior but did not yield stable two-seed
  correctness improvement; no robust reward-design claim is supported.
- **Negative:** despite occasional base-policy correct trajectories,
  correctness-only GRPO did not reliably improve greedy held-out correctness at
  this model size, rollout budget, and horizon. This does not establish that
  GRPO generally fails or that model capacity alone is limiting.
- **Exploration-limited:** correctness-only reward was too sparse to provide
  useful relative-success groups at `G=4`; this is a bounded exploration
  diagnosis, not a general GRPO claim.

The 200-task evaluation and two treatment seeds support only bounded evidence.
They do not establish broad population-level generalization, `k=5` competence,
or superiority of GRPO outside this implementation.

## Completion Status (Post-Execution)

The treatment execution and required treatment evaluations are complete. The
planned contemporaneous Kaggle base rerun was not performed; final analysis
uses the existing frozen base report (`9/200`), a documented protocol deviation
that does not affect the nonpositive paired-success decision. The final
classification is **mixed/unstable, with a negative primary correctness
result**. Neither predeclared correctness-only seed met its step-200
paired-success criterion; both retained successful-variance rollout groups, and
seed `777` had material structural regression. The step-100 checkpoints were
evaluated diagnostically as required and were not selected as primary endpoints.

The canonical post-execution evidence and interpretation are recorded in
`reports/countdown_rlvr_week1.md`; paired task transitions are in
`reports/countdown_milestone1_closure_analysis.json`. This annotation reports
execution status only and does not alter the predeclared design, criteria, or
permissible conclusions above.
