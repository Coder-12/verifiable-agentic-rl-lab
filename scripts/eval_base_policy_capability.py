"""
Base-policy capability audit for Countdown RLVR.

This script evaluates the untouched base model under the same stochastic rollout
settings used by GRPO training. It is not a trained-model evaluation and it does
not modify model weights.

Main questions:
- How often does a single sampled rollout solve the task?
- How often does a group of rollouts contain at least one correct solution?
- How often does a group contain reward variation that GRPO can use?
- Does useful signal differ by puzzle size?
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tasks.countdown.dataset import generate_countdown_dataset
from src.tasks.countdown.eval_utils import (
    analyze_expression,
    select_best_completion,
    trim_generated_ids,
)
from src.tasks.countdown.prompts import (
    COUNTDOWN_SYSTEM_PROMPT,
    extract_answer,
    format_countdown_prompt,
)
from src.tasks.countdown.reward import REWARD_MAPPING, reward_from_verification_result
from src.tasks.countdown.verifier import verify_countdown_expression

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer


def get_device() -> Any:
    """Safely determines the best available inference device."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_dtype(device: Any) -> Any:
    """Chooses a practical inference dtype for the active device."""
    import torch

    if device.type in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    """Sets RNG state for reproducible stochastic generation."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def allocate_counts_by_k(
    *,
    total_prompts: int,
    min_numbers: int,
    max_numbers: int,
) -> dict[int, int]:
    """Allocates a nearly balanced number of prompts to each puzzle size."""
    if min_numbers > max_numbers:
        raise ValueError("min_numbers must be <= max_numbers.")
    if total_prompts <= 0:
        raise ValueError("total_prompts must be positive.")

    sizes = list(range(min_numbers, max_numbers + 1))
    base_count, remainder = divmod(total_prompts, len(sizes))
    return {
        k: base_count + (1 if index < remainder else 0)
        for index, k in enumerate(sizes)
    }


def generate_balanced_tasks(
    *,
    total_prompts: int | None,
    prompts_per_k: int | None,
    min_numbers: int,
    max_numbers: int,
    min_value: int,
    max_value: int,
    dataset_seed: int,
) -> list[dict[str, Any]]:
    """Generates fresh training-distribution tasks with controlled puzzle sizes."""
    if prompts_per_k is not None and prompts_per_k <= 0:
        raise ValueError("prompts_per_k must be positive when provided.")
    if total_prompts is not None and total_prompts <= 0:
        raise ValueError("total_prompts must be positive when provided.")
    if total_prompts is None and prompts_per_k is None:
        raise ValueError("Either total_prompts or prompts_per_k must be provided.")

    random.seed(dataset_seed)
    if prompts_per_k is not None:
        counts_by_k = {
            k: prompts_per_k for k in range(min_numbers, max_numbers + 1)
        }
    else:
        assert total_prompts is not None
        counts_by_k = allocate_counts_by_k(
            total_prompts=total_prompts,
            min_numbers=min_numbers,
            max_numbers=max_numbers,
        )

    tasks: list[dict[str, Any]] = []
    for k, count in counts_by_k.items():
        tasks.extend(
            generate_countdown_dataset(
                num_samples=count,
                min_numbers=k,
                max_numbers=k,
                min_value=min_value,
                max_value=max_value,
            )
        )

    random.Random(dataset_seed).shuffle(tasks)
    return tasks


def prepare_prompts(dataset: list[dict[str, Any]], tokenizer: "AutoTokenizer") -> list[str]:
    """Formats tasks with the same chat template path used by baseline evaluation."""
    prompts = []
    for task in dataset:
        user_prompt = format_countdown_prompt(task["numbers"], task["target"])
        messages = [
            {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def evaluate_completion(
    *,
    task: dict[str, Any],
    completion: str,
    sample_index: int,
    generated_token_count: int,
    hit_length_limit: bool,
) -> dict[str, Any]:
    """Runs extraction, verifier scoring, and diagnostic analysis for one sample."""
    extracted_math = extract_answer(completion)
    analysis = analyze_expression(task, extracted_math)

    if extracted_math is None:
        reason = "format_error"
        is_valid = False
        reward = 0.0
        used_numbers = None
    else:
        verdict = verify_countdown_expression(
            extracted_math,
            task["target"],
            task["numbers"],
        )
        reason = verdict.reason.value
        is_valid = verdict.is_valid
        reward = reward_from_verification_result(verdict)
        used_numbers = (
            verdict.used_numbers
            if verdict.used_numbers is not None
            else analysis["used_numbers"]
        )

    return {
        "sample_index": sample_index,
        "completion": completion,
        "extracted_math": extracted_math,
        "is_valid": is_valid,
        "reason": reason,
        "reward": reward,
        "used_numbers": used_numbers,
        "analysis": analysis,
        "generated_token_count": generated_token_count,
        "hit_length_limit": hit_length_limit,
    }


def compute_group_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes GRPO-relevant reward and repetition diagnostics for one group."""
    rewards = [sample["reward"] for sample in samples]
    has_correct = any(sample["is_valid"] for sample in samples)
    rewards_differ = len(set(rewards)) > 1
    completions = [sample["completion"].strip() for sample in samples]
    extracted_answers = [
        sample["extracted_math"]
        for sample in samples
        if sample["extracted_math"] is not None
    ]

    return {
        "rewards": rewards,
        "verification_statuses": [sample["reason"] for sample in samples],
        "has_correct": has_correct,
        "rewards_differ": rewards_differ,
        "success_variance": has_correct and any(reward < 1.0 for reward in rewards),
        "structural_only_variance": rewards_differ and not has_correct,
        "all_zero_rewards": all(reward == 0.0 for reward in rewards),
        "all_same_nonzero_rewards": len(set(rewards)) == 1 and rewards[0] > 0.0,
        "duplicate_completion": len(set(completions)) < len(completions),
        "duplicate_answer": len(set(extracted_answers)) < len(extracted_answers),
        "any_hit_length_limit": any(sample["hit_length_limit"] for sample in samples),
    }


def _percentage(count: int, total: int) -> float:
    return (count / total) * 100 if total else 0.0


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates capability and trainability metrics over prompt groups."""
    if not groups:
        raise ValueError("groups must not be empty.")

    total_groups = len(groups)
    total_completions = sum(len(group["samples"]) for group in groups)
    all_samples = [
        sample
        for group in groups
        for sample in group["samples"]
    ]

    correct_completions = sum(1 for sample in all_samples if sample["is_valid"])
    first_sample_correct = sum(
        1 for group in groups if group["samples"][0]["is_valid"]
    )
    group_successes = sum(1 for group in groups if group["has_correct"])
    reward_variance_groups = sum(1 for group in groups if group["rewards_differ"])
    success_variance_groups = sum(1 for group in groups if group["success_variance"])
    structural_only_variance_groups = sum(
        1 for group in groups if group["structural_only_variance"]
    )
    all_zero_groups = sum(1 for group in groups if group["all_zero_rewards"])
    all_same_nonzero_groups = sum(
        1 for group in groups if group["all_same_nonzero_rewards"]
    )
    duplicate_completion_groups = sum(
        1 for group in groups if group["duplicate_completion"]
    )
    duplicate_answer_groups = sum(1 for group in groups if group["duplicate_answer"])
    length_limit_groups = sum(1 for group in groups if group["any_hit_length_limit"])
    length_limit_completions = sum(
        1 for sample in all_samples if sample["hit_length_limit"]
    )

    selected_correct = 0
    selected_reward_total = 0.0
    selected_status_counts: Counter[str] = Counter()
    for group in groups:
        selected = select_best_completion(group["samples"])
        selected_correct += int(selected["is_valid"])
        selected_reward_total += selected["reward"]
        selected_status_counts[selected["reason"]] += 1

    reward_counts: Counter[str] = Counter(str(sample["reward"]) for sample in all_samples)
    status_counts: Counter[str] = Counter(sample["reason"] for sample in all_samples)

    safe_parseable_count = sum(
        1 for sample in all_samples if sample["analysis"]["safe_parseable"]
    )
    exact_usage_count = sum(
        1 for sample in all_samples if sample["analysis"]["exact_number_usage"]
    )
    too_few_literals_count = sum(
        1 for sample in all_samples if sample["analysis"]["uses_too_few_literals"]
    )
    forbidden_literals_count = sum(
        1
        for sample in all_samples
        if sample["analysis"]["introduces_forbidden_literals"]
    )
    reward_total = sum(sample["reward"] for sample in all_samples)
    output_char_total = sum(len(sample["completion"]) for sample in all_samples)
    output_token_total = sum(sample["generated_token_count"] for sample in all_samples)

    return {
        "total_groups": total_groups,
        "total_completions": total_completions,
        "sample_1_success_rate": _percentage(first_sample_correct, total_groups),
        "correct_rollout_rate": _percentage(correct_completions, total_completions),
        "group_success_rate": _percentage(group_successes, total_groups),
        "oracle_pass_at_num_generations": _percentage(group_successes, total_groups),
        "verifier_reranked_success_at_num_generations": _percentage(
            selected_correct, total_groups
        ),
        "nonzero_reward_variance_group_rate": _percentage(
            reward_variance_groups, total_groups
        ),
        "success_variance_group_rate": _percentage(
            success_variance_groups, total_groups
        ),
        "structural_only_variance_group_rate": _percentage(
            structural_only_variance_groups, total_groups
        ),
        "all_zero_group_rate": _percentage(all_zero_groups, total_groups),
        "all_same_nonzero_group_rate": _percentage(
            all_same_nonzero_groups, total_groups
        ),
        "duplicate_completion_group_rate": _percentage(
            duplicate_completion_groups, total_groups
        ),
        "duplicate_answer_group_rate": _percentage(
            duplicate_answer_groups, total_groups
        ),
        "length_limit_group_rate": _percentage(length_limit_groups, total_groups),
        "length_limit_completion_rate": _percentage(
            length_limit_completions, total_completions
        ),
        "safe_parseable_rate": _percentage(safe_parseable_count, total_completions),
        "exact_number_usage_rate": _percentage(exact_usage_count, total_completions),
        "uses_too_few_literals_count": too_few_literals_count,
        "introduces_forbidden_literals_count": forbidden_literals_count,
        "mean_reward": reward_total / total_completions,
        "selected_mean_reward": selected_reward_total / total_groups,
        "avg_output_length_chars": output_char_total / total_completions,
        "avg_generated_tokens": output_token_total / total_completions,
        "reward_distribution": dict(reward_counts),
        "status_counts": dict(status_counts),
        "selected_status_counts": dict(selected_status_counts),
    }


def summarize_by_k(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Computes the same summary separately for each puzzle size."""
    groups_by_k: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        groups_by_k[str(group["k"])].append(group)
    return {
        k: summarize_groups(k_groups)
        for k, k_groups in sorted(groups_by_k.items(), key=lambda item: int(item[0]))
    }


def build_report_config(
    *,
    args: argparse.Namespace,
    device: Any,
    model_dtype: Any,
    dataset_size: int,
) -> dict[str, Any]:
    """Captures the run configuration needed to reproduce the audit."""
    return {
        "experiment": "base_policy_capability_audit",
        "model": args.model,
        "dataset_source": "fresh_training_distribution",
        "dataset_size": dataset_size,
        "min_numbers": args.min_numbers,
        "max_numbers": args.max_numbers,
        "min_value": args.min_value,
        "max_value": args.max_value,
        "num_prompts": args.num_prompts,
        "prompts_per_k": args.prompts_per_k,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_dtype": str(model_dtype),
        "dataset_seed": args.dataset_seed,
        "generation_seed": args.generation_seed,
        "reward_mapping": dict(REWARD_MAPPING),
        "decoding": {
            "strategy": "stochastic_grpo_rollout_audit",
            "num_generations": args.num_generations,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "semantics": {
            "sample_1_success_rate": (
                "Success rate of the first sampled rollout per prompt. "
                "This is not greedy pass@1."
            ),
            "oracle_pass_at_num_generations": (
                "Fraction of prompt groups with at least one verifier-correct "
                "completion among num_generations samples."
            ),
            "verifier_reranked_success_at_num_generations": (
                "Fraction of prompt groups where verifier reward ranking selects "
                "a correct completion."
            ),
        },
        "environment": {
            "python": platform.python_version(),
        },
        "command": {
            "argv": sys.argv,
        },
    }


def safe_model_name(model_name: str) -> str:
    """Converts a model ID/path into a filename-safe label."""
    return model_name.rstrip("/").split("/")[-1].replace(" ", "_")


def report_path_for_args(args: argparse.Namespace, total_tasks: int) -> Path:
    """Builds a deterministic report path for the audit configuration."""
    return (
        Path("reports")
        / (
            f"eval_{safe_model_name(args.model)}"
            f"_base_policy_capability"
            f"_tasks{total_tasks}"
            f"_g{args.num_generations}"
            f"_seed{args.generation_seed}.json"
        )
    )


def _strip_trailing_pad_ids(token_ids: list[int], pad_token_id: int | None) -> list[int]:
    if pad_token_id is None:
        return token_ids
    while token_ids and token_ids[-1] == pad_token_id:
        token_ids.pop()
    return token_ids


def generated_token_metadata(
    generated_ids: list[Any],
    *,
    max_new_tokens: int,
    pad_token_id: int | None,
) -> list[dict[str, Any]]:
    """Computes approximate generation-length and length-limit diagnostics."""
    metadata = []
    for sample_ids in generated_ids:
        if hasattr(sample_ids, "tolist"):
            token_ids = sample_ids.tolist()
        else:
            token_ids = list(sample_ids)
        visible_token_ids = _strip_trailing_pad_ids(token_ids, pad_token_id)
        metadata.append(
            {
                "generated_token_count": len(visible_token_ids),
                "hit_length_limit": len(visible_token_ids) >= max_new_tokens,
            }
        )
    return metadata


def print_summary(metrics: dict[str, Any], metrics_by_k: dict[str, dict[str, Any]]) -> None:
    """Prints the key audit metrics in a compact terminal format."""
    print("\n" + "=" * 58)
    print("BASE-POLICY CAPABILITY AUDIT")
    print("=" * 58)
    print(f"Prompt groups:                      {metrics['total_groups']}")
    print(f"Total sampled completions:           {metrics['total_completions']}")
    print(f"Sample-1 success rate:               {metrics['sample_1_success_rate']:.2f}%")
    print(f"Correct rollout rate:                {metrics['correct_rollout_rate']:.2f}%")
    print(f"Oracle group success rate:           {metrics['group_success_rate']:.2f}%")
    print(
        "Verifier-reranked group success:     "
        f"{metrics['verifier_reranked_success_at_num_generations']:.2f}%"
    )
    print(
        "Nonzero reward-variance group rate:  "
        f"{metrics['nonzero_reward_variance_group_rate']:.2f}%"
    )
    print(
        "Success-variance group rate:         "
        f"{metrics['success_variance_group_rate']:.2f}%"
    )
    print(
        "Structural-only variance group rate: "
        f"{metrics['structural_only_variance_group_rate']:.2f}%"
    )
    print(f"All-zero group rate:                 {metrics['all_zero_group_rate']:.2f}%")
    print(f"Reward distribution:                 {metrics['reward_distribution']}")
    print(f"Status counts:                       {metrics['status_counts']}")
    print("-" * 58)
    print("By puzzle size:")
    for k, k_metrics in metrics_by_k.items():
        print(
            f"  k={k}: groups={k_metrics['total_groups']}, "
            f"sample1={k_metrics['sample_1_success_rate']:.2f}%, "
            f"group_success={k_metrics['group_success_rate']:.2f}%, "
            f"reward_variance={k_metrics['nonzero_reward_variance_group_rate']:.2f}%, "
            f"success_variance={k_metrics['success_variance_group_rate']:.2f}%, "
            f"all_zero={k_metrics['all_zero_group_rate']:.2f}%"
        )
    print("=" * 58)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit base-model Countdown capability under GRPO rollout settings."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument(
        "--prompts-per-k",
        type=int,
        default=None,
        help="If set, overrides --num-prompts and generates this many prompts per k.",
    )
    parser.add_argument("--min-numbers", type=int, default=3)
    parser.add_argument("--max-numbers", type=int, default=5)
    parser.add_argument("--min-value", type=int, default=1)
    parser.add_argument("--max-value", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    device = get_device()
    model_dtype = get_model_dtype(device)

    print(f"Loading {args.model} on {device} (dtype={model_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=device,
    )
    model.eval()

    dataset = generate_balanced_tasks(
        total_prompts=args.num_prompts,
        prompts_per_k=args.prompts_per_k,
        min_numbers=args.min_numbers,
        max_numbers=args.max_numbers,
        min_value=args.min_value,
        max_value=args.max_value,
        dataset_seed=args.dataset_seed,
    )
    prompts = prepare_prompts(dataset, tokenizer)
    expanded_prompts = [
        prompt
        for prompt in prompts
        for _ in range(args.num_generations)
    ]

    set_seed(args.generation_seed)
    print(
        f"Generating {args.num_generations} sampled rollouts for "
        f"{len(dataset)} prompts ({len(expanded_prompts)} completions total)..."
    )

    expanded_completions: list[str] = []
    expanded_token_metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(expanded_prompts), args.batch_size)):
            batch_prompts = expanded_prompts[batch_start : batch_start + args.batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_ids = trim_generated_ids(output_ids, inputs.attention_mask)
            expanded_token_metadata.extend(
                generated_token_metadata(
                    generated_ids,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )
            expanded_completions.extend(
                tokenizer.decode(sample_ids, skip_special_tokens=True)
                for sample_ids in generated_ids
            )

    groups = []
    for task_index, task in enumerate(dataset):
        start = task_index * args.num_generations
        end = start + args.num_generations
        task_completions = expanded_completions[start:end]
        task_token_metadata = expanded_token_metadata[start:end]
        samples = [
            evaluate_completion(
                task=task,
                completion=completion,
                sample_index=sample_index,
                generated_token_count=metadata["generated_token_count"],
                hit_length_limit=metadata["hit_length_limit"],
            )
            for sample_index, (completion, metadata) in enumerate(
                zip(task_completions, task_token_metadata, strict=True)
            )
        ]
        group_diagnostics = compute_group_diagnostics(samples)
        selected = select_best_completion(samples)
        groups.append(
            {
                "task_index": task_index,
                "task": task,
                "k": len(task["numbers"]),
                "selected_completion_index": selected["sample_index"],
                "selected_completion": selected,
                "samples": samples,
                **group_diagnostics,
            }
        )

    metrics = summarize_groups(groups)
    metrics_by_k = summarize_by_k(groups)
    print_summary(metrics, metrics_by_k)

    report_config = build_report_config(
        args=args,
        device=device,
        model_dtype=model_dtype,
        dataset_size=len(dataset),
    )
    report_config["environment"]["torch"] = torch.__version__
    report_config["environment"]["transformers"] = transformers.__version__

    report_path = report_path_for_args(args, len(dataset))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": report_config,
                "metrics": metrics,
                "metrics_by_k": metrics_by_k,
                "groups": groups,
            },
            f,
            indent=2,
        )

    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
