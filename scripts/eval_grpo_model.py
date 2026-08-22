"""
Evaluate a GRPO-trained Countdown policy checkpoint with the same strict pass@1
metrics used for the base-model baselines.
"""

import argparse
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tasks.countdown.eval_utils import (
    analyze_expression,
    resolve_pass1_report_path,
    trim_generated_ids,
)
from src.tasks.countdown.prompts import (
    COUNTDOWN_SYSTEM_PROMPT,
    extract_answer,
    format_countdown_prompt,
)
from src.tasks.countdown.reward import reward_from_verification_result
from src.tasks.countdown.verifier import verify_countdown_expression
from scripts.train_grpo_countdown import disable_peft_torchao_adapter_path


def get_device() -> torch.device:
    """Safely determines the best available inference device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_dtype(device: torch.device) -> torch.dtype:
    """Chooses a practical inference dtype for the active device."""
    if device.type in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Loads the frozen evaluation dataset from disk."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_prompts(dataset: list[dict[str, Any]], tokenizer: AutoTokenizer) -> list[str]:
    """Formats tasks using the same chat template path as baseline evaluation."""
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


def build_report_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
    dataset_size: int,
) -> dict[str, Any]:
    """Captures the config needed to reproduce the trained-model evaluation."""
    return {
        "adapter_path": args.model,
        "dataset": args.dataset,
        "dataset_size": dataset_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_dtype": str(model_dtype),
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "command": {
            "argv": sys.argv,
        },
    }


@torch.no_grad()
def main() -> None:
    disable_peft_torchao_adapter_path()
    from peft import AutoPeftModelForCausalLM

    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained model checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved GRPO adapter/checkpoint.")
    parser.add_argument("--dataset", type=str, default="data/countdown_eval_200.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional explicit artifact path; required when checkpoint names would collide.",
    )
    args = parser.parse_args()

    device = get_device()
    model_dtype = get_model_dtype(device)
    print(f"Loading trained checkpoint {args.model} on {device} (dtype={model_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=device,
    )
    model.eval()

    dataset = load_dataset(args.dataset)
    prompts = prepare_prompts(dataset, tokenizer)
    report_config = build_report_config(
        args=args,
        device=device,
        model_dtype=model_dtype,
        dataset_size=len(dataset),
    )

    print(f"Generating greedy completions for {len(dataset)} tasks...")
    completions = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_ids = trim_generated_ids(output_ids, inputs.attention_mask)
        completions.extend(
            tokenizer.decode(sample_ids, skip_special_tokens=True)
            for sample_ids in generated_ids
        )

    print("Evaluating completions...")
    results = []
    status_counts = Counter()
    status_counts_by_k: dict[str, Counter[str]] = defaultdict(Counter)
    total_length = 0
    format_failures = 0
    correct_count = 0
    safe_parseable_count = 0
    exact_number_usage_count = 0
    mean_reward_total = 0.0
    uses_too_few_literals_count = 0
    forbidden_integer_literal_count = 0

    for task, completion in zip(dataset, completions, strict=True):
        total_length += len(completion)
        extracted_math = extract_answer(completion)
        analysis = analyze_expression(task, extracted_math)

        if extracted_math is None:
            reason = "format_error"
            is_valid = False
            reward = 0.0
            used_numbers = None
            format_failures += 1
        else:
            verdict = verify_countdown_expression(extracted_math, task["target"], task["numbers"])
            reason = verdict.reason.value
            is_valid = verdict.is_valid
            reward = reward_from_verification_result(verdict)
            used_numbers = (
                verdict.used_numbers if verdict.used_numbers is not None else analysis["used_numbers"]
            )
            if is_valid:
                correct_count += 1

        if analysis["safe_parseable"]:
            safe_parseable_count += 1
        if analysis["exact_number_usage"]:
            exact_number_usage_count += 1
        if analysis["uses_too_few_literals"]:
            uses_too_few_literals_count += 1
        if analysis["introduces_forbidden_literals"]:
            forbidden_integer_literal_count += 1

        mean_reward_total += reward
        status_counts[reason] += 1
        status_counts_by_k[str(len(task["numbers"]))][reason] += 1
        results.append(
            {
                "task": task,
                "completion": completion,
                "extracted_math": extracted_math,
                "is_valid": is_valid,
                "reason": reason,
                "reward": reward,
                "safe_parseable": analysis["safe_parseable"],
                "used_numbers": used_numbers,
                "exact_number_usage": analysis["exact_number_usage"],
                "uses_too_few_literals": analysis["uses_too_few_literals"],
                "introduces_forbidden_literals": analysis["introduces_forbidden_literals"],
            }
        )

    total_examples = len(dataset)
    pass_at_1 = (correct_count / total_examples) * 100
    format_error_rate = (format_failures / total_examples) * 100
    avg_length = total_length / total_examples
    safe_parseable_rate = (safe_parseable_count / total_examples) * 100
    exact_number_usage_rate = (exact_number_usage_count / total_examples) * 100
    mean_reward = mean_reward_total / total_examples

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS (trained pass@1)")
    print("=" * 40)
    print(f"Checkpoint:            {args.model}")
    print(f"pass@1:                {pass_at_1:.2f}%")
    print(f"Format Error Rate:     {format_error_rate:.2f}%")
    print(f"Safe Parseable Rate:   {safe_parseable_rate:.2f}%")
    print(f"Exact Usage Rate:      {exact_number_usage_rate:.2f}%")
    print(f"Mean Reward:           {mean_reward:.3f}")
    print(f"Avg Output Length:     {avg_length:.1f} chars")
    print(f"Too Few Literals:      {uses_too_few_literals_count}")
    print(f"Forbidden Literals:    {forbidden_integer_literal_count}")
    print("-" * 40)
    print("Failure Categories Breakdown:")
    for status, count in status_counts.most_common():
        print(f"  - {status}: {count} ({(count / total_examples) * 100:.1f}%)")
    print("-" * 40)
    print("Status Breakdown by Puzzle Size:")
    for k, counts in sorted(status_counts_by_k.items(), key=lambda item: int(item[0])):
        print(f"  - k={k}: {dict(counts)}")
    print("=" * 40)

    report_path = resolve_pass1_report_path(args.model, args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": report_config,
                "metrics": {
                    "pass_at_1": pass_at_1,
                    "format_error_rate": format_error_rate,
                    "safe_parseable_rate": safe_parseable_rate,
                    "exact_number_usage_rate": exact_number_usage_rate,
                    "mean_reward": mean_reward,
                    "avg_output_length": avg_length,
                    "status_counts": dict(status_counts),
                    "status_counts_by_k": {
                        k: dict(counts) for k, counts in status_counts_by_k.items()
                    },
                    "uses_too_few_literals_count": uses_too_few_literals_count,
                    "introduces_forbidden_literals_count": forbidden_integer_literal_count,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
