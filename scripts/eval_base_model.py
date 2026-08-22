"""
Base Model pass@1 Evaluation Script for Milestone 1.

Edge cases handled:
- Hardware Mapping: Dynamically selects 'mps' for Apple Silicon to ensure 
  maximum hardware acceleration without quantization (uses unquantized bfloat16).
- Padding constraints: Explicitly sets `padding_side="left"` for the tokenizer, 
  which is required for batched causal language modeling.
- Chat Templates: Uses the model's native `apply_chat_template` to format the 
  prompts perfectly, ensuring the instruction-tuned model behaves as expected.
- Graceful Degradation: If `extract_answer` fails (format error), the script 
  captures it explicitly without crashing and maps it to a 0.0 reward equivalent.
"""
import sys
import os
import argparse
from importlib import metadata as importlib_metadata
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tasks.countdown.prompts import (
    COUNTDOWN_SYSTEM_PROMPT,
    extract_answer,
    format_countdown_prompt,
)
from src.tasks.countdown.eval_utils import (
    analyze_expression,
    resolve_pass1_report_path,
    trim_generated_ids,
)
from src.tasks.countdown.reward import reward_from_verification_result
from src.tasks.countdown.verifier import verify_countdown_expression


def get_device() -> torch.device:
    """Safely determines the optimal hardware device."""
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
    """Loads the frozen dataset from disk."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run generation script first.")
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_prompts(dataset: list[dict[str, Any]], tokenizer: AutoTokenizer) -> list[str]:
    """Formats tasks into conversational prompts using the model's native chat template."""
    formatted_prompts = []
    
    for task in dataset:
        user_prompt = format_countdown_prompt(task["numbers"], task["target"])
        messages = [
            {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        # apply_chat_template adds the correct control tokens (e.g., <|im_start|>)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt)
        
    return formatted_prompts


def build_report_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
    dataset_size: int,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Captures the run configuration needed to reproduce an evaluation artifact."""
    return {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_size": dataset_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_dtype": str(model_dtype),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "packages": {
                name: _package_version(name)
                for name in ("torch", "transformers", "datasets", "accelerate", "trl", "peft")
            },
        },
        "command": {
            "argv": sys.argv,
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _resolved_revision(resource: Any) -> str | None:
    commit_hash = getattr(resource, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    init_kwargs = getattr(resource, "init_kwargs", {})
    commit_hash = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
    return str(commit_hash) if commit_hash else None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pass@1 for a base model.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default="data/countdown_eval_50.json",
                        help="Path to the evaluation dataset")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation. Adjust based on M1 RAM.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional explicit artifact path; avoids overwriting historical reports.",
    )
    args = parser.parse_args()

    # 1. Setup Environment
    device = get_device()
    model_dtype = get_model_dtype(device)
    print(f"Loading {args.model} on device: {device} (Precision: {model_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=device,
    )
    model.eval()

    # 2. Load Data and Prepare Prompts
    dataset = load_dataset(args.dataset)
    raw_prompts = prepare_prompts(dataset, tokenizer)
    report_config = build_report_config(
        args=args,
        device=device,
        model_dtype=model_dtype,
        dataset_size=len(dataset),
        model_revision=_resolved_revision(model.config),
        tokenizer_revision=_resolved_revision(tokenizer),
    )
    
    # 3. Generation Loop
    print(f"Generating pass@1 completions for {len(dataset)} tasks...")
    completions = []
    
    for i in tqdm(range(0, len(raw_prompts), args.batch_size)):
        batch_prompts = raw_prompts[i : i + args.batch_size]
        
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        
        # Greedy decoding for pass@1 (temperature=0.0 is deterministic)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = trim_generated_ids(output_ids, inputs.attention_mask)
        batch_texts = [
            tokenizer.decode(sample_ids, skip_special_tokens=True)
            for sample_ids in generated_ids
        ]
        completions.extend(batch_texts)

    # 4. Evaluation and Diagnostics Collection
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

    print("Evaluating completions...")
    for task, completion in zip(dataset, completions):
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
            used_numbers = verdict.used_numbers if verdict.used_numbers is not None else analysis["used_numbers"]

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

        results.append({
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
        })

    # 5. Compile Metrics
    total_examples = len(dataset)
    pass_at_1 = (correct_count / total_examples) * 100
    format_error_rate = (format_failures / total_examples) * 100
    avg_length = total_length / total_examples
    safe_parseable_rate = (safe_parseable_count / total_examples) * 100
    exact_number_usage_rate = (exact_number_usage_count / total_examples) * 100
    mean_reward = mean_reward_total / total_examples

    print("\n" + "="*40)
    print("🎯 EVALUATION RESULTS (pass@1)")
    print("="*40)
    print(f"Model:                {args.model}")
    print(f"pass@1:               {pass_at_1:.2f}%")
    print(f"Format Error Rate:    {format_error_rate:.2f}%")
    print(f"Safe Parseable Rate:  {safe_parseable_rate:.2f}%")
    print(f"Exact Usage Rate:     {exact_number_usage_rate:.2f}%")
    print(f"Mean Reward:          {mean_reward:.3f}")
    print(f"Avg Output Length:    {avg_length:.1f} chars")
    print(f"Too Few Literals:     {uses_too_few_literals_count}")
    print(f"Forbidden Literals:   {forbidden_integer_literal_count}")
    print("-" * 40)
    print("Failure Categories Breakdown:")
    for status, count in status_counts.most_common():
        print(f"  - {status}: {count} ({(count/total_examples)*100:.1f}%)")
    print("-" * 40)
    print("Status Breakdown by Puzzle Size:")
    for k, counts in sorted(status_counts_by_k.items(), key=lambda item: int(item[0])):
        print(f"  - k={k}: {dict(counts)}")
    print("="*40)

    # 6. Save Artifacts for Manual Analysis
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
                "results": results
            },
            f, indent=2
        )
    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
