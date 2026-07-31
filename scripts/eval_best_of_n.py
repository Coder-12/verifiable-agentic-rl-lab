"""
Best-of-N Evaluation Script for Milestone 1.

Edge cases handled:
- Diverse Sampling: Uses `do_sample=True`, `temperature=0.7`, and `top_p=0.95` 
  to ensure the N completions explore different mathematical paths.
- Memory Management: Expands the prompts list by N and chunks them by `batch_size`.
  This allows strict control over VRAM usage on Apple Silicon (M1).
- Oracle vs. Reranked Metrics: Separately reports whether ANY of the N completions
  is correct and whether the verifier-selected final completion is correct.
- Diagnostic Passthrough: Inherits the `eval_utils` diagnostics from the pass@1 
  refactoring for consistent failure analysis.
"""

import argparse
import json
import os
import platform
import random
import sys
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

from src.tasks.countdown.eval_utils import (
    analyze_expression,
    select_best_completion,
    summarize_seed_sweep,
    trim_generated_ids,
)
from src.tasks.countdown.prompts import (
    COUNTDOWN_SYSTEM_PROMPT,
    extract_answer,
    format_countdown_prompt,
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
    """Formats tasks into conversational prompts using the native chat template."""
    formatted_prompts = []
    for task in dataset:
        user_prompt = format_countdown_prompt(task["numbers"], task["target"])
        messages = [
            {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt)
    return formatted_prompts


def set_seed(seed: int) -> None:
    """Sets the RNG state for reproducible best-of-N sampling."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def parse_seeds(seed_argument: str | None, default_seed: int) -> list[int]:
    """Parses a comma-separated seed list, falling back to the default single seed."""
    if seed_argument is None:
        return [default_seed]

    seeds = [int(part.strip()) for part in seed_argument.split(",") if part.strip()]
    if not seeds:
        raise ValueError("Expected at least one seed in --seeds.")
    return seeds


def build_report_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
    dataset_size: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Captures the run configuration needed to reproduce a best-of-N artifact."""
    return {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_size": dataset_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_dtype": str(model_dtype),
        "n_value": args.n,
        "seed": args.seed,
        "seeds": seeds,
        "decoding": {
            "strategy": "best_of_n_sampling",
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
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
def run_best_of_n_for_seed(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: list[dict[str, Any]],
    base_prompts: list[str],
    device: torch.device,
    n: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Runs a single seeded best-of-N evaluation and returns metrics plus detailed results."""
    set_seed(seed)

    # Expand: [Prompt A, Prompt A, Prompt A, Prompt A, Prompt B, Prompt B...]
    expanded_prompts = [prompt for prompt in base_prompts for _ in range(n)]
    
    # 3. Generation Loop
    print(f"Generating {n} completions per task ({len(expanded_prompts)} total) with seed={seed}...")
    expanded_completions = []
    
    for i in tqdm(range(0, len(expanded_prompts), batch_size)):
        batch_prompts = expanded_prompts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        
        # Sampling is strictly required for Best-of-N diversity
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = trim_generated_ids(output_ids, inputs.attention_mask)
        batch_texts = [
            tokenizer.decode(sample_ids, skip_special_tokens=True)
            for sample_ids in generated_ids
        ]
        expanded_completions.extend(batch_texts)

    # 4. Evaluation and Aggregation
    print("Evaluating completions...")
    
    results = []
    oracle_tasks_passed = 0
    reranked_tasks_passed = 0
    total_completions = len(expanded_completions)
    
    # Diagnostics counters
    global_status_counts = Counter()
    status_counts_by_k: dict[str, Counter[str]] = defaultdict(Counter)
    selected_status_counts = Counter()
    format_failures = 0
    safe_parseable_count = 0
    exact_number_usage_count = 0
    uses_too_few_literals_count = 0
    forbidden_integer_literal_count = 0
    reward_total = 0.0
    raw_correct_completions = 0
    selected_reward_total = 0.0

    # Group completions back by task
    for i, task in enumerate(dataset):
        task_completions = expanded_completions[i * n : (i + 1) * n]
        
        oracle_task_passed = False
        task_results = []
        
        for sample_index, completion in enumerate(task_completions):
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
                    oracle_task_passed = True
                    raw_correct_completions += 1
            
            # Aggregate global diagnostics
            if analysis["safe_parseable"]:
                safe_parseable_count += 1
            if analysis["exact_number_usage"]:
                exact_number_usage_count += 1
            if analysis["uses_too_few_literals"]:
                uses_too_few_literals_count += 1
            if analysis["introduces_forbidden_literals"]:
                forbidden_integer_literal_count += 1

            reward_total += reward
            global_status_counts[reason] += 1
            status_counts_by_k[str(len(task["numbers"]))][reason] += 1
            
            task_results.append({
                "sample_index": sample_index,
                "completion": completion,
                "extracted_math": extracted_math,
                "is_valid": is_valid,
                "reason": reason,
                "reward": reward,
                "used_numbers": used_numbers,
                "analysis": analysis,
            })

        selected_completion = select_best_completion(task_results)
        selected_reward_total += selected_completion["reward"]
        selected_status_counts[selected_completion["reason"]] += 1

        if oracle_task_passed:
            oracle_tasks_passed += 1
        if selected_completion["is_valid"]:
            reranked_tasks_passed += 1

        results.append({
            "task": task,
            "oracle_task_passed": oracle_task_passed,
            "selected_completion_index": selected_completion["sample_index"],
            "selected_completion": selected_completion,
            "completions": task_results,
        })

    # 5. Compile Metrics
    total_tasks = len(dataset)
    oracle_pass_at_n = (oracle_tasks_passed / total_tasks) * 100
    reranked_success_at_n = (reranked_tasks_passed / total_tasks) * 100
    raw_completion_success_rate = (raw_correct_completions / total_completions) * 100
    format_error_rate = (format_failures / total_completions) * 100
    safe_parseable_rate = (safe_parseable_count / total_completions) * 100
    exact_number_usage_rate = (exact_number_usage_count / total_completions) * 100
    mean_reward = reward_total / total_completions
    selected_mean_reward = selected_reward_total / total_tasks

    metrics = {
        "n_value": n,
        "seed": seed,
        "total_tasks": total_tasks,
        "total_completions": total_completions,
        "pass_at_n": oracle_pass_at_n,
        "oracle_pass_at_n": oracle_pass_at_n,
        "reranked_success_at_n": reranked_success_at_n,
        "raw_completion_success_rate": raw_completion_success_rate,
        "format_error_rate": format_error_rate,
        "safe_parseable_rate": safe_parseable_rate,
        "exact_number_usage_rate": exact_number_usage_rate,
        "mean_reward": mean_reward,
        "selected_mean_reward": selected_mean_reward,
        "global_status_counts": dict(global_status_counts),
        "selected_status_counts": dict(selected_status_counts),
        "status_counts_by_k": {
            k: dict(counts) for k, counts in status_counts_by_k.items()
        },
        "uses_too_few_literals_count": uses_too_few_literals_count,
        "introduces_forbidden_literals_count": forbidden_integer_literal_count,
    }

    return {"seed": seed, "metrics": metrics, "results": results}


def print_single_run_summary(model_name: str, metrics: dict[str, Any]) -> None:
    """Prints the summary for one seeded run."""
    n_value = metrics["n_value"]
    total_tasks = metrics["total_tasks"]
    total_completions = metrics["total_completions"]

    print("\n" + "="*45)
    print(f"🎯 EVALUATION RESULTS (Best-of-{n_value})")
    print("="*45)
    print(f"Model:                {model_name}")
    print(f"Seed:                 {metrics['seed']}")
    print(f"Tasks Evaluated:      {total_tasks}")
    print(f"Oracle Pass@{n_value}:       {metrics['oracle_pass_at_n']:.2f}%")
    print(f"Reranked Success@{n_value}:  {metrics['reranked_success_at_n']:.2f}%")
    print(f"Raw Completion Success: {metrics['raw_completion_success_rate']:.2f}%")
    print(f"Format Error Rate:    {metrics['format_error_rate']:.2f}% (across all completions)")
    print(f"Safe Parseable Rate:  {metrics['safe_parseable_rate']:.2f}%")
    print(f"Exact Usage Rate:     {metrics['exact_number_usage_rate']:.2f}%")
    print(f"Mean Reward:          {metrics['mean_reward']:.3f}")
    print(f"Selected Mean Reward: {metrics['selected_mean_reward']:.3f}")
    print("-" * 45)
    print("Global Failure Breakdown (all generations):")
    for status, count in Counter(metrics["global_status_counts"]).most_common():
        print(f"  - {status}: {count} ({(count/total_completions)*100:.1f}%)")
    print("-" * 45)
    print("Selected Output Breakdown:")
    for status, count in Counter(metrics["selected_status_counts"]).most_common():
        print(f"  - {status}: {count} ({(count/total_tasks)*100:.1f}%)")
    print("="*45)

    
def print_seed_sweep_summary(model_name: str, n_value: int, summary: dict[str, Any]) -> None:
    """Prints aggregate stats across multiple seeds."""
    aggregate = summary["aggregate_metrics"]
    stability = summary["stability_summary"]

    print("\n" + "=" * 52)
    print(f"🎯 SEED SWEEP SUMMARY (Best-of-{n_value})")
    print("=" * 52)
    print(f"Model:                {model_name}")
    print(f"Seeds Evaluated:      {stability['total_runs']}")
    print(
        f"Oracle Pass@{n_value}:       "
        f"{aggregate['oracle_pass_at_n']['mean']:.2f}% "
        f"+/- {aggregate['oracle_pass_at_n']['std']:.2f}"
    )
    print(
        f"Reranked Success@{n_value}:  "
        f"{aggregate['reranked_success_at_n']['mean']:.2f}% "
        f"+/- {aggregate['reranked_success_at_n']['std']:.2f}"
    )
    print(
        "Raw Completion Success: "
        f"{aggregate['raw_completion_success_rate']['mean']:.2f}% "
        f"+/- {aggregate['raw_completion_success_rate']['std']:.2f}"
    )
    print(
        f"Tasks solved in any seed (oracle): {stability['oracle_tasks_solved_in_any_seed']}"
    )
    print(
        "Tasks solved in at least half the seeds (oracle): "
        f"{stability['oracle_tasks_solved_in_at_least_half_of_seeds']}"
    )
    print(
        f"Tasks solved in any seed (reranked): {stability['reranked_tasks_solved_in_any_seed']}"
    )
    print(
        "Tasks solved in at least half the seeds (reranked): "
        f"{stability['reranked_tasks_solved_in_at_least_half_of_seeds']}"
    )
    print("=" * 52)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best-of-N for a base model.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="data/countdown_eval_50.json")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of sequences generated simultaneously.")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of diverse completions to generate per task (Best-of-N).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list for repeated runs, e.g. 41,42,43,44,45")
    args = parser.parse_args()

    # 1. Setup Environment
    device = get_device()
    seeds = parse_seeds(args.seeds, args.seed)
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

    # 2. Load Data and Prepare Prompts
    dataset = load_dataset(args.dataset)
    base_prompts = prepare_prompts(dataset, tokenizer)
    report_config = build_report_config(
        args=args,
        device=device,
        model_dtype=model_dtype,
        dataset_size=len(dataset),
        seeds=seeds,
    )

    seed_runs = []
    for seed in seeds:
        run = run_best_of_n_for_seed(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            base_prompts=base_prompts,
            device=device,
            n=args.n,
            batch_size=args.batch_size,
            seed=seed,
        )
        seed_runs.append(run)
        print_single_run_summary(args.model, run["metrics"])

    # 6. Save Artifacts
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model_name = args.model.split("/")[-1]
    if len(seed_runs) == 1:
        report_path = reports_dir / f"eval_{safe_model_name}_best_of_{args.n}_results.json"
        payload = {
            "config": report_config,
            "metrics": seed_runs[0]["metrics"],
            "results": seed_runs[0]["results"],
        }
    else:
        summary = summarize_seed_sweep(seed_runs)
        print_seed_sweep_summary(args.model, args.n, summary)
        report_path = reports_dir / f"eval_{safe_model_name}_best_of_{args.n}_seed_sweep_results.json"
        payload = {
            "config": report_config,
            "summary": summary,
            "runs": seed_runs,
        }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
