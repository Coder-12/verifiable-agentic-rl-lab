import ast
import math
from collections import Counter
from typing import Any


def extract_integer_literals(expression: str) -> tuple[bool, list[int] | None]:
    """
    Returns whether the expression is safely parseable under verifier-like rules and,
    if so, the integer literals it contains.
    """
    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,
        ast.USub,
    )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False, None

    used_numbers: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            return False, None
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                return False, None
            used_numbers.append(node.value)

    return True, used_numbers


def analyze_expression(task: dict[str, Any], expression: str | None) -> dict[str, Any]:
    """Computes literal-usage diagnostics independent of the verifier verdict."""
    required_numbers = task["numbers"]
    analysis = {
        "safe_parseable": False,
        "used_numbers": None,
        "exact_number_usage": False,
        "uses_too_few_literals": False,
        "introduces_forbidden_literals": False,
    }

    if expression is None:
        return analysis

    safe_parseable, used_numbers = extract_integer_literals(expression)
    analysis["safe_parseable"] = safe_parseable
    analysis["used_numbers"] = used_numbers

    if not safe_parseable or used_numbers is None:
        return analysis

    used_counter = Counter(used_numbers)
    required_counter = Counter(required_numbers)

    analysis["exact_number_usage"] = used_counter == required_counter
    analysis["uses_too_few_literals"] = len(used_numbers) < len(required_numbers)
    analysis["introduces_forbidden_literals"] = bool(used_counter - required_counter)
    return analysis


def trim_generated_ids(
    output_ids: Any,
    attention_mask: Any,
) -> list[Any]:
    """
    Removes left padding and prompt tokens per sample.
    Works with tensors or plain nested sequences.
    """
    generated_sequences: list[Any] = []
    input_width = len(attention_mask[0])

    for sample_ids, sample_mask in zip(output_ids, attention_mask):
        if hasattr(sample_mask, "sum"):
            prompt_length = int(sample_mask.sum().item())
        else:
            prompt_length = int(sum(sample_mask))
        pad_prefix = input_width - prompt_length
        trimmed_sample = sample_ids[pad_prefix:]
        generated_sequences.append(trimmed_sample[prompt_length:])

    return generated_sequences


def completion_ranking_key(completion_result: dict[str, Any]) -> tuple[Any, ...]:
    """
    Deterministic ranking for verifier-guided best-of-N selection.
    Higher tuple values are better.
    """
    analysis = completion_result["analysis"]
    sample_index = completion_result["sample_index"]

    return (
        completion_result["reward"],
        int(completion_result["is_valid"]),
        int(analysis["exact_number_usage"]),
        int(analysis["safe_parseable"]),
        int(not analysis["introduces_forbidden_literals"]),
        int(not analysis["uses_too_few_literals"]),
        -sample_index,
    )


def select_best_completion(completion_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Selects the single final output that a verifier-guided best-of-N system would return.
    """
    return max(completion_results, key=completion_ranking_key)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def summarize_seed_sweep(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregates repeated best-of-N runs across seeds into stable summary statistics.
    """
    if not seed_runs:
        raise ValueError("seed_runs must not be empty")

    metric_names = [
        "oracle_pass_at_n",
        "reranked_success_at_n",
        "raw_completion_success_rate",
        "format_error_rate",
        "safe_parseable_rate",
        "exact_number_usage_rate",
        "mean_reward",
        "selected_mean_reward",
    ]

    aggregate_metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        values = [run["metrics"][metric_name] for run in seed_runs]
        aggregate_metrics[metric_name] = {
            "values": values,
            "mean": _mean(values),
            "std": _stddev(values),
            "min": min(values),
            "max": max(values),
        }

    total_runs = len(seed_runs)
    task_summaries = []
    for task_index, task_result in enumerate(seed_runs[0]["results"]):
        oracle_successes = sum(
            1 for run in seed_runs if run["results"][task_index]["oracle_task_passed"]
        )
        reranked_successes = sum(
            1
            for run in seed_runs
            if run["results"][task_index]["selected_completion"]["is_valid"]
        )
        task_summaries.append(
            {
                "task_index": task_index,
                "task": task_result["task"],
                "oracle_success_count": oracle_successes,
                "reranked_success_count": reranked_successes,
            }
        )

    stability_summary = {
        "total_runs": total_runs,
        "oracle_tasks_solved_in_any_seed": sum(
            1 for task in task_summaries if task["oracle_success_count"] > 0
        ),
        "oracle_tasks_solved_in_all_seeds": sum(
            1 for task in task_summaries if task["oracle_success_count"] == total_runs
        ),
        "oracle_tasks_solved_in_at_least_half_of_seeds": sum(
            1 for task in task_summaries if task["oracle_success_count"] * 2 >= total_runs
        ),
        "reranked_tasks_solved_in_any_seed": sum(
            1 for task in task_summaries if task["reranked_success_count"] > 0
        ),
        "reranked_tasks_solved_in_all_seeds": sum(
            1 for task in task_summaries if task["reranked_success_count"] == total_runs
        ),
        "reranked_tasks_solved_in_at_least_half_of_seeds": sum(
            1 for task in task_summaries if task["reranked_success_count"] * 2 >= total_runs
        ),
    }

    return {
        "aggregate_metrics": aggregate_metrics,
        "task_stability": task_summaries,
        "stability_summary": stability_summary,
    }
