"""Analyze predeclared Milestone 1 closure reports without loading a model."""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    """Loads a saved pass@1 report and validates the minimum comparison schema."""
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report.get("results"), list):
        raise ValueError(f"Report has no results list: {path}")
    return report


def task_key(result: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    """Identifies an evaluation task independently of its stored reference solution."""
    task = result["task"]
    return tuple(task["numbers"]), int(task["target"])


def is_correct(result: dict[str, Any]) -> bool:
    """Uses the strict verifier outcome saved by the evaluation report."""
    return bool(result.get("is_valid", result.get("reason") == "correct"))


def _exact_sign_test_p_value(gained: int, lost: int) -> float | None:
    """Two-sided exact sign-test p-value for descriptive paired uncertainty only."""
    discordant = gained + lost
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, successes) for successes in range(min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def analyze_report_pair(base_report: dict[str, Any], treatment_report: dict[str, Any]) -> dict[str, Any]:
    """Computes exact task transitions, per-k transitions, and metric deltas."""
    base_results = {task_key(result): result for result in base_report["results"]}
    treatment_results = {task_key(result): result for result in treatment_report["results"]}
    if base_results.keys() != treatment_results.keys():
        raise ValueError("Reports do not evaluate the same frozen task set.")

    transitions = {"gained": 0, "lost": 0, "both_correct": 0, "both_incorrect": 0}
    transitions_by_k: dict[str, dict[str, int]] = {}
    for key, base_result in base_results.items():
        treatment_result = treatment_results[key]
        k = str(len(key[0]))
        per_k = transitions_by_k.setdefault(k, {name: 0 for name in transitions})
        base_correct = is_correct(base_result)
        treatment_correct = is_correct(treatment_result)
        if base_correct and treatment_correct:
            transition = "both_correct"
        elif treatment_correct:
            transition = "gained"
        elif base_correct:
            transition = "lost"
        else:
            transition = "both_incorrect"
        transitions[transition] += 1
        per_k[transition] += 1

    metric_names = (
        "pass_at_1",
        "format_error_rate",
        "safe_parseable_rate",
        "exact_number_usage_rate",
        "mean_reward",
    )
    base_metrics = base_report.get("metrics", {})
    treatment_metrics = treatment_report.get("metrics", {})
    metric_deltas = {
        name: treatment_metrics.get(name, 0.0) - base_metrics.get(name, 0.0)
        for name in metric_names
    }
    return {
        "base_correct_count": sum(is_correct(result) for result in base_results.values()),
        "treatment_correct_count": sum(is_correct(result) for result in treatment_results.values()),
        "task_count": len(base_results),
        "transitions": transitions,
        "transitions_by_k": dict(sorted(transitions_by_k.items(), key=lambda item: int(item[0]))),
        "descriptive_exact_sign_test_p_value": _exact_sign_test_p_value(
            transitions["gained"], transitions["lost"]
        ),
        "metric_deltas": metric_deltas,
    }


def parse_labelled_path(value: str) -> tuple[str, Path]:
    """Parses `label=path` CLI values while allowing paths containing equals later."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.")
    label, path = value.split("=", maxsplit=1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected a non-empty LABEL and PATH.")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument(
        "--treatment-report",
        action="append",
        type=parse_labelled_path,
        required=True,
        help="Repeat as SEED=PATH for each predeclared treatment report.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_report = load_report(args.base_report)
    comparisons = {}
    for label, treatment_path in args.treatment_report:
        comparisons[label] = {
            "treatment_report": str(treatment_path),
            **analyze_report_pair(base_report, load_report(treatment_path)),
        }

    output = {
        "base_report": str(args.base_report),
        "comparisons": comparisons,
        "note": (
            "Paired uncertainty is descriptive for this 200-task Milestone 1 protocol; "
            "it is not a population-level significance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
