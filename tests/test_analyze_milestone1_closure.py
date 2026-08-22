from scripts.analyze_milestone1_closure import analyze_report_pair


def _result(numbers, target, is_valid):
    return {
        "task": {"numbers": numbers, "target": target, "expected_solution": "unused"},
        "is_valid": is_valid,
        "reason": "correct" if is_valid else "wrong_value",
    }


def test_analyze_report_pair_computes_task_transitions_and_metric_deltas():
    base_report = {
        "metrics": {
            "pass_at_1": 33.33,
            "format_error_rate": 10.0,
            "safe_parseable_rate": 80.0,
            "exact_number_usage_rate": 50.0,
            "mean_reward": 0.2,
        },
        "results": [
            _result([1, 2, 3], 6, False),
            _result([2, 3, 4, 5], 14, True),
            _result([1, 2, 3, 4, 5], 15, False),
        ],
    }
    treatment_report = {
        "metrics": {
            "pass_at_1": 66.67,
            "format_error_rate": 5.0,
            "safe_parseable_rate": 90.0,
            "exact_number_usage_rate": 55.0,
            "mean_reward": 0.4,
        },
        "results": [
            _result([1, 2, 3], 6, True),
            _result([2, 3, 4, 5], 14, False),
            _result([1, 2, 3, 4, 5], 15, True),
        ],
    }

    analysis = analyze_report_pair(base_report, treatment_report)

    assert analysis["base_correct_count"] == 1
    assert analysis["treatment_correct_count"] == 2
    assert analysis["transitions"] == {
        "gained": 2,
        "lost": 1,
        "both_correct": 0,
        "both_incorrect": 0,
    }
    assert analysis["transitions_by_k"]["3"]["gained"] == 1
    assert analysis["transitions_by_k"]["4"]["lost"] == 1
    assert analysis["metric_deltas"]["safe_parseable_rate"] == 10.0
