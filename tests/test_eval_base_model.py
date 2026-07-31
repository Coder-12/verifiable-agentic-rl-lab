from src.tasks.countdown.eval_utils import (
    analyze_expression,
    completion_ranking_key,
    select_best_completion,
    summarize_seed_sweep,
    trim_generated_ids,
)


def test_trim_generated_ids_removes_left_padding_and_prompt_tokens():
    output_ids = [
        [0, 0, 11, 12, 13, 14, 91, 92],
        [21, 22, 23, 24, 25, 26, 93, 94],
    ]
    attention_mask = [
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]

    trimmed = trim_generated_ids(output_ids, attention_mask)

    assert trimmed[0] == [91, 92]
    assert trimmed[1] == [93, 94]


def test_analyze_expression_reports_exact_usage():
    task = {"numbers": [1, 3, 4, 6], "target": 24}

    analysis = analyze_expression(task, "6 / (1 - 3/4)")

    assert analysis["safe_parseable"] is True
    assert analysis["used_numbers"] == [6, 1, 3, 4]
    assert analysis["exact_number_usage"] is True
    assert analysis["uses_too_few_literals"] is False
    assert analysis["introduces_forbidden_literals"] is False


def test_analyze_expression_reports_too_few_and_forbidden_literals():
    task = {"numbers": [1, 3, 4, 6], "target": 24}

    too_few = analyze_expression(task, "6 / (1 - 3)")
    forbidden = analyze_expression(task, "24 - (1 + 3)")

    assert too_few["safe_parseable"] is True
    assert too_few["exact_number_usage"] is False
    assert too_few["uses_too_few_literals"] is True
    assert too_few["introduces_forbidden_literals"] is False

    assert forbidden["safe_parseable"] is True
    assert forbidden["exact_number_usage"] is False
    assert forbidden["uses_too_few_literals"] is True
    assert forbidden["introduces_forbidden_literals"] is True


def test_analyze_expression_rejects_decimal_literals():
    task = {"numbers": [1, 3, 4, 6], "target": 24}

    analysis = analyze_expression(task, "6 / (1 - 0.75)")

    assert analysis["safe_parseable"] is False
    assert analysis["used_numbers"] is None
    assert analysis["exact_number_usage"] is False


def test_select_best_completion_prefers_valid_output():
    completions = [
        {
            "sample_index": 0,
            "is_valid": False,
            "reason": "wrong_numbers",
            "reward": 0.2,
            "analysis": {
                "exact_number_usage": False,
                "safe_parseable": True,
                "introduces_forbidden_literals": True,
                "uses_too_few_literals": False,
            },
        },
        {
            "sample_index": 1,
            "is_valid": True,
            "reason": "correct",
            "reward": 1.0,
            "analysis": {
                "exact_number_usage": True,
                "safe_parseable": True,
                "introduces_forbidden_literals": False,
                "uses_too_few_literals": False,
            },
        },
    ]

    selected = select_best_completion(completions)

    assert selected["sample_index"] == 1
    assert selected["reason"] == "correct"


def test_select_best_completion_breaks_reward_ties_with_cleaner_candidate():
    wrong_numbers = {
        "sample_index": 0,
        "is_valid": False,
        "reason": "wrong_numbers",
        "reward": 0.2,
        "analysis": {
            "exact_number_usage": False,
            "safe_parseable": True,
            "introduces_forbidden_literals": True,
            "uses_too_few_literals": False,
        },
    }
    wrong_value = {
        "sample_index": 1,
        "is_valid": False,
        "reason": "wrong_value",
        "reward": 0.2,
        "analysis": {
            "exact_number_usage": True,
            "safe_parseable": True,
            "introduces_forbidden_literals": False,
            "uses_too_few_literals": False,
        },
    }

    assert completion_ranking_key(wrong_value) > completion_ranking_key(wrong_numbers)
    assert select_best_completion([wrong_numbers, wrong_value])["sample_index"] == 1


def test_select_best_completion_is_deterministic_for_identical_scores():
    first = {
        "sample_index": 0,
        "is_valid": False,
        "reason": "wrong_value",
        "reward": 0.2,
        "analysis": {
            "exact_number_usage": True,
            "safe_parseable": True,
            "introduces_forbidden_literals": False,
            "uses_too_few_literals": False,
        },
    }
    second = {
        "sample_index": 1,
        "is_valid": False,
        "reason": "wrong_value",
        "reward": 0.2,
        "analysis": {
            "exact_number_usage": True,
            "safe_parseable": True,
            "introduces_forbidden_literals": False,
            "uses_too_few_literals": False,
        },
    }

    assert select_best_completion([first, second])["sample_index"] == 0


def test_summarize_seed_sweep_aggregates_metrics_and_task_stability():
    seed_runs = [
        {
            "metrics": {
                "oracle_pass_at_n": 10.0,
                "reranked_success_at_n": 10.0,
                "raw_completion_success_rate": 3.0,
                "format_error_rate": 12.0,
                "safe_parseable_rate": 85.0,
                "exact_number_usage_rate": 25.0,
                "mean_reward": 0.18,
                "selected_mean_reward": 0.28,
            },
            "results": [
                {
                    "task": {"numbers": [1, 2, 3], "target": 6},
                    "oracle_task_passed": True,
                    "selected_completion": {"is_valid": True},
                },
                {
                    "task": {"numbers": [4, 5, 6], "target": 15},
                    "oracle_task_passed": False,
                    "selected_completion": {"is_valid": False},
                },
            ],
        },
        {
            "metrics": {
                "oracle_pass_at_n": 20.0,
                "reranked_success_at_n": 10.0,
                "raw_completion_success_rate": 4.0,
                "format_error_rate": 10.0,
                "safe_parseable_rate": 87.0,
                "exact_number_usage_rate": 28.0,
                "mean_reward": 0.20,
                "selected_mean_reward": 0.30,
            },
            "results": [
                {
                    "task": {"numbers": [1, 2, 3], "target": 6},
                    "oracle_task_passed": True,
                    "selected_completion": {"is_valid": False},
                },
                {
                    "task": {"numbers": [4, 5, 6], "target": 15},
                    "oracle_task_passed": True,
                    "selected_completion": {"is_valid": True},
                },
            ],
        },
    ]

    summary = summarize_seed_sweep(seed_runs)

    oracle_stats = summary["aggregate_metrics"]["oracle_pass_at_n"]
    assert oracle_stats["values"] == [10.0, 20.0]
    assert oracle_stats["mean"] == 15.0
    assert oracle_stats["min"] == 10.0
    assert oracle_stats["max"] == 20.0

    stability = summary["stability_summary"]
    assert stability["total_runs"] == 2
    assert stability["oracle_tasks_solved_in_any_seed"] == 2
    assert stability["oracle_tasks_solved_in_all_seeds"] == 1
    assert stability["reranked_tasks_solved_in_any_seed"] == 2
    assert stability["reranked_tasks_solved_in_all_seeds"] == 0

    task_stability = summary["task_stability"]
    assert task_stability[0]["oracle_success_count"] == 2
    assert task_stability[0]["reranked_success_count"] == 1
    assert task_stability[1]["oracle_success_count"] == 1
    assert task_stability[1]["reranked_success_count"] == 1
