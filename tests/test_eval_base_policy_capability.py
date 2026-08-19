from scripts.eval_base_policy_capability import (
    allocate_counts_by_k,
    compute_group_diagnostics,
    summarize_by_k,
    summarize_groups,
)


def _sample(sample_index, reward, is_valid=False, reason="wrong_value", answer=None):
    return {
        "sample_index": sample_index,
        "completion": f"completion-{sample_index}-{answer}",
        "extracted_math": answer,
        "is_valid": is_valid,
        "reason": "correct" if is_valid else reason,
        "reward": reward,
        "analysis": {
            "safe_parseable": reward > 0.0,
            "exact_number_usage": reason in {"wrong_value", "correct"} or is_valid,
            "uses_too_few_literals": False,
            "introduces_forbidden_literals": False,
        },
        "generated_token_count": 12,
        "hit_length_limit": False,
    }


def test_allocate_counts_by_k_balances_total_prompts():
    counts = allocate_counts_by_k(total_prompts=200, min_numbers=3, max_numbers=5)

    assert counts == {3: 67, 4: 67, 5: 66}
    assert sum(counts.values()) == 200


def test_compute_group_diagnostics_separates_success_and_structural_variance():
    success_group = [
        _sample(0, 0.3, reason="wrong_value", answer="1 + 2 + 3"),
        _sample(1, 1.0, is_valid=True, answer="1 * 2 * 3"),
        _sample(2, 0.05, reason="wrong_numbers", answer="1 + 2"),
        _sample(3, 0.0, reason="format_error", answer=None),
    ]
    structural_group = [
        _sample(0, 0.0, reason="format_error", answer=None),
        _sample(1, 0.05, reason="wrong_numbers", answer="1 + 2"),
        _sample(2, 0.3, reason="wrong_value", answer="1 + 2 + 3"),
        _sample(3, 0.05, reason="wrong_numbers", answer="1 + 3"),
    ]

    success_diagnostics = compute_group_diagnostics(success_group)
    structural_diagnostics = compute_group_diagnostics(structural_group)

    assert success_diagnostics["has_correct"] is True
    assert success_diagnostics["success_variance"] is True
    assert success_diagnostics["structural_only_variance"] is False

    assert structural_diagnostics["has_correct"] is False
    assert structural_diagnostics["rewards_differ"] is True
    assert structural_diagnostics["success_variance"] is False
    assert structural_diagnostics["structural_only_variance"] is True


def test_compute_group_diagnostics_detects_duplicate_answers_and_all_zero_groups():
    samples = [
        _sample(0, 0.0, reason="format_error", answer=None),
        _sample(1, 0.0, reason="format_error", answer=None),
        _sample(2, 0.0, reason="parse_error", answer="1 +"),
        _sample(3, 0.0, reason="parse_error", answer="1 +"),
    ]

    diagnostics = compute_group_diagnostics(samples)

    assert diagnostics["all_zero_rewards"] is True
    assert diagnostics["duplicate_answer"] is True
    assert diagnostics["duplicate_completion"] is False


def test_summarize_groups_reports_capability_and_trainability_rates():
    groups = [
        {
            "k": 3,
            "samples": [
                _sample(0, 1.0, is_valid=True, answer="1 * 2 * 3"),
                _sample(1, 0.3, reason="wrong_value", answer="1 + 2 + 3"),
                _sample(2, 0.05, reason="wrong_numbers", answer="1 + 2"),
                _sample(3, 0.0, reason="format_error", answer=None),
            ],
            **compute_group_diagnostics(
                [
                    _sample(0, 1.0, is_valid=True, answer="1 * 2 * 3"),
                    _sample(1, 0.3, reason="wrong_value", answer="1 + 2 + 3"),
                    _sample(2, 0.05, reason="wrong_numbers", answer="1 + 2"),
                    _sample(3, 0.0, reason="format_error", answer=None),
                ]
            ),
        },
        {
            "k": 4,
            "samples": [
                _sample(0, 0.0, reason="format_error", answer=None),
                _sample(1, 0.05, reason="wrong_numbers", answer="1 + 2"),
                _sample(2, 0.3, reason="wrong_value", answer="1 + 2 + 3 + 4"),
                _sample(3, 0.05, reason="wrong_numbers", answer="1 + 3"),
            ],
            **compute_group_diagnostics(
                [
                    _sample(0, 0.0, reason="format_error", answer=None),
                    _sample(1, 0.05, reason="wrong_numbers", answer="1 + 2"),
                    _sample(2, 0.3, reason="wrong_value", answer="1 + 2 + 3 + 4"),
                    _sample(3, 0.05, reason="wrong_numbers", answer="1 + 3"),
                ]
            ),
        },
    ]

    metrics = summarize_groups(groups)
    metrics_by_k = summarize_by_k(groups)

    assert metrics["total_groups"] == 2
    assert metrics["total_completions"] == 8
    assert metrics["sample_1_success_rate"] == 50.0
    assert metrics["correct_rollout_rate"] == 12.5
    assert metrics["group_success_rate"] == 50.0
    assert metrics["nonzero_reward_variance_group_rate"] == 100.0
    assert metrics["success_variance_group_rate"] == 50.0
    assert metrics["structural_only_variance_group_rate"] == 50.0

    assert metrics_by_k["3"]["group_success_rate"] == 100.0
    assert metrics_by_k["4"]["group_success_rate"] == 0.0
