import importlib.util
from argparse import Namespace

import pytest

from scripts.train_grpo_countdown import (
    TrainingRewardDiagnostics,
    build_countdown_grpo_reward,
    build_run_config,
    countdown_grpo_reward,
    disable_peft_torchao_adapter_path,
    generate_training_tasks,
    group_normalized_advantages,
    prepare_train_dataset,
    resolve_training_dtype_from_torch,
    serialize_torch_dtype,
    summarize_k_distribution,
    validate_checkpoint_config,
    validate_dataset_config,
    validate_output_dir,
    validate_effective_batch_size,
    validate_grpo_config_support,
)
from src.tasks.countdown.reward import CORRECTNESS_ONLY_REWARD_PRESET, get_reward_mapping


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["content"] for message in messages)


class FakeCuda:
    def __init__(self, major):
        self.major = major

    def is_available(self):
        return True

    def get_device_capability(self):
        return self.major, 0


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self, major):
        self.cuda = FakeCuda(major)


class FakeGRPOConfigWithScaling:
    def __init__(self, scale_rewards=None):
        self.scale_rewards = scale_rewards


class FakeGRPOConfigWithoutScaling:
    def __init__(self, temperature=None):
        self.temperature = temperature


def test_countdown_grpo_reward_matches_reward_buckets():
    completions = [
        "<reasoning>ok</reasoning><answer>(3 + 4) * 2</answer>",
        "<reasoning>wrong</reasoning><answer>2 + 3 + 4</answer>",
        "<reasoning>bad nums</reasoning><answer>2 + 3</answer>",
        "no answer tags here",
    ]
    rewards = countdown_grpo_reward(
        completions=completions,
        target=[14, 14, 14, 14],
        numbers=[[2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4]],
    )

    assert rewards == [1.0, 0.3, 0.05, 0.0]


def test_correctness_only_reward_preset_removes_all_partial_credit():
    reward_mapping = get_reward_mapping(CORRECTNESS_ONLY_REWARD_PRESET)
    diagnostics = TrainingRewardDiagnostics(num_generations=4)
    reward_function = build_countdown_grpo_reward(
        reward_mapping=reward_mapping,
        diagnostics=diagnostics,
    )

    rewards = reward_function(
        completions=[
            "<reasoning>ok</reasoning><answer>(3 + 4) * 2</answer>",
            "<reasoning>wrong</reasoning><answer>2 + 3 + 4</answer>",
            "<reasoning>numbers</reasoning><answer>2 + 3</answer>",
            "missing answer tags",
        ],
        target=[14, 14, 14, 14],
        numbers=[[2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4]],
    )

    assert rewards == [1.0, 0.0, 0.0, 0.0]
    assert diagnostics.to_dict()["successful_variance_group_count"] == 1
    assert diagnostics.to_dict()["zero_variance_group_count"] == 0
    assert diagnostics.to_dict()["status_counts"] == {
        "correct": 1,
        "wrong_value": 1,
        "wrong_numbers": 1,
        "format_error": 1,
    }


def test_correctness_only_reward_creates_relative_group_advantages_only_on_success():
    assert group_normalized_advantages([0.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 0.0]

    advantages = group_normalized_advantages([1.0, 0.0, 0.0, 0.0])

    assert advantages[0] > 0.0
    assert all(advantage < 0.0 for advantage in advantages[1:])
    assert sum(advantages) == pytest.approx(0.0)


def test_resolve_training_dtype_uses_fp16_for_t4_class_gpu():
    dtype, bf16, fp16 = resolve_training_dtype_from_torch(FakeTorch(major=7))

    assert dtype == "fp16"
    assert bf16 is False
    assert fp16 is True


def test_resolve_training_dtype_allows_bf16_for_ampere_or_newer():
    dtype, bf16, fp16 = resolve_training_dtype_from_torch(FakeTorch(major=8))

    assert dtype == "bf16"
    assert bf16 is True
    assert fp16 is False


def test_prepare_train_dataset_contains_prompt_and_verifier_inputs():
    dataset = prepare_train_dataset(num_samples=5, tokenizer=DummyTokenizer())

    assert len(dataset) == 5
    row = dataset[0]
    assert set(row) == {"prompt", "numbers", "target"}
    assert "Numbers:" in row["prompt"]
    assert "Target:" in row["prompt"]
    assert isinstance(row["numbers"], list)
    assert isinstance(row["target"], int)


def test_default_training_task_generation_supports_each_configured_size():
    for k in (3, 4, 5):
        dataset = generate_training_tasks(
            num_samples=5,
            min_numbers=k,
            max_numbers=k,
            balanced_by_k=False,
        )

        assert len(dataset) == 5
        assert {len(row["numbers"]) for row in dataset} == {k}


def test_balanced_training_task_generation_spans_k3_k4_evenly():
    dataset = generate_training_tasks(
        num_samples=20,
        min_numbers=3,
        max_numbers=4,
        balanced_by_k=True,
    )
    distribution = summarize_k_distribution(dataset)

    assert distribution == {"3": 10, "4": 10}


def test_balanced_training_task_generation_handles_remainder():
    dataset = generate_training_tasks(
        num_samples=21,
        min_numbers=3,
        max_numbers=4,
        balanced_by_k=True,
    )
    distribution = summarize_k_distribution(dataset)

    assert distribution == {"3": 11, "4": 10}


def test_prepare_train_dataset_accepts_curriculum_controls():
    dataset = prepare_train_dataset(
        num_samples=8,
        tokenizer=DummyTokenizer(),
        min_numbers=3,
        max_numbers=4,
        balanced_by_k=True,
    )

    assert summarize_k_distribution(dataset) == {"3": 4, "4": 4}


def test_validate_dataset_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="num_samples"):
        validate_dataset_config(num_samples=0, min_numbers=3, max_numbers=4)

    with pytest.raises(ValueError, match="min_numbers"):
        validate_dataset_config(num_samples=10, min_numbers=0, max_numbers=4)

    with pytest.raises(ValueError, match="min_numbers must be <= max_numbers"):
        validate_dataset_config(num_samples=10, min_numbers=5, max_numbers=4)


def test_validate_effective_batch_size_accepts_divisible_case():
    validate_effective_batch_size(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
    )


def test_validate_effective_batch_size_rejects_invalid_case():
    with pytest.raises(ValueError):
        validate_effective_batch_size(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            num_generations=4,
        )


def test_validate_checkpoint_config_accepts_sweep_case():
    validate_checkpoint_config(save_steps=50, max_steps=300, save_total_limit=10)


def test_validate_checkpoint_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_checkpoint_config(save_steps=0, max_steps=300, save_total_limit=10)

    with pytest.raises(ValueError):
        validate_checkpoint_config(save_steps=350, max_steps=300, save_total_limit=10)

    with pytest.raises(ValueError):
        validate_checkpoint_config(save_steps=50, max_steps=300, save_total_limit=0)


def test_validate_grpo_config_support_requires_group_scaling_option():
    validate_grpo_config_support(FakeGRPOConfigWithScaling)

    with pytest.raises(RuntimeError, match="scale_rewards"):
        validate_grpo_config_support(FakeGRPOConfigWithoutScaling)


def test_build_run_config_captures_core_training_metadata():
    args = Namespace(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        train_samples=10,
        output_dir="models/test",
        seed=42,
        max_steps=5,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=128,
        temperature=0.7,
        top_p=0.95,
        min_numbers=3,
        max_numbers=4,
        balanced_by_k=True,
        save_steps=50,
        save_total_limit=10,
        overwrite_output_dir=False,
    )
    dataset = prepare_train_dataset(num_samples=2, tokenizer=DummyTokenizer())

    config = build_run_config(
        args=args,
        train_dataset=dataset,
        train_dtype="torch.float16",
        bf16_enabled=False,
        fp16_enabled=True,
        python_version="3.11.0",
        torch_version="2.2.0",
        transformers_version="5.12.1",
    )

    assert config["model"] == args.model
    assert config["train_samples"] == 2
    assert config["seed"] == 42
    assert config["hyperparameters"]["num_generations"] == 4
    assert config["hyperparameters"]["save_steps"] == 50
    assert config["hyperparameters"]["save_total_limit"] == 10
    assert config["dataset_config"]["min_numbers"] == 3
    assert config["dataset_config"]["max_numbers"] == 4
    assert config["dataset_config"]["balanced_by_k"] is True
    assert config["dataset_config"]["k_distribution"] == summarize_k_distribution(dataset)
    assert config["reward_mapping"]["correct"] == 1.0
    assert config["reward_mapping"]["wrong_value"] == 0.3
    assert config["reward_mapping"]["wrong_numbers"] == 0.05
    assert config["reward_preset"] == "shaped"
    assert config["hyperparameters"]["scale_rewards"] == "group"
    assert config["lora"]["r"] == 16
    assert "src/tasks/countdown/verifier.py" in config["source_hashes"]


def test_validate_output_dir_rejects_non_empty_directory(tmp_path):
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        validate_output_dir(output_dir, overwrite_output_dir=False)


def test_validate_output_dir_allows_explicit_overwrite(tmp_path):
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")

    validate_output_dir(output_dir, overwrite_output_dir=True)


def test_serialize_torch_dtype_returns_json_safe_string():
    assert serialize_torch_dtype("torch.float16") == "float16"
    assert serialize_torch_dtype("torch.bfloat16") == "bfloat16"
    assert serialize_torch_dtype("float32") == "float32"


def test_disable_peft_torchao_adapter_path_forces_standard_lora():
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed in this environment")

    disable_peft_torchao_adapter_path()

    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.torchao as peft_torchao

    assert peft_import_utils.is_torchao_available() is False
    assert peft_torchao.is_torchao_available() is False
