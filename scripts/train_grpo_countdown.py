"""
GRPO smoke training for Milestone 1 Countdown RLVR.

Design goals:
- Run cleanly on a single Kaggle T4 GPU.
- Keep the run small enough to validate the RL loop before heavier tuning.
- Preserve the same prompt/verifier/reward logic used in baseline evaluation.
- Save the generated training tasks and run config for reproducibility.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tasks.countdown.dataset import generate_countdown_dataset
from src.tasks.countdown.prompts import (
    COUNTDOWN_SYSTEM_PROMPT,
    extract_answer,
    format_countdown_prompt,
)
from src.tasks.countdown.reward import (
    REWARD_MAPPING,
    REWARD_PRESETS,
    SHAPED_REWARD_PRESET,
    get_reward_mapping,
    reward_from_verification_result,
)
from src.tasks.countdown.verifier import VerificationStatus, verify_countdown_expression

if TYPE_CHECKING:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def set_seed(seed: int) -> None:
    """Seeds Python and PyTorch for reproducible smoke runs."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_dtype_from_torch(torch_module: Any) -> tuple[Any, bool, bool]:
    """Chooses mixed precision from CUDA hardware capability, not framework claims."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("GRPO smoke training requires CUDA. This script is intended for Kaggle T4 or similar GPUs.")

    major, _minor = torch_module.cuda.get_device_capability()
    # True native BF16 acceleration starts with Ampere-class CUDA GPUs.
    # T4 is compute capability 7.5, so use FP16 even if PyTorch reports BF16 support.
    if major >= 8:
        return torch_module.bfloat16, True, False
    return torch_module.float16, False, True


def resolve_training_dtype() -> tuple[Any, bool, bool]:
    """Chooses a practical mixed-precision mode for the active CUDA device."""
    import torch

    return resolve_training_dtype_from_torch(torch)


def serialize_torch_dtype(dtype: Any) -> str:
    """
    Converts a torch dtype to a JSON-serializable value.

    GRPOConfig is logged through Trainer callbacks, so any nested field must be
    serializable by json.dumps.
    """
    if hasattr(dtype, "name") and dtype.name:
        return dtype.name
    dtype_str = str(dtype)
    if dtype_str.startswith("torch."):
        return dtype_str.split(".", maxsplit=1)[1]
    return dtype_str


def disable_peft_torchao_adapter_path() -> None:
    """
    Force PEFT to use the standard LoRA path instead of the torchao adapter path.

    Kaggle images may include an incompatible torchao install. GRPO only needs
    vanilla LoRA here, so we disable the torchao dispatch branch proactively to
    keep training environment-independent.
    """
    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.torchao as peft_torchao

    def _torchao_unavailable() -> bool:
        return False

    peft_import_utils.is_torchao_available = _torchao_unavailable
    peft_torchao.is_torchao_available = _torchao_unavailable


def _allocate_balanced_counts_by_k(
    *,
    num_samples: int,
    min_numbers: int,
    max_numbers: int,
) -> dict[int, int]:
    """Allocates a near-equal number of generated tasks to each puzzle size."""
    sizes = list(range(min_numbers, max_numbers + 1))
    base_count, remainder = divmod(num_samples, len(sizes))
    return {
        k: base_count + (1 if index < remainder else 0)
        for index, k in enumerate(sizes)
    }


def validate_dataset_config(
    *,
    num_samples: int,
    min_numbers: int,
    max_numbers: int,
) -> None:
    """Validates training task generation controls before launching training."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive. Got {num_samples}.")
    if min_numbers <= 0:
        raise ValueError(f"min_numbers must be positive. Got {min_numbers}.")
    if max_numbers <= 0:
        raise ValueError(f"max_numbers must be positive. Got {max_numbers}.")
    if min_numbers > max_numbers:
        raise ValueError(
            "min_numbers must be <= max_numbers. "
            f"Got {min_numbers=} and {max_numbers=}."
        )


def generate_training_tasks(
    *,
    num_samples: int,
    min_numbers: int,
    max_numbers: int,
    balanced_by_k: bool,
) -> list[dict[str, Any]]:
    """Generates training tasks with optional balanced puzzle-size control."""
    validate_dataset_config(
        num_samples=num_samples,
        min_numbers=min_numbers,
        max_numbers=max_numbers,
    )

    if not balanced_by_k:
        return generate_countdown_dataset(
            num_samples=num_samples,
            min_numbers=min_numbers,
            max_numbers=max_numbers,
        )

    tasks: list[dict[str, Any]] = []
    for k, count in _allocate_balanced_counts_by_k(
        num_samples=num_samples,
        min_numbers=min_numbers,
        max_numbers=max_numbers,
    ).items():
        tasks.extend(
            generate_countdown_dataset(
                num_samples=count,
                min_numbers=k,
                max_numbers=k,
            )
        )

    random.shuffle(tasks)
    return tasks


def summarize_k_distribution(train_dataset: list[dict[str, Any]]) -> dict[str, int]:
    """Counts generated training examples by puzzle size for reproducibility."""
    counts = Counter(str(len(row["numbers"])) for row in train_dataset)
    return dict(sorted(counts.items()))


def sha256_file(path: Path) -> str:
    """Returns a stable content hash for a generated artifact or source file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Hashes the task-defining source files used by a training run."""
    relative_paths = (
        "src/tasks/countdown/dataset.py",
        "src/tasks/countdown/prompts.py",
        "src/tasks/countdown/reward.py",
        "src/tasks/countdown/verifier.py",
        "scripts/train_grpo_countdown.py",
    )
    return {
        relative_path: sha256_file(Path(PROJECT_ROOT) / relative_path)
        for relative_path in relative_paths
    }


def git_metadata() -> dict[str, Any]:
    """Records repository state without requiring git to be installed at runtime."""
    def run_git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "is_dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str | None]:
    """Captures the packages that materially affect TRL GRPO behavior."""
    versions: dict[str, str | None] = {}
    for package_name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def cuda_hardware_metadata(torch_module: Any) -> dict[str, Any]:
    """Captures the active accelerator needed to reproduce mixed-precision behavior."""
    if not torch_module.cuda.is_available():
        return {"cuda_available": False}

    device_index = torch_module.cuda.current_device()
    capability = torch_module.cuda.get_device_capability(device_index)
    return {
        "cuda_available": True,
        "device_name": torch_module.cuda.get_device_name(device_index),
        "device_capability": f"{capability[0]}.{capability[1]}",
        "cuda_runtime": getattr(torch_module.version, "cuda", None),
    }


def resolved_revision(resource: Any) -> str | None:
    """Returns the Hub commit resolved by Transformers when it is available."""
    if hasattr(resource, "_commit_hash") and resource._commit_hash:
        return str(resource._commit_hash)
    init_kwargs = getattr(resource, "init_kwargs", {})
    commit_hash = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
    return str(commit_hash) if commit_hash else None


def group_normalized_advantages(rewards: list[float]) -> list[float]:
    """Computes the relative reward signal GRPO uses within one rollout group."""
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    if variance == 0.0:
        return [0.0] * len(rewards)
    standard_deviation = variance**0.5
    return [(reward - mean_reward) / standard_deviation for reward in rewards]


@dataclass
class TrainingRewardDiagnostics:
    """Aggregates verifier outcomes from GRPO rollout groups without storing text."""

    num_generations: int
    total_completions: int = 0
    total_groups: int = 0
    correct_rollout_count: int = 0
    successful_variance_group_count: int = 0
    zero_variance_group_count: int = 0
    all_zero_group_count: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    status_counts_by_k: dict[str, Counter[str]] = field(default_factory=dict)
    reward_counts: Counter[str] = field(default_factory=Counter)

    def record(self, records: list[dict[str, Any]]) -> None:
        """Adds one or more contiguous prompt groups emitted by GRPO rollouts."""
        if len(records) % self.num_generations != 0:
            raise ValueError(
                "GRPO reward diagnostics require complete prompt groups. "
                f"Got {len(records)} records for num_generations={self.num_generations}."
            )

        for record in records:
            status = record["status"]
            k = str(record["k"])
            self.total_completions += 1
            self.correct_rollout_count += int(status == VerificationStatus.CORRECT.value)
            self.status_counts[status] += 1
            self.status_counts_by_k.setdefault(k, Counter())[status] += 1
            self.reward_counts[str(record["reward"])] += 1

        for start in range(0, len(records), self.num_generations):
            group = records[start : start + self.num_generations]
            rewards = [record["reward"] for record in group]
            advantages = group_normalized_advantages(rewards)
            self.total_groups += 1
            if not any(advantages):
                self.zero_variance_group_count += 1
            if all(reward == 0.0 for reward in rewards):
                self.all_zero_group_count += 1
            if any(record["status"] == VerificationStatus.CORRECT.value for record in group) and any(
                reward < 1.0 for reward in rewards
            ):
                self.successful_variance_group_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Serializes aggregate diagnostics and rates for a persisted training artifact."""
        group_count = self.total_groups
        return {
            "num_generations": self.num_generations,
            "total_completions": self.total_completions,
            "total_groups": group_count,
            "correct_rollout_count": self.correct_rollout_count,
            "correct_rollout_rate": self.correct_rollout_count / self.total_completions
            if self.total_completions
            else 0.0,
            "successful_variance_group_count": self.successful_variance_group_count,
            "successful_variance_group_rate": self.successful_variance_group_count / group_count
            if group_count
            else 0.0,
            "zero_variance_group_count": self.zero_variance_group_count,
            "zero_variance_group_rate": self.zero_variance_group_count / group_count
            if group_count
            else 0.0,
            "all_zero_group_count": self.all_zero_group_count,
            "all_zero_group_rate": self.all_zero_group_count / group_count if group_count else 0.0,
            "status_counts": dict(self.status_counts),
            "status_counts_by_k": {
                k: dict(counts) for k, counts in sorted(self.status_counts_by_k.items())
            },
            "reward_counts": dict(self.reward_counts),
        }


def prepare_train_dataset(
    num_samples: int,
    tokenizer: Any,
    *,
    min_numbers: int = 3,
    max_numbers: int = 5,
    balanced_by_k: bool = False,
) -> list[dict[str, Any]]:
    """
    Generates fresh Countdown tasks and stores prompt + verifier inputs.
    GRPOTrainer requires a 'prompt' column. We pre-apply the chat template so
    training and evaluation use the same prompt format.
    """
    print(
        f"Generating {num_samples} fresh training tasks "
        f"(k={min_numbers}..{max_numbers}, balanced_by_k={balanced_by_k})..."
    )
    raw_data = generate_training_tasks(
        num_samples=num_samples,
        min_numbers=min_numbers,
        max_numbers=max_numbers,
        balanced_by_k=balanced_by_k,
    )

    dataset_rows = []
    for task in raw_data:
        user_prompt = format_countdown_prompt(task["numbers"], task["target"])
        messages = [
            {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        dataset_rows.append(
            {
                "prompt": prompt_text,
                "numbers": task["numbers"],
                "target": task["target"],
            }
        )

    return dataset_rows


def build_countdown_grpo_reward(
    *,
    reward_mapping: dict[str, float],
    diagnostics: TrainingRewardDiagnostics | None = None,
) -> Any:
    """Builds a GRPO reward callable bound to one named, fixed reward mapping."""
    def reward_function(
        completions: list[str],
        target: list[int],
        numbers: list[list[int]],
        **kwargs: Any,
    ) -> list[float]:
        del kwargs

        rewards: list[float] = []
        records: list[dict[str, Any]] = []
        for completion, completion_target, completion_numbers in zip(
            completions, target, numbers, strict=True
        ):
            extracted_math = extract_answer(completion)
            if extracted_math is None:
                status = "format_error"
                reward = 0.0
            else:
                verdict = verify_countdown_expression(
                    extracted_math,
                    completion_target,
                    completion_numbers,
                )
                status = verdict.reason.value
                reward = reward_from_verification_result(verdict, reward_mapping)

            rewards.append(reward)
            records.append(
                {
                    "status": status,
                    "reward": reward,
                    "k": len(completion_numbers),
                }
            )

        if diagnostics is not None:
            diagnostics.record(records)
        return rewards

    return reward_function


def countdown_grpo_reward(
    completions: list[str],
    target: list[int],
    numbers: list[list[int]],
    **kwargs: Any,
) -> list[float]:
    """
    Countdown verifier reward in the format expected by TRL GRPOTrainer.
    The trainer forwards dataset columns such as 'target' and 'numbers'.
    """
    del kwargs

    return build_countdown_grpo_reward(reward_mapping=dict(REWARD_MAPPING))(
        completions=completions,
        target=target,
        numbers=numbers,
    )


def build_run_config(
    *,
    args: argparse.Namespace,
    train_dataset: list[dict[str, Any]],
    train_dtype: Any,
    bf16_enabled: bool,
    fp16_enabled: bool,
    python_version: str,
    torch_version: str,
    transformers_version: str,
    train_dataset_sha256: str | None = None,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Captures the training configuration for reproducibility."""
    reward_preset = getattr(args, "reward_preset", SHAPED_REWARD_PRESET)
    return {
        "model": args.model,
        "train_samples": len(train_dataset),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "optimizer": "adamw_torch",
        "torch_dtype": str(train_dtype),
        "bf16": bf16_enabled,
        "fp16": fp16_enabled,
        "reward_preset": reward_preset,
        "reward_mapping": get_reward_mapping(reward_preset),
        "dataset_config": {
            "min_numbers": args.min_numbers,
            "max_numbers": args.max_numbers,
            "balanced_by_k": args.balanced_by_k,
            "k_distribution": summarize_k_distribution(train_dataset),
            "sha256": train_dataset_sha256,
            "historical_pairing_note": (
                "Same-seed regenerated data; historical train_dataset.jsonl artifacts "
                "were not retained, so byte identity with shaped runs is not verified."
            ),
        },
        "hyperparameters": {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "scale_rewards": "group",
            "loss_type": "dapo",
            "beta": 0.0,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "target_modules": list(LORA_TARGET_MODULES),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "source_hashes": source_hashes(),
        "git": git_metadata(),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "environment": {
            "python": python_version,
            "torch": torch_version,
            "transformers": transformers_version,
            "packages": package_versions(),
            "hardware": hardware or {},
        },
        "command": {
            "argv": sys.argv,
        },
    }


def validate_effective_batch_size(
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
) -> None:
    """
    GRPO requires the effective batch size to be divisible by num_generations.
    For the current single-GPU smoke run, that is:
    per_device_train_batch_size * gradient_accumulation_steps.
    """
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    if effective_batch_size % num_generations != 0:
        raise ValueError(
            "Effective batch size must be divisible by num_generations. "
            f"Got {effective_batch_size=} and {num_generations=}."
        )


def validate_output_dir(output_dir: Path, overwrite_output_dir: bool) -> None:
    """Prevents accidentally mixing or replacing separate training runs."""
    if not output_dir.exists():
        return

    existing_files = [path for path in output_dir.iterdir() if path.name != ".DS_Store"]
    if existing_files and not overwrite_output_dir:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a fresh --output-dir or pass --overwrite-output-dir explicitly."
        )


def validate_checkpoint_config(*, save_steps: int, max_steps: int, save_total_limit: int) -> None:
    """Validates checkpoint retention settings before launching an expensive run."""
    if save_steps <= 0:
        raise ValueError(f"save_steps must be positive. Got {save_steps}.")
    if save_steps > max_steps:
        raise ValueError(f"save_steps must be <= max_steps. Got {save_steps=} and {max_steps=}.")
    if save_total_limit <= 0:
        raise ValueError(f"save_total_limit must be positive. Got {save_total_limit}.")


def validate_grpo_config_support(grpo_config_class: Any) -> None:
    """Fails early if the installed TRL cannot honor the frozen group scaling setting."""
    if "scale_rewards" not in inspect.signature(grpo_config_class).parameters:
        raise RuntimeError(
            "The installed TRL GRPOConfig does not support scale_rewards. "
            "Install the experiment's compatible TRL version before running this protocol."
        )


def main() -> None:
    from datasets import Dataset
    import torch
    import transformers
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    validate_grpo_config_support(GRPOConfig)

    parser = argparse.ArgumentParser(description="Run GRPO smoke training for Countdown.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="models/Qwen2.5-1.5B-GRPO-Smoke")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--reward-preset",
        choices=sorted(REWARD_PRESETS),
        default=SHAPED_REWARD_PRESET,
        help="Fixed verifier-reward mapping used only for this training run.",
    )
    parser.add_argument("--min-numbers", type=int, default=3)
    parser.add_argument("--max-numbers", type=int, default=5)
    parser.add_argument(
        "--balanced-by-k",
        action="store_true",
        help="Generate a near-equal number of training tasks for each puzzle size.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Checkpoint interval. Defaults to max_steps so smoke runs save only the final checkpoint.",
    )
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()

    if args.save_steps is None:
        args.save_steps = args.max_steps

    set_seed(args.seed)
    train_dtype, bf16_enabled, fp16_enabled = resolve_training_dtype()
    disable_peft_torchao_adapter_path()
    validate_effective_batch_size(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
    )
    validate_dataset_config(
        num_samples=args.train_samples,
        min_numbers=args.min_numbers,
        max_numbers=args.max_numbers,
    )
    validate_checkpoint_config(
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        save_total_limit=args.save_total_limit,
    )

    output_dir = Path(args.output_dir)
    validate_output_dir(output_dir, args.overwrite_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting GRPO smoke training on CUDA with dtype={train_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset_rows = prepare_train_dataset(
        args.train_samples,
        tokenizer,
        min_numbers=args.min_numbers,
        max_numbers=args.max_numbers,
        balanced_by_k=args.balanced_by_k,
    )
    train_dataset = Dataset.from_list(train_dataset_rows)
    train_dataset_path = output_dir / "train_dataset.jsonl"
    train_dataset.to_json(str(train_dataset_path))
    model_config = AutoConfig.from_pretrained(args.model)

    run_config = build_run_config(
        args=args,
        train_dataset=train_dataset_rows,
        train_dtype=train_dtype,
        bf16_enabled=bf16_enabled,
        fp16_enabled=fp16_enabled,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        train_dataset_sha256=sha256_file(train_dataset_path),
        model_revision=resolved_revision(model_config),
        tokenizer_revision=resolved_revision(tokenizer),
        hardware=cuda_hardware_metadata(torch),
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        logging_steps=10,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        report_to="tensorboard",
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={"torch_dtype": serialize_torch_dtype(train_dtype)},
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
    )

    training_diagnostics = TrainingRewardDiagnostics(args.num_generations)
    training_reward = build_countdown_grpo_reward(
        reward_mapping=get_reward_mapping(args.reward_preset),
        diagnostics=training_diagnostics,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[training_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting GRPO smoke run...")
    trainer.train()

    print(f"Training complete. Saving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with (output_dir / "training_reward_diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(training_diagnostics.to_dict(), file, indent=2)
    with (output_dir / "trainer_log_history.json").open("w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, indent=2, default=str)


if __name__ == "__main__":
    main()
