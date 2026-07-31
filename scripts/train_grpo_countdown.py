"""
GRPO smoke training for Milestone 1 Countdown RLVR.

Design goals:
- Run cleanly on a single Kaggle T4 GPU.
- Keep the run small enough to validate the RL loop before heavier tuning.
- Preserve the same prompt/verifier/reward logic used in baseline evaluation.
- Save the generated training tasks and run config for reproducibility.
"""

import argparse
import json
import os
import platform
import random
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
from src.tasks.countdown.reward import REWARD_MAPPING, reward_from_verification_result
from src.tasks.countdown.verifier import verify_countdown_expression

if TYPE_CHECKING:
    import torch
    import transformers
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer


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


def prepare_train_dataset(num_samples: int, tokenizer: Any) -> list[dict[str, Any]]:
    """
    Generates fresh Countdown tasks and stores prompt + verifier inputs.
    GRPOTrainer requires a 'prompt' column. We pre-apply the chat template so
    training and evaluation use the same prompt format.
    """
    print(f"Generating {num_samples} fresh training tasks...")
    raw_data = generate_countdown_dataset(num_samples=num_samples)

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

    rewards: list[float] = []
    for completion, completion_target, completion_numbers in zip(
        completions, target, numbers, strict=True
    ):
        extracted_math = extract_answer(completion)
        if extracted_math is None:
            rewards.append(0.0)
            continue

        verdict = verify_countdown_expression(
            extracted_math,
            completion_target,
            completion_numbers,
        )
        rewards.append(reward_from_verification_result(verdict))

    return rewards


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
) -> dict[str, Any]:
    """Captures the training configuration for reproducibility."""
    return {
        "model": args.model,
        "train_samples": len(train_dataset),
        "output_dir": args.output_dir,
        "seed": args.seed,
        "optimizer": "adamw_torch",
        "torch_dtype": str(train_dtype),
        "bf16": bf16_enabled,
        "fp16": fp16_enabled,
        "reward_mapping": dict(REWARD_MAPPING),
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
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
        },
        "environment": {
            "python": python_version,
            "torch": torch_version,
            "transformers": transformers_version,
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


def main() -> None:
    from datasets import Dataset
    import torch
    import transformers
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

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

    train_dataset_rows = prepare_train_dataset(args.train_samples, tokenizer)
    train_dataset = Dataset.from_list(train_dataset_rows)
    train_dataset.to_json(str(output_dir / "train_dataset.jsonl"))

    run_config = build_run_config(
        args=args,
        train_dataset=train_dataset_rows,
        train_dtype=train_dtype,
        bf16_enabled=bf16_enabled,
        fp16_enabled=fp16_enabled,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
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
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[countdown_grpo_reward],
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


if __name__ == "__main__":
    main()
