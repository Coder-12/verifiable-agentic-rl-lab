"""
Dataset Generation Script for Milestone 1: Baseline Evaluation.

Hardware & Precision Acknowledgment:
As requested, for the upcoming model evaluation steps, we will load the Qwen base 
models in native float16 or bfloat16. We will NOT use 4-bit or 8-bit quantization, 
ensuring the model's reasoning capabilities remain strictly undegraded and intact 
on your Mac M1.

This script strictly generates the static, frozen JSON datasets so that all future 
model executions (0.5B debug and 1.5B main) evaluate against the exact same targets.

Edge cases handled:
- Reproducibility: Sets a fixed global seed (42) so the exact same puzzles are 
  generated if the script is rerun.
- Path Management: Safely checks for and creates the `data/` directory using pathlib 
  to avoid FileNotFoundError on fresh clones.
- Data Integrity: Serializes output with indent=2 for human-readable debugging during 
  the manual failure analysis phase.
"""
import os
import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tasks.countdown.dataset import generate_countdown_dataset


def main() -> None:
    """
    Generates and freezes the static datasets required for Step 4 of Milestone 1.
    """
    # 1. Set seed for strict reproducibility across reruns
    random.seed(42)
    
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating debug simple dataset (20 samples)...")
    eval_20 = generate_countdown_dataset(num_samples=20)
    path_20 = data_dir / "countdown_eval_simple_20.json"
    
    with path_20.open("w", encoding="utf-8") as f:
        json.dump(eval_20, f, indent=2)
    print(f"-> Save 20 debug simple dataset to {path_20}")
    
    print("Generating debug dataset (50 samples)...")
    eval_50 = generate_countdown_dataset(num_samples=50)
    path_50 = data_dir / "countdown_eval_50.json"
    
    with path_50.open("w", encoding="utf-8") as f:
        json.dump(eval_50, f, indent=2)
    print(f"-> Save 50 debug dataset to {path_50}")
    
    print("Generating main evaluation dataset (200 samples)...")
    eval_200 = generate_countdown_dataset(num_samples=200)
    path_200 = data_dir / "countdown_eval_200.json"
    
    with path_200.open("w", encoding="utf-8") as f:
        json.dump(eval_200, f, indent=2)
    print(f"-> Save 200 debug dataset to {path_200}")
    
    print("\nData generation complete. Evaluation set is now frozen.")


if __name__ == "__main__":
    main()
    