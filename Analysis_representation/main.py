from __future__ import annotations
"""Entry point for running multilingual alignment experiments."""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(os.getenv("HF_ENDPOINT"))

from pathlib import Path
from scripts.run_experiment import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

def main() -> None:
    # Resolve config/output relative to the repository root so `python main.py`
    # works regardless of the current working directory.
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    summary_path = PROJECT_ROOT / "results/embedding/" / "results_similarity1.json"
    run_pipeline(config_path, summary_path)

if __name__ == "__main__":
    main()
