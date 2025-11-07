"""Entry point for running multilingual alignment experiments."""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(os.getenv("HF_ENDPOINT"))
from __future__ import annotations

from pathlib import Path

from scripts.run_experiment import run_pipeline


def main() -> None:
    config_path = Path("config/settings.yaml")
    summary_path = Path("results/results_similarity1.json")
    run_pipeline(config_path, summary_path)

if __name__ == "__main__":
    main()
