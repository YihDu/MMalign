from __future__ import annotations
"""Entry point for running multilingual alignment experiments."""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(os.getenv("HF_ENDPOINT"))

from pathlib import Path
from scripts.run_experiment import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

def main() -> None:
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    summary_path = Path("/root/personal/expriments/results/embedding/results_1109_2.json")
    debug_csv_path = Path("/root/personal/expriments/results/debug_metrics_1109_2.csv")
    run_pipeline(config_path, summary_path, debug_csv_path)

if __name__ == "__main__":
    main()
