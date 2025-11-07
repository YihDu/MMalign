"""Entry point for running multilingual alignment experiments."""

from __future__ import annotations

from pathlib import Path

from scripts.run_experiment import run_pipeline


def main() -> None:
    config_path = Path("config/settings.yaml")
    summary_path = Path("report/summary.md")
    run_pipeline(config_path, summary_path)


if __name__ == "__main__":
    main()
