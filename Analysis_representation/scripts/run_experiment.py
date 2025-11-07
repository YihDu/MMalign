"""Run alignment experiments defined in the configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from data import COCODataset
from experiment import build_conditions, run_experiment
from models.llava_loader import LLaVAModelLoader, LLaVALoaderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--output", type=Path, default=Path("report/distance_map.json"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_pipeline(config_path: Path, summary_path: Path) -> None:
    config = load_config(config_path)
    languages: Sequence[str] = config["experiment"]["languages"]

    # 加载多语言 COCO 数据，按照配置组合指定 split
    dataset_cfg = config["data"]["coco"]
    dataset = COCODataset(
        data_dir=Path(dataset_cfg["data_dir"]),
        splits=dataset_cfg.get("splits", ["train"]),
        seed=config["experiment"].get("seed"),
        caption_index=dataset_cfg.get("caption_index", 0),
        filter_empty_languages=dataset_cfg.get("filter_empty_languages", True),
        language_aliases=dataset_cfg.get("language_aliases"),
    )

    # 初始化 LLaVA 模型与对应 Processor
    loader = LLaVAModelLoader(
        LLaVALoaderConfig(
            model_name=config["model"]["name"],
            revision=config["model"].get("revision"),
            device=config["model"].get("device", "cpu"),
            dtype=config["model"].get("dtype", "auto"),
        )
    )
    model, processor = loader.load()

    # 抽样构建包含所有目标语言的批次
    batch = dataset.build_batch(
        limit=config["experiment"].get("sample_size", 32),
        languages=languages,
    )

    # 执行三种条件，得到完整的 distance_map
    conditions = build_conditions()
    distance_map = run_experiment(
        model=model,
        processor=processor,
        batch=batch,
        conditions=conditions,
        languages=languages,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(distance_map, handle, ensure_ascii=False, indent=2)

    print(f"distance_map written to {summary_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.output)


if __name__ == "__main__":
    main()
