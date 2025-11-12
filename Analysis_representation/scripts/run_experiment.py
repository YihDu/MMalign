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
from models.qwen_loader import QwenModelLoader, QwenLoaderConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "distance_map.json"
DEFAULT_DEBUG_CSV = PROJECT_ROOT / "results" / "debug_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug-csv", type=Path, default=DEFAULT_DEBUG_CSV)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def run_pipeline(config_path: Path, summary_path: Path, debug_csv_path: Path | None = None) -> None:
    config = load_config(config_path)
    languages: Sequence[str] = config["experiment"]["languages"]

    print("加载COCO数据集")
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
    
    print("加载COCO数据集完毕")
    print("------------------------")
    

    print("加载 model")
    model_cfg = config["model"]
    model_type = str(model_cfg.get("type", "llava")).lower()
    if model_type == "qwen-vl":
        loader = QwenModelLoader(
            QwenLoaderConfig(
                model_name=model_cfg["name"],
                revision=model_cfg.get("revision"),
                device=model_cfg.get("device", "cpu"),
                dtype=model_cfg.get("dtype", "auto"),
            )
        )
    else:
        loader = LLaVAModelLoader(
            LLaVALoaderConfig(
                model_name=model_cfg["name"],
                revision=model_cfg.get("revision"),
                device=model_cfg.get("device", "cpu"),
                dtype=model_cfg.get("dtype", "auto"),
            )
        )
    model, processor = loader.load()
    embedding_context = loader.embedding_context()


    print("加载 model 完毕")
    print("------------------------")

    print("构建 batch")
    # 抽样构建包含所有目标语言的批次
    batch = dataset.build_batch(
        limit=config["experiment"].get("sample_size", 32),
        languages=languages,
    )
    micro_batch_size = int(config["experiment"].get("batch_size", 8))
    print("构建 batch 完毕")
    print("------------------------")

    print(f"[debug][batch] size={len(batch.examples)} languages={list(languages)}")
    if batch.examples:
        sample = batch.examples[0]
        filename = sample.image.filename or "N/A"
        print(
            f"[debug][batch] sample image_id={sample.image.image_id} "
            f"filename={filename} metadata={sample.image.metadata}"
        )
        for language in languages:
            caption = sample.captions.get(language)
            if caption is None:
                print(f"  - {language}: <missing>")
                continue
            print(
                f"  - {language}: len={len(caption.text)} "
                f"text=\"{_preview_text(caption.text)}\""
            )
    
    
    # 内部就是一个 List[MultilingualExample]，每个 MultilingualExample 包含：
    # image: ImageSample（含 image_id、image_data 或 image_path、filename、metadata）
    # captions: Dict[str, CaptionSample]，键为语言代码，值里有 language、text、source
    # 因此 build_batch 返回的对象可迭代 for example in batch，或用 batch.examples 直接访问所有 MultilingualExample，方便后续传给 experiment.runner.run_experiment()。

    # 执行三种条件，得到完整的 distance_map
    condition_overrides = config.get("experiment", {}).get("conditions")
    conditions = build_conditions(condition_overrides)
    distance_map = run_experiment(
        model=model,
        processor=processor,
        batch=batch,
        conditions=conditions,
        languages=languages,
        analysis_config=config.get("analysis", {}),
        embedding_context=embedding_context,
        debug_csv_path=debug_csv_path,
        micro_batch_size=micro_batch_size,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(distance_map, handle, ensure_ascii=False, indent=2)

    print(f"distance_map written to {summary_path}")




def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.output, args.debug_csv)


if __name__ == "__main__":
    main()
