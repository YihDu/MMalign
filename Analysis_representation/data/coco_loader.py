"""Loader for the parquet-based COCO multilingual dataset described in data_guide.md."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "COCODataset requires the `datasets` package. Install it via `pip install datasets`."
    ) from exc

from PIL import Image as PILImage

from .schemas import CaptionSample, ImageSample, MultilingualExample, SampleBatch


class COCODataset:
    """Utility wrapper around the parquet-based multilingual COCO release."""

    def __init__(
        self,
        data_dir: Path,
        splits: Sequence[str],
        *,
        seed: Optional[int] = None,
        caption_index: int = 0,
        filter_empty_languages: bool = True,
        language_aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not splits:
            raise ValueError("At least one dataset split must be provided.")
        self._data_dir = Path(data_dir)
        self._splits = list(splits)
        self._rng = random.Random(seed)
        self._caption_index = caption_index
        self._filter_empty = filter_empty_languages
        self._language_aliases: Mapping[str, str] = language_aliases or {}

        # 预先加载每个 split 的 parquet 数据，避免重复扫描磁盘
        self._dataset_dict = self._load_splits(self._data_dir, splits)
        self._combined = self._concatenate(self._dataset_dict, splits)

    def build_batch(self, limit: int, languages: Sequence[str]) -> SampleBatch:
        """Assemble a sample batch containing multilingual captions for each image."""

        if limit <= 0:
            raise ValueError("`limit` must be a positive integer.")
        if not languages:
            raise ValueError("`languages` cannot be empty.")

        dataset = (
            self._combined.shuffle(seed=self._rng.randint(0, 1_000_000))
            if limit < len(self._combined)
            else self._combined
        )
        examples: list[MultilingualExample] = []
        for sample in dataset:
            # 针对每个样本检查语言是否齐全，若缺失则跳过
            example = self._build_example(sample, languages)
            if example is None:
                continue
            examples.append(example)
            if len(examples) == limit:
                break

        if len(examples) < limit:
            raise ValueError(
                f"Unable to assemble batch of {limit} examples. "
                "Consider relaxing language requirements or using more splits."
            )
        return SampleBatch(examples)

    def _build_example(
        self,
        sample: Mapping[str, object],
        languages: Sequence[str],
    ) -> Optional[MultilingualExample]:
        captions: MutableMapping[str, CaptionSample] = {}
        for language in languages:
            dataset_key = self._language_aliases.get(language, language)
            entries = sample.get(dataset_key)
            if entries is None:
                if self._filter_empty:
                    return None
                continue
            text = self._select_caption(entries)
            if not text:
                if self._filter_empty:
                    return None
                continue
            captions[language] = CaptionSample(
                language=language,
                text=text,
                source="dataset",
            )

        if len(captions) < len(languages):
            return None

        image = self._resolve_image(sample.get("image"))
        if image is None:
            return None
        # 将 PIL.Image 与必要的 ID/文件名打包成 ImageSample
        image_sample = ImageSample(
            image_id=int(sample["cocoid"]),
            image_data=image,
            filename=sample.get("filename"),  # type: ignore[arg-type]
            metadata={"split": sample.get("split")},
        )
        return MultilingualExample(image=image_sample, captions=captions)

    def _select_caption(self, entries: object) -> Optional[str]:
        # 每个语言字段可能是字符串或字符串列表，统一抽取首个非空文本
        if isinstance(entries, str):
            values: Sequence[object] = [entries]
        elif isinstance(entries, Sequence):
            values = entries
        else:
            return None
        cleaned = [
            candidate.strip()
            for candidate in values
            if isinstance(candidate, str) and candidate.strip()
        ]
        if not cleaned:
            return None
        if 0 <= self._caption_index < len(values):
            preferred = values[self._caption_index]
            if isinstance(preferred, str) and preferred.strip():
                return preferred.strip()
        return cleaned[0]

    @staticmethod
    def _resolve_image(image_entry: object) -> Optional[PILImage.Image]:
        if isinstance(image_entry, PILImage.Image):
            return image_entry.convert("RGB")
        if isinstance(image_entry, Mapping):
            path = image_entry.get("path")
            data = image_entry.get("bytes")
            if data is not None:
                return PILImage.open(io.BytesIO(data)).convert("RGB")
            if path is not None:
                return PILImage.open(path).convert("RGB")
        return None

    @staticmethod
    def _concatenate(dataset_dict: DatasetDict, splits: Sequence[str]) -> Dataset:
        selected = [dataset_dict[split] for split in splits]
        if len(selected) == 1:
            return selected[0]
        return concatenate_datasets(selected)

    @staticmethod
    def _load_splits(data_dir: Path, splits: Sequence[str]) -> DatasetDict:
        data_files = {
            split: str(data_dir / f"{split}-*.parquet") for split in splits
        }
        dataset_dict = load_dataset("parquet", data_files=data_files)
        missing = [split for split in splits if split not in dataset_dict]
        if missing:
            raise ValueError(f"Missing requested splits: {', '.join(missing)}")
        return dataset_dict
