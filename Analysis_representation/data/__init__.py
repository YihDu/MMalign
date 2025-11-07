"""Data loading and preprocessing utilities for multilingual vision-language experiments."""

from .schemas import (
    CaptionSample,
    ImageSample,
    MultilingualExample,
    SampleBatch,
)  # noqa: F401

try:
    from .coco_loader import COCODataset  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    COCODataset = None  # type: ignore
