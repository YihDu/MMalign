"""Data loading and preprocessing utilities for multilingual vision-language experiments."""

from .schemas import (
    CaptionSample,
    ImageSample,
    MultilingualExample,
    SampleBatch,
)  

try:
    from .coco_loader import COCODataset as _COCODatasetImpl
except ImportError as exc:  # pragma: no cover - optional dependency
    class COCODataset:  # type: ignore[override]
        """Placeholder that explains the missing optional dependency."""

        def __init__(self, *_, **__):
            raise ImportError(
                "COCODataset requires the optional `datasets` package. "
                "Install it via `pip install datasets`."
            ) from exc
else:
    COCODataset = _COCODatasetImpl
