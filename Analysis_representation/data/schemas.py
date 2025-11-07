"""Shared data structures used across the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

try:  # Pillow is optional until an image needs to be materialised
    from PIL import Image as PILImage  # type: ignore
except ImportError:  # pragma: no cover - evaluated only when Pillow is missing
    PILImage = None  # type: ignore


@dataclass(frozen=True)
class ImageSample:
    """Represents a single image entry from the dataset."""

    image_id: int
    image_path: Path | None = None
    image_data: "PILImage | None" = None
    filename: str | None = None
    coco_url: Optional[str] = None
    metadata: Mapping[str, object] | None = None

    def ensure_exists(self) -> None:
        """Raise an error if the image file does not exist on disk."""

        if self.image_data is not None:
            return
        if self.image_path is None or not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

    def to_model_input(self):
        """Return a PIL image ready to be consumed by a processor."""

        if self.image_data is not None:
            return self.image_data
        if self.image_path is None:
            raise ValueError("ImageSample does not contain in-memory data or a file path.")
        if PILImage is None:
            raise ImportError("Pillow is required to materialise images from disk.")
        return PILImage.open(self.image_path).convert("RGB")


@dataclass(frozen=True)
class CaptionSample:
    """Represents a caption in a specific language."""

    language: str
    text: str
    source: str | None = None  # e.g. "human", "machine_translation"


@dataclass(frozen=True)
class MultilingualExample:
    """A multilingual bundle of captions associated with an image."""

    image: ImageSample
    captions: Mapping[str, CaptionSample]

    def languages(self) -> Sequence[str]:
        return list(self.captions.keys())

    def caption_for(self, language: str) -> CaptionSample:
        try:
            return self.captions[language]
        except KeyError:
            raise KeyError(f"Language '{language}' not available in example {self.image.image_id}")


@dataclass
class SampleBatch:
    """A batch of multilingual examples for downstream processing."""

    examples: List[MultilingualExample]

    def __iter__(self) -> Iterable[MultilingualExample]:
        return iter(self.examples)

    def filter_by_language_pair(self, lang_a: str, lang_b: str) -> "SampleBatch":
        """Return a new batch containing only examples with both languages present."""

        filtered = [
            example
            for example in self.examples
            if lang_a in example.captions and lang_b in example.captions
        ]
        return SampleBatch(filtered)

    def to_language_dict(self, language: str) -> Dict[int, str]:
        """Return a dictionary mapping image ids to captions for the requested language."""

        result: Dict[int, str] = {}
        for example in self.examples:
            if language in example.captions:
                result[example.image.image_id] = example.captions[language].text
        return result
