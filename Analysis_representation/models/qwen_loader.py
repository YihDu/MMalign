"""Loader for Qwen2.5-VL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


@dataclass
class QwenLoaderConfig:
    model_name: str
    revision: Optional[str] = None
    device: str = "cpu"
    dtype: str = "auto"
    use_safetensors: bool = True


class QwenModelLoader:
    """Encapsulates loading for Qwen2.5-VL models."""

    def __init__(self, config: QwenLoaderConfig) -> None:
        self._config = config
        self._model = None
        self._processor = None

    def load(self) -> Tuple[Any, Any]:
        if self._model is None or self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self._config.model_name,
                revision=self._config.revision,
                use_safetensors=self._config.use_safetensors,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._config.model_name,
                revision=self._config.revision,
                use_safetensors=self._config.use_safetensors,
                torch_dtype=self._config.dtype,
                low_cpu_mem_usage=True,
            )
            self._model.to(self._config.device)
        return self._model, self._processor

    def embedding_context(self) -> Dict[str, Any]:
        """Return hints that guide downstream embedding extraction."""

        return {
            "model_type": "qwen-vl",
            "use_image_placeholder_mask": True,
        }

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Model has not been loaded yet. Call `load()` first.")
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError("Processor has not been loaded yet. Call `load()` first.")
        return self._processor

    def config_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self._config.model_name,
            "revision": self._config.revision,
            "device": self._config.device,
            "dtype": self._config.dtype,
            "use_safetensors": self._config.use_safetensors,
        }
