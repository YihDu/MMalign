"""Loader for pretrained LLaVA models via Hugging Face."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from transformers import AutoProcessor, LlavaForConditionalGeneration


@dataclass
class LLaVALoaderConfig:
    model_name: str
    revision: Optional[str] = None
    device: str = "cpu"
    dtype: str = "auto"
    use_safetensors: bool = True

class LLaVAModelLoader:
    """Encapsulates loading of weights and processors for LLaVA models."""

    def __init__(self, config: LLaVALoaderConfig) -> None:
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
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self._config.model_name,
                revision=self._config.revision,
                use_safetensors=self._config.use_safetensors,
                torch_dtype=self._config.dtype,
                low_cpu_mem_usage=True,
            )
            self._model.to(self._config.device)
        return self._model, self._processor

    def embedding_context(self) -> Dict[str, Any]:
        """Hints for downstream embedding extraction."""

        return {
            "model_type": "llava",
            "use_image_placeholder_mask": False,
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
