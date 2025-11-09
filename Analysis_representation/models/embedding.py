"""Lightweight hidden-state extraction for image/text pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.schemas import MultilingualExample


@dataclass
class MultilayerEmbedding:
    per_layer: List[np.ndarray]
    pooled: np.ndarray | None = None


@dataclass
class LanguageEmbedding:
    text: MultilayerEmbedding
    image: MultilayerEmbedding


@dataclass
class EmbeddingBatch:
    captions: Dict[str, LanguageEmbedding]


@dataclass
class SimpleEmbeddingConfig:
    image_prompt: str = "{image_token}"
    text_prefix: str = ""
    text_suffix: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "SimpleEmbeddingConfig":
        if not data:
            return cls()
        return cls(
            image_prompt=str(data.get("image_prompt", cls.image_prompt)),
            text_prefix=str(data.get("text_prefix", cls.text_prefix)),
            text_suffix=str(data.get("text_suffix", cls.text_suffix)),
        )

    def build_image_prompt(self) -> str:
        prompt = self.image_prompt.replace("{image_token}", "<image>").strip()
        if "<image>" not in prompt:
            raise ValueError("image_prompt must include '<image>' token.")
        return prompt

    def build_text(self, caption: str) -> str:
        return f"{self.text_prefix}{caption.strip()}{self.text_suffix}".strip()


def encode_examples(
    examples: Iterable[MultilingualExample],
    model,
    processor,
    fusion_config: Mapping[str, object] | None = None,
) -> EmbeddingBatch:
    config = SimpleEmbeddingConfig.from_mapping(fusion_config)
    image_prompt = config.build_image_prompt()

    images: List[Image.Image] = []
    texts_by_language: Dict[str, List[str]] = {}
    for example in examples:
        images.append(_ensure_pil(example.image.to_model_input()))
        for language, caption in example.captions.items():
            texts_by_language.setdefault(language, []).append(config.build_text(caption.text))

    if not images:
        raise ValueError("No examples provided for encoding.")

    language_model = _resolve_language_model(model)
    tokenizer = _resolve_tokenizer(processor)
    device, dtype = _module_device_dtype(language_model, fallback_device=_model_device(model))


    # get image embedding
    image_embedding = _encode_image_hidden_states(
        images=images,
        prompt=image_prompt,
        processor=processor,
        language_model=language_model,
        device=device,
        dtype=dtype,
    )

    # get text embedding
    embeddings: Dict[str, LanguageEmbedding] = {}
    with torch.inference_mode():
        for language, texts in texts_by_language.items():
            if len(texts) != len(images):
                raise ValueError(
                    "Every image must provide a caption for all languages; "
                    f"language '{language}' is missing {len(images) - len(texts)} captions."
                )
            tokenized = tokenizer(  # type: ignore[call-arg]
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = _ensure_attention_mask(tokenized)
            tokenized = _move_tensors(tokenized, device=device)
            outputs = language_model(  # type: ignore[call-arg]
                **tokenized,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            text_layers = _last_token_layers(outputs.hidden_states, tokenized["attention_mask"])
            text_embedding = _to_multilayer_embedding(text_layers, domain=f"text:{language}")
            embeddings[language] = LanguageEmbedding(text=text_embedding, image=image_embedding)

    return EmbeddingBatch(captions=embeddings)


def _encode_image_hidden_states(
    *,
    images: List[Image.Image],
    prompt: str,
    processor,
    language_model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> MultilayerEmbedding:
    processor_inputs = processor(  # type: ignore[call-arg]
        text=[prompt] * len(images), # 为每张图像生成一个相同的文本提示（prompt），这个提示通常用于告诉模型“这是一张图像”，如<image>。
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    processor_inputs = _move_tensors(processor_inputs, device=device, dtype=dtype)
    attention_mask = processor_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("Processor outputs must include attention_mask.")
    with torch.inference_mode():
        outputs = language_model(  # type: ignore[call-arg]
            **processor_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    image_layers = _last_token_layers(outputs.hidden_states, attention_mask)
    return _to_multilayer_embedding(image_layers, domain="image")


def _last_token_layers(hidden_states, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    if not hidden_states:
        return []
    indices = _last_token_indices(attention_mask)
    batch = torch.arange(indices.size(0), device=indices.device)
    usable = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
    gathered: List[torch.Tensor] = []
    for layer in usable:
        gathered.append(layer[batch, indices, :])
    return gathered


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D [batch, seq_len].")
    mask = attention_mask.to(dtype=torch.long)
    lengths = mask.sum(dim=1) - 1
    return torch.clamp(lengths, min=0)


def _to_multilayer_embedding(layers: List[torch.Tensor], *, domain: str) -> MultilayerEmbedding:
    per_layer: List[np.ndarray] = []
    for tensor in layers:
        array = tensor.detach().to(dtype=torch.float32).cpu().numpy()
        per_layer.append(array)
    pooled = per_layer[-1] if per_layer else None
    return MultilayerEmbedding(per_layer=per_layer, pooled=pooled)


def _ensure_attention_mask(model_inputs: Mapping[str, object]) -> Mapping[str, object]:
    if "attention_mask" in model_inputs:
        return model_inputs
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("Tokenizer outputs must include input_ids tensor when attention_mask is missing.")
    updated = dict(model_inputs)
    updated["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
    return updated


def _move_tensors(
    data: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Mapping[str, object]:
    moved: Dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, non_blocking=True)
            if dtype is not None and torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=dtype)
            moved[key] = tensor
        else:
            moved[key] = value
    return moved


def _resolve_tokenizer(processor):
    for attr in ("tokenizer", "text_tokenizer"):
        tokenizer = getattr(processor, attr, None)
        if tokenizer is not None:
            return tokenizer
    text_processor = getattr(processor, "text_processor", None)
    if text_processor is not None:
        for attr in ("tokenizer", "processor"):
            tokenizer = getattr(text_processor, attr, None)
            if tokenizer is not None:
                return tokenizer
    raise AttributeError("Processor does not expose a tokenizer-compatible attribute.")


def _resolve_language_model(model) -> nn.Module:
    for attr in ("language_model", "model"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            return module
    if isinstance(model, nn.Module):
        return model
    raise TypeError("Unable to resolve language model from the provided model instance.")


def _module_device_dtype(
    module: nn.Module,
    *,
    fallback_device: torch.device | None = None,
) -> tuple[torch.device, torch.dtype]:
    try:
        param = next(module.parameters())
        return param.device, param.dtype
    except StopIteration:
        device = fallback_device if fallback_device is not None else torch.device("cpu")
        return device, torch.float32


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        device = getattr(model, "device")
        if isinstance(device, torch.device):
            return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _ensure_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.array(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        max_val = float(array.max()) if array.size else 1.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")
