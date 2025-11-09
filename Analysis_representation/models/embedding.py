"""Lightweight hidden-state extraction for image/text pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
    image_pooling: str = "mean"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "SimpleEmbeddingConfig":
        if not data:
            return cls()
        pooling = str(data.get("image_pooling", cls.image_pooling)).strip().lower()
        if pooling not in {"mean", "first"}:
            raise ValueError("image_pooling must be one of {'mean', 'first'}.")
        return cls(
            image_prompt=str(data.get("image_prompt", cls.image_prompt)),
            text_prefix=str(data.get("text_prefix", cls.text_prefix)),
            text_suffix=str(data.get("text_suffix", cls.text_suffix)),
            image_pooling=pooling,
        )

    def build_image_prompt(self) -> str:
        prompt = self.image_prompt.replace("{image_token}", "<image>").strip()
        if "<image>" not in prompt:
            raise ValueError("image_prompt must include '<image>' token.")
        return prompt

    def build_text(self, caption: str) -> str:
        return f"{self.text_prefix}{caption.strip()}{self.text_suffix}".strip()

    def build_joint_prompt(self, caption: str) -> str:
        """Compose the final prompt that pairs <image> with the caption text."""
        prompt = self.build_image_prompt()
        text = self.build_text(caption)
        if text:
            return f"{prompt}\n{text}".strip()
        return prompt


def encode_examples(
    examples: Iterable[MultilingualExample],
    model,
    processor,
    fusion_config: Mapping[str, object] | None = None,
    micro_batch_size: int | None = None,
) -> EmbeddingBatch:
    config = SimpleEmbeddingConfig.from_mapping(fusion_config)

    images: List[Image.Image] = []
    texts_by_language: Dict[str, List[str]] = {}
    for example in examples:
        images.append(_ensure_pil(example.image.to_model_input()))
        for language, caption in example.captions.items():
            texts_by_language.setdefault(language, []).append(config.build_joint_prompt(caption.text))

    if not images:
        raise ValueError("No examples provided for encoding.")

    language_model = _resolve_language_model(model)
    tokenizer = _resolve_tokenizer(processor)
    device, dtype = _module_device_dtype(language_model, fallback_device=_model_device(model))
    image_token_id = _resolve_image_token_id(tokenizer)
    pad_token_id = _resolve_pad_token_id(tokenizer, model)
    num_image_patches = _infer_num_image_patches(model)
    image_pooling = config.image_pooling

    # get text embedding
    embeddings: Dict[str, LanguageEmbedding] = {}
    batch_targets = list(range(len(images)))
    chunk_size = micro_batch_size or len(images)
    with torch.inference_mode():
        for language, texts in texts_by_language.items():
            if len(texts) != len(images):
                raise ValueError(
                    "Every image must provide a caption for all languages; "
                    f"language '{language}' is missing {len(images) - len(texts)} captions."
                )
            text_layers_accum: List[List[torch.Tensor]] = []
            image_layers_accum: List[List[torch.Tensor]] = []
            for start in range(0, len(batch_targets), chunk_size):
                end = start + chunk_size
                idx_range = batch_targets[start:end]
                chunk_images = [images[idx] for idx in idx_range]
                chunk_texts = [texts[idx] for idx in idx_range]
                processor_inputs = processor(  # type: ignore[call-arg]
                    text=list(chunk_texts),
                    images=chunk_images,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                processor_inputs = _ensure_attention_mask(processor_inputs)
                processor_inputs = _move_tensors(processor_inputs, device=device, dtype=dtype)
                input_ids = processor_inputs.get("input_ids")
                if not isinstance(input_ids, torch.Tensor):
                    raise ValueError("Processor outputs must include input_ids.")
                attention_mask = processor_inputs.get("attention_mask")
                if not isinstance(attention_mask, torch.Tensor):
                    raise ValueError("Processor outputs must include attention_mask.")
                expanded_positions = _expanded_token_positions(
                    input_ids,
                    image_token_id=image_token_id,
                    num_image_patches=num_image_patches,
                    pad_token_id=pad_token_id,
                )
                last_token_positions = _last_token_positions(attention_mask, expanded_positions)
                image_token_starts = _image_token_starts(
                    input_ids,
                    expanded_positions,
                    image_token_id=image_token_id,
                    num_image_patches=num_image_patches,
                )
                outputs = model(  # type: ignore[call-arg]
                    **processor_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = _extract_hidden_states(outputs)
                image_layers = _image_token_layers(
                    hidden_states,
                    image_token_starts,
                    span_length=num_image_patches,
                    pooling=image_pooling,
                )
                text_layers = _gather_token_layers(hidden_states, last_token_positions)
                if not text_layers or not image_layers:
                    continue
                text_layers_accum.append(text_layers)
                image_layers_accum.append(image_layers)

            if not text_layers_accum or not image_layers_accum:
                raise ValueError(f"No embeddings collected for language '{language}'.")
            concatenated_text = _concat_layer_chunks(text_layers_accum)
            concatenated_image = _concat_layer_chunks(image_layers_accum)
            text_embedding = _to_multilayer_embedding(concatenated_text, domain=f"text:{language}")
            image_embedding = _to_multilayer_embedding(concatenated_image, domain=f"image:{language}")
            embeddings[language] = LanguageEmbedding(text=text_embedding, image=image_embedding)

    return EmbeddingBatch(captions=embeddings)


def _gather_token_layers(hidden_states, indices: torch.Tensor) -> List[torch.Tensor]:
    batch = torch.arange(indices.size(0), device=indices.device)
    usable = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
    gathered: List[torch.Tensor] = []
    for layer in usable:
        gathered.append(layer[batch, indices, :])
    return gathered


def _concat_layer_chunks(chunks: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    layers = len(chunks[0])
    concatenated: List[torch.Tensor] = []
    for layer_idx in range(layers):
        concatenated.append(torch.cat([chunk[layer_idx] for chunk in chunks], dim=0))
    return concatenated


def _image_token_layers(
    hidden_states,
    start_indices: torch.Tensor,
    *,
    span_length: int,
    pooling: str,
) -> List[torch.Tensor]:
    if not hidden_states:
        return []
    if span_length <= 0:
        raise ValueError("span_length for image tokens must be positive.")
    usable = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
    gathered: List[torch.Tensor] = []
    for layer in usable:
        vectors: List[torch.Tensor] = []
        for batch_idx in range(start_indices.size(0)):
            start = int(start_indices[batch_idx].item())
            end = start + span_length
            token_span = layer[batch_idx, start:end, :]
            if token_span.shape[0] != span_length:
                raise ValueError(
                    f"Image token span (batch={batch_idx}) has unexpected length {token_span.shape[0]} "
                    f"(expected {span_length})."
                )
            vectors.append(_pool_image_tokens(token_span, pooling))
        gathered.append(torch.stack(vectors, dim=0))
    return gathered


def _pool_image_tokens(span: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return span.mean(dim=0)
    if pooling == "first":
        return span[0]
    raise ValueError(f"Unsupported image pooling strategy '{pooling}'.")


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


def _resolve_image_token_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<image>")
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    additional_tokens = getattr(tokenizer, "additional_special_tokens", []) or []
    additional_ids = getattr(tokenizer, "additional_special_tokens_ids", []) or []
    for token, special_id in zip(additional_tokens, additional_ids):
        if token == "<image>" and isinstance(special_id, int):
            return special_id
    raise ValueError("Unable to resolve a token id for '<image>'.")


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


def _extract_hidden_states(outputs: Any):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states
    language_outputs = getattr(outputs, "language_model_outputs", None)
    if language_outputs is not None:
        lm_hidden = getattr(language_outputs, "hidden_states", None)
        if lm_hidden is not None:
            return lm_hidden
    raise ValueError(
        "Model outputs do not contain hidden_states. Ensure output_hidden_states=True is passed to the model call."
    )


def _expanded_token_positions(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    num_image_patches: int,
    pad_token_id: int | None,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be 2D [batch, seq_len].")
    if num_image_patches <= 0:
        raise ValueError("num_image_patches must be positive.")
    special_mask = input_ids == image_token_id
    increments = special_mask.to(dtype=torch.long) * (num_image_patches - 1) + 1
    positions = torch.cumsum(increments, dim=-1) - 1

    num_special = special_mask.sum(dim=-1)
    max_special = int(num_special.max().item())
    max_embed_dim = max_special * (num_image_patches - 1) + input_ids.size(1)
    nb_image_pad = max_embed_dim - 1 - positions[:, -1]

    left_padding = False
    if pad_token_id is not None:
        pad_matches = input_ids[:, -1] == pad_token_id
        left_padding = torch.sum(pad_matches).item() == 0

    if left_padding:
        positions = positions + nb_image_pad.unsqueeze(-1)

    return positions


def _last_token_positions(attention_mask: torch.Tensor, expanded_positions: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D [batch, seq_len].")
    mask = attention_mask.to(dtype=torch.long)
    lengths = torch.clamp(mask.sum(dim=1) - 1, min=0)
    batch = torch.arange(lengths.size(0), device=expanded_positions.device)
    return expanded_positions[batch, lengths]


def _image_token_starts(
    input_ids: torch.Tensor,
    expanded_positions: torch.Tensor,
    *,
    image_token_id: int,
    num_image_patches: int,
) -> torch.Tensor:
    if num_image_patches <= 0:
        raise ValueError("num_image_patches must be positive.")
    batch_size = input_ids.size(0)
    starts = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
    for batch_idx in range(batch_size):
        matches = torch.nonzero(input_ids[batch_idx] == image_token_id, as_tuple=False).flatten()
        if matches.numel() == 0:
            continue
        first_idx = matches[0]
        end_pos = expanded_positions[batch_idx, first_idx]
        starts[batch_idx] = end_pos - (num_image_patches - 1)
    if torch.any(starts < 0):
        missing = torch.nonzero(starts < 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"<image> token missing for samples at indices {missing}.")
    return starts


def _resolve_pad_token_id(tokenizer, model) -> int | None:
    candidates: Sequence[int | None] = (
        getattr(tokenizer, "pad_token_id", None),
        getattr(getattr(tokenizer, "tokenizer", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and candidate >= 0:
            return candidate
    return None


def _infer_num_image_patches(model) -> int:
    vision_config = getattr(model, "config", None)
    if vision_config is not None:
        vision_config = getattr(vision_config, "vision_config", None)
    if vision_config is not None:
        count = _patch_count(
            getattr(vision_config, "image_size", None),
            getattr(vision_config, "patch_size", None),
        )
        if count is not None:
            return count

    vision_tower = getattr(model, "vision_tower", None)
    candidate = getattr(getattr(vision_tower, "config", None), "num_image_tokens", None)
    if isinstance(candidate, int) and candidate > 0:
        return candidate

    raise ValueError("Unable to infer the number of image tokens from the model configuration.")


def _patch_count(image_size: Any, patch_size: Any) -> int | None:
    def _size_tuple(value: Any) -> tuple[int, int] | None:
        if isinstance(value, int):
            return value, value
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        return None

    size = _size_tuple(image_size)
    patches = _size_tuple(patch_size)
    if size is None or patches is None:
        return None
    if patches[0] <= 0 or patches[1] <= 0:
        return None
    return (size[0] // patches[0]) * (size[1] // patches[1])
