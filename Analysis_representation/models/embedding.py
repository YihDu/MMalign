"""Post-fusion hidden-state extraction utilities for LLaVA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.schemas import MultilingualExample


@dataclass
class MultilayerEmbedding:
    """Container for per-layer hidden states plus an optional pooled embedding."""

    per_layer: List[np.ndarray]
    pooled: np.ndarray | None = None


@dataclass
class SequenceSpanInfo:
    """Bookkeeping for fused sequences (per-sample spans and indices)."""

    fused_lengths: List[int]
    image_spans: List[Tuple[int, int]]  # [start, end) indices
    text_last_indices: List[int]
    num_image_tokens: int


@dataclass
class LanguageFusionEmbedding:
    """Per-language representations derived from the fused LLM forward pass."""

    text: MultilayerEmbedding  # h(text | image)
    image: MultilayerEmbedding  # pooled image-span hidden states
    text_only: MultilayerEmbedding  # h(text | ∅)
    delta_text: MultilayerEmbedding  # Δh = text - text_only per layer
    spans: SequenceSpanInfo


@dataclass
class EmbeddingBatch:
    """Grouped fused embeddings keyed by language code."""

    captions: Dict[str, LanguageFusionEmbedding]

    @property
    def languages(self) -> Dict[str, LanguageFusionEmbedding]:
        return self.captions


@dataclass
class FusionConfig:
    """Runtime knobs controlling prompt templates and pooling behaviors."""

    prompt_with_image: str = "{image_token}\n{caption}"
    prompt_text_only: str = "{caption}"
    image_pooling: str = "mean"
    keep_image_tokens: bool = False
    delta_enabled: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "FusionConfig":
        if not data:
            return cls()
        prompt_with_image = str(data.get("prompt_with_image", cls.prompt_with_image))
        prompt_text_only = str(data.get("prompt_text_only", cls.prompt_text_only))
        image_pooling = str(data.get("image_pooling", cls.image_pooling))
        keep_image_tokens = bool(data.get("keep_image_tokens", cls.keep_image_tokens))
        delta_enabled = bool(data.get("delta_enabled", cls.delta_enabled))
        return cls(
            prompt_with_image=prompt_with_image,
            prompt_text_only=prompt_text_only,
            image_pooling=image_pooling,
            keep_image_tokens=keep_image_tokens,
            delta_enabled=delta_enabled,
        )


class PromptBuilder:
    """Utility to standardise prompt construction for each caption."""

    def __init__(self, with_image: str, text_only: str) -> None:
        self._with_image = with_image
        self._text_only = text_only

    def build_with_image(self, caption: str) -> str:
        prompt = self._with_image.format(
            caption=caption.strip(),
            image_token="<image>",
        )
        if prompt.count("<image>") != 1:
            raise ValueError("Prompt with image must contain exactly one '<image>' token.")
        return prompt

    def build_text_only(self, caption: str) -> str:
        return self._text_only.format(caption=caption.strip()).strip()


def encode_examples(
    examples: Iterable[MultilingualExample],
    model,
    processor,
    fusion_config: Mapping[str, object] | None = None,
) -> EmbeddingBatch:
    """Extract fused hidden states for each language with optional Δh vectors."""

    config = FusionConfig.from_mapping(fusion_config)
    prompt_builder = PromptBuilder(
        with_image=config.prompt_with_image,
        text_only=config.prompt_text_only,
    )

    texts_by_language: dict[str, list[str]] = {}
    image_inputs: List[Image.Image] = []
    for example in examples:
        image_inputs.append(_ensure_pil(example.image.to_model_input()))
        for language, caption in example.captions.items():
            texts_by_language.setdefault(language, []).append(caption.text)

    if not image_inputs:
        raise ValueError("No examples provided for encoding.")

    tokenizer = _resolve_tokenizer(processor)
    language_model = _resolve_language_model(model)
    fallback_device = _model_device(model)
    language_device, language_dtype = _module_device_dtype(
        language_model,
        fallback_device=fallback_device,
    )

    projected_image_tokens = _project_image_tokens(
        images=image_inputs,
        model=model,
        processor=processor,
    )
    projected_image_tokens = projected_image_tokens.to(
        device=language_device,
        dtype=language_dtype,
        non_blocking=True,
    )

    embeddings_by_language: Dict[str, LanguageFusionEmbedding] = {}
    for language, texts in texts_by_language.items():
        if len(texts) != len(image_inputs):
            raise ValueError(
                "Post-fusion analysis requires every sample to provide all languages; "
                f"language '{language}' is missing {len(image_inputs) - len(texts)} captions."
            )
        language_embedding = _encode_language_fusion(
            language=language,
            captions=texts,
            tokenizer=tokenizer,
            language_model=language_model,
            image_tokens=projected_image_tokens,
            prompt_builder=prompt_builder,
            config=config,
        )
        embeddings_by_language[language] = language_embedding

    return EmbeddingBatch(captions=embeddings_by_language)


def _encode_language_fusion(
    *,
    language: str,
    captions: Sequence[str],
    tokenizer,
    language_model,
    image_tokens: torch.Tensor,
    prompt_builder: PromptBuilder,
    config: FusionConfig,
) -> LanguageFusionEmbedding:
    label = language or "text"
    print(
        f"[debug][fusion:{label}] samples={len(captions)} "
        f"prompt_with_image='{_truncate_text_preview(prompt_builder.build_with_image(captions[0]))}'"
    )

    prompts_with_image = [prompt_builder.build_with_image(text) for text in captions]
    prompts_text_only = [prompt_builder.build_text_only(text) for text in captions]

    fused_inputs, fused_mask, span_info = _build_fused_inputs(
        prompts=prompts_with_image,
        tokenizer=tokenizer,
        language_model=language_model,
        image_tokens=image_tokens,
    )

    with torch.no_grad():
        fused_outputs = language_model(  # type: ignore[call-arg]
            inputs_embeds=fused_inputs,
            attention_mask=fused_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    fused_hidden_layers = _select_transformer_layers(fused_outputs.hidden_states)
    text_indices_tensor = torch.tensor(
        span_info.text_last_indices,
        device=fused_inputs.device,
        dtype=torch.long,
    )
    text_layers = _gather_text_vectors(fused_hidden_layers, text_indices_tensor)
    image_layers = _gather_image_vectors(
        fused_hidden_layers,
        span_info.image_spans,
        span_info.num_image_tokens,
        pooling=config.image_pooling,
    )

    text_only_layers: List[torch.Tensor] = []
    delta_layers: List[torch.Tensor] = []
    if config.delta_enabled:
        text_only_inputs = _tokenize_texts(tokenizer, prompts_text_only)
        text_only_inputs = _ensure_attention_mask(text_only_inputs)
        text_only_inputs = _move_tensors(text_only_inputs, device=fused_inputs.device)

        with torch.no_grad():
            text_only_outputs = language_model(  # type: ignore[call-arg]
                **text_only_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        text_only_hidden = _select_transformer_layers(text_only_outputs.hidden_states)
        text_only_indices = _last_token_indices(text_only_inputs["attention_mask"])
        text_only_layers = _gather_text_vectors(text_only_hidden, text_only_indices)
        delta_layers = [a - b for a, b in zip(text_layers, text_only_layers, strict=False)]
    else:
        text_only_layers = []
        delta_layers = []

    text_embedding = _to_multilayer_embedding(text_layers, domain=f"text|image:{label}")
    image_embedding = _to_multilayer_embedding(image_layers, domain=f"image:{label}")
    text_only_embedding = _to_multilayer_embedding(text_only_layers, domain=f"text_only:{label}")
    delta_embedding = _to_multilayer_embedding(delta_layers, domain=f"delta:{label}")

    return LanguageFusionEmbedding(
        text=text_embedding,
        image=image_embedding,
        text_only=text_only_embedding,
        delta_text=delta_embedding,
        spans=span_info,
    )


def _project_image_tokens(
    images: Sequence[Image.Image],
    model,
    processor,
) -> torch.Tensor:
    fallback_device = _model_device(model)
    vision_tower = _resolve_vision_tower(model)
    projector = _resolve_mm_projector(model)
    if projector is None:
        raise AttributeError("Model does not expose a multimodal projector.")

    pixel_values = _prepare_pixel_values(images, processor)
    device, dtype = _module_device_dtype(vision_tower, fallback_device=fallback_device)
    pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)

    with torch.no_grad():
        outputs = vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )

    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None:
        raise ValueError("Vision tower did not return last_hidden_state.")
    patch_tokens = last_hidden[:, 1:, :]  # drop CLS token
    projected = projector(patch_tokens)
    print(
        f"[debug][image] batch={projected.shape[0]} patches={projected.shape[1]} hidden={projected.shape[2]}"
    )
    return projected.detach()


def _build_fused_inputs(
    *,
    prompts: Sequence[str],
    tokenizer,
    language_model,
    image_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, SequenceSpanInfo]:
    tokenized = _tokenize_texts(tokenizer, prompts)
    tokenized = _ensure_attention_mask(tokenized)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("Tokenizer must return torch.Tensor inputs for fused prompts.")

    embed_layer = language_model.get_input_embeddings()
    token_embeds = embed_layer(input_ids.to(device=image_tokens.device))
    attention_mask = attention_mask.to(device=image_tokens.device)

    image_token_id = _resolve_image_token_id(tokenizer)
    batch_size, seq_len = input_ids.shape
    num_image_tokens = int(image_tokens.size(1))
    fused_embeddings: List[torch.Tensor] = []
    fused_masks: List[torch.Tensor] = []
    fused_lengths: List[int] = []
    image_spans: List[Tuple[int, int]] = []
    text_last_indices: List[int] = []

    for batch_idx in range(batch_size):
        text_len = int(attention_mask[batch_idx].sum().item())
        if text_len == 0:
            raise ValueError("Prompt tokenization produced empty sequence.")
        valid_ids = input_ids[batch_idx, :text_len]
        valid_embeds = token_embeds[batch_idx, :text_len, :]
        placeholder_positions = (valid_ids == image_token_id).nonzero(as_tuple=False)
        if placeholder_positions.numel() != 1:
            raise ValueError(
                "Each prompt must contain exactly one '<image>' token that stays within the attention span."
            )
        placeholder_idx = int(placeholder_positions.item())
        prefix = valid_embeds[:placeholder_idx]
        suffix = valid_embeds[placeholder_idx + 1 :]

        fused = torch.cat(
            [prefix, image_tokens[batch_idx].to(valid_embeds.dtype), suffix],
            dim=0,
        )
        fused_len = fused.size(0)
        expected_len = text_len - 1 + num_image_tokens
        if fused_len != expected_len:
            raise ValueError(
                f"Sequence length mismatch after fusion: expected {expected_len}, got {fused_len}."
            )

        fused_embeddings.append(fused)
        mask = torch.zeros(fused_len, device=fused.device)
        mask[:fused_len] = 1
        fused_masks.append(mask)
        fused_lengths.append(int(fused_len))

        image_start = int(prefix.size(0))
        image_end = image_start + num_image_tokens
        image_spans.append((image_start, image_end))
        if suffix.size(0) > 0:
            last_text_idx = image_end + suffix.size(0) - 1
        elif prefix.size(0) > 0:
            last_text_idx = prefix.size(0) - 1
        else:
            raise ValueError("Prompt must include text tokens besides '<image>'.")
        text_last_indices.append(int(last_text_idx))

    max_fused_len = max(fused_lengths)
    hidden_size = fused_embeddings[0].size(-1)
    fused_batch = torch.zeros(
        (batch_size, max_fused_len, hidden_size),
        device=image_tokens.device,
        dtype=fused_embeddings[0].dtype,
    )
    fused_mask_batch = torch.zeros(
        (batch_size, max_fused_len),
        device=image_tokens.device,
        dtype=attention_mask.dtype,
    )
    for idx, fused in enumerate(fused_embeddings):
        length = fused.size(0)
        fused_batch[idx, :length, :] = fused
        fused_mask_batch[idx, :length] = fused_masks[idx]

    span_info = SequenceSpanInfo(
        fused_lengths=fused_lengths,
        image_spans=image_spans,
        text_last_indices=text_last_indices,
        num_image_tokens=num_image_tokens,
    )
    _validate_spans(span_info)
    return fused_batch, fused_mask_batch, span_info


def _gather_text_vectors(
    hidden_layers: Sequence[torch.Tensor],
    positions: torch.Tensor,
) -> List[torch.Tensor]:
    if not hidden_layers:
        return []
    if positions.ndim != 1:
        positions = positions.view(-1)
    batch_size = positions.size(0)
    batch_indices = torch.arange(batch_size, device=positions.device)
    gathered: List[torch.Tensor] = []
    for layer in hidden_layers:
        vectors = layer[batch_indices, positions, :]
        gathered.append(vectors)
    return gathered


def _gather_image_vectors(
    hidden_layers: Sequence[torch.Tensor],
    spans: Sequence[Tuple[int, int]],
    num_image_tokens: int,
    pooling: str,
) -> List[torch.Tensor]:
    if not hidden_layers:
        return []
    pooled_layers: List[torch.Tensor] = []
    for layer in hidden_layers:
        pooled_vectors: List[torch.Tensor] = []
        for batch_idx, (start, end) in enumerate(spans):
            if end - start != num_image_tokens:
                raise ValueError("Image span length mismatch detected.")
            span_hidden = layer[batch_idx, start:end, :]
            if pooling == "mean":
                pooled = span_hidden.mean(dim=0)
            else:
                raise ValueError(f"Unsupported image_pooling '{pooling}'.")
            pooled_vectors.append(pooled)
        pooled_layers.append(torch.stack(pooled_vectors, dim=0))
    return pooled_layers


def _to_multilayer_embedding(
    layer_tensors: Sequence[torch.Tensor],
    *,
    domain: str,
) -> MultilayerEmbedding:
    per_layer: List[np.ndarray] = []
    for idx, tensor in enumerate(layer_tensors):
        array = tensor.detach().to(dtype=torch.float32).cpu().numpy()
        per_layer.append(array)
        if idx in {0, len(layer_tensors) - 1}:
            _debug_layer_stats(domain, f"layer_{idx:02d}", array)
    pooled = per_layer[-1] if per_layer else None
    return MultilayerEmbedding(per_layer=per_layer, pooled=pooled)


def _resolve_image_token_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<image>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer does not contain a valid '<image>' token.")
    return int(token_id)


def _ensure_attention_mask(model_inputs: Mapping[str, object]) -> Mapping[str, object]:
    if "attention_mask" in model_inputs:
        return model_inputs
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("Tokenizer outputs must include input_ids tensor when attention_mask is missing.")
    model_inputs = dict(model_inputs)
    model_inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
    return model_inputs


def _move_tensors(data: Mapping[str, object], *, device: torch.device) -> Mapping[str, object]:
    moved: Dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _validate_spans(span_info: SequenceSpanInfo) -> None:
    for fused_len, (start, end) in zip(span_info.fused_lengths, span_info.image_spans, strict=False):
        if end - start != span_info.num_image_tokens:
            raise ValueError("Image span length mismatch detected during validation.")
        if not (0 <= start < end <= fused_len):
            raise ValueError("Image span indices fall outside the fused sequence.")
    for fused_len, last_idx in zip(span_info.fused_lengths, span_info.text_last_indices, strict=False):
        if not (0 <= last_idx < fused_len):
            raise ValueError("Text last-token index is outside the fused sequence.")


def _prepare_pixel_values(images: Sequence[Image.Image], processor) -> torch.Tensor:
    if not images:
        raise ValueError("No images provided for encoding.")

    image_processor = processor.image_processor
    crop_size_dict = getattr(image_processor, "crop_size", None)
    if crop_size_dict and "height" in crop_size_dict and "width" in crop_size_dict:
        target_size = {"height": crop_size_dict["height"], "width": crop_size_dict["width"]}
    else:
        size_dict = getattr(image_processor, "size", None)
        if size_dict and "shortest_edge" in size_dict:
            edge = size_dict["shortest_edge"]
            target_size = {"height": edge, "width": edge}
        else:
            target_size = {"height": 224, "width": 224}

    final_size = min(target_size.values())
    kwargs = dict(
        do_resize=True,
        size={"shortest_edge": final_size},
        resample=2,
        do_center_crop=True,
        crop_size=target_size,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        return_tensors="np",
    )

    batch = image_processor(images=list(images), **kwargs)
    pixel_values = batch["pixel_values"]
    pixel_values = np.array(pixel_values, copy=True)
    if isinstance(pixel_values, np.ndarray):
        final_tensor = torch.from_numpy(pixel_values)
    elif isinstance(pixel_values, (list, tuple)):
        final_tensor = torch.from_numpy(np.stack(pixel_values))
    else:
        raise TypeError(f"Unexpected pixel_values type: {type(pixel_values)}")
    if final_tensor.ndim == 3:
        final_tensor = final_tensor.unsqueeze(0)
    return final_tensor


def _select_transformer_layers(hidden_states) -> List[torch.Tensor]:
    if not hidden_states:
        return []
    if len(hidden_states) > 1:
        return list(hidden_states[1:])
    return list(hidden_states)


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


def _tokenize_texts(tokenizer, texts: Sequence[str]):
    return tokenizer(  # type: ignore[call-arg]
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D [batch, seq_len].")
    mask = attention_mask.to(dtype=torch.long)
    lengths = mask.sum(dim=1) - 1
    return torch.clamp(lengths, min=0)


def _ensure_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
    else:
        array = np.array(image)
    if array.dtype not in (np.uint8, np.uint16):
        max_val = float(array.max()) if array.size else 1.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array).convert("RGB")


def _truncate_text_preview(text: str, limit: int = 60) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _debug_layer_stats(domain: str, label: str, vectors: np.ndarray) -> None:
    if vectors.size == 0:
        print(f"[debug][{domain}] {label}: empty vectors")
        return
    matrix = vectors if vectors.ndim == 2 else np.reshape(vectors, (1, -1))
    norms = np.linalg.norm(matrix, axis=1)
    first = matrix[0][: min(5, matrix.shape[1])]
    preview = np.array2string(first, precision=3, separator=", ")
    print(
        f"[debug][{domain}] {label}: shape={matrix.shape} "
        f"norm[min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}] "
        f"first[:5]={preview}"
    )


def _resolve_vision_tower(model) -> nn.Module:
    tower = None
    if hasattr(model, "get_vision_tower"):
        tower = model.get_vision_tower()
    if tower is None:
        tower = getattr(model, "vision_tower", None)
    if tower is None:
        raise AttributeError("Model does not expose a vision tower accessor.")
    if isinstance(tower, (list, tuple)) and tower:
        tower = tower[0]
    if isinstance(tower, nn.ModuleList) and len(tower) > 0:
        tower = tower[0]
    for attr in ("vision_tower", "vision_model"):
        tower = getattr(tower, attr, tower)
    if not isinstance(tower, nn.Module):
        raise TypeError("Resolved vision tower is not a torch.nn.Module.")
    return tower


def _resolve_language_model(model) -> nn.Module:
    for attr in ("language_model", "model"):
        language_model = getattr(model, attr, None)
        if isinstance(language_model, nn.Module):
            return language_model
    if isinstance(model, nn.Module):
        return model
    raise TypeError("Unable to resolve language model from the provided LLaVA instance.")


def _resolve_mm_projector(model) -> nn.Module | None:
    def _probe(container) -> nn.Module | None:
        if container is None:
            return None
        for attr in ("mm_projector", "multi_modal_projector", "vision_projector", "projector"):
            module = getattr(container, attr, None)
            if isinstance(module, nn.Module):
                return module
        return None

    projector = _probe(model)
    if projector is not None:
        return projector
    core = getattr(model, "model", None)
    projector = _probe(core)
    if projector is not None:
        return projector
    get_model = getattr(model, "get_model", None)
    if callable(get_model):
        return _probe(get_model())
    return None


def _module_device_dtype(
    module: nn.Module,
    fallback_device=None,
) -> tuple[torch.device, torch.dtype]:
    try:
        first_param = next(module.parameters())
        return first_param.device, first_param.dtype
    except StopIteration:
        device = fallback_device if fallback_device is not None else torch.device("cpu")
        dtype = torch.float32
        return device, dtype


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e5, neginf=-1e5)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    clipped_norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / clipped_norms
