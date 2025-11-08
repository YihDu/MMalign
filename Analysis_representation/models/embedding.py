"""Hidden-state extraction utilities built on top of LLaVA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

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
class EmbeddingBatch:
    """Grouped embeddings for images and multilingual captions."""

    images: MultilayerEmbedding
    captions: Dict[str, MultilayerEmbedding]


def encode_examples(
    examples: Iterable[MultilingualExample],
    model,
    processor,
) -> EmbeddingBatch:
    """抽取批量样本的图像/文本多层 hidden state。

    步骤：
    1. 将每个样本的图像统一转成 PIL，并按语言累积文本。
    2. 调用图像/文本编码函数获取各自的 per-layer 表示。
    3. 打包为 EmbeddingBatch，方便下游按语言读取。
    """

    texts_by_language: dict[str, list[str]] = {}
    image_inputs: List[Image.Image] = []
    for example in examples:
        # 统一将图像转成 RGB PIL，方便 processor 做预处理
        image_inputs.append(_ensure_pil(example.image.to_model_input()))
        for language, caption in example.captions.items():
            # 多语言文本分别累计，保持与图像索引一致
            texts_by_language.setdefault(language, []).append(caption.text)

    image_embeddings = _encode_images_multilayer(
        images=image_inputs,
        model=model,
        processor=processor,
    )
    text_embeddings = {
        language: _encode_texts_multilayer(texts, model, processor)
        for language, texts in texts_by_language.items()
    }
    return EmbeddingBatch(images=image_embeddings, captions=text_embeddings)


def _encode_images_multilayer(
    images: Sequence[Image.Image],
    model,
    processor,
    normalize: bool = True,
) -> MultilayerEmbedding:
    """按照 LLaVA 规范通过 vision tower 提取图像各层 hidden state。

    步骤：
    1. 利用 processor 生成符合模型要求的 pixel_values。
    2. 将像素迁移到 vision tower 的设备/精度，并开启 output_hidden_states。
    3. 对每层取 CLS patch，必要时做归一化，组成 MultilayerEmbedding。
    """

    fallback_device = _model_device(model)
    # 官方接口：通过 get_vision_tower() 获取真正的 CLIPVisionModel
    vision_tower = _resolve_vision_tower(model)
    pixel_values = _prepare_pixel_values(images, processor)
    device, dtype = _module_device_dtype(vision_tower, fallback_device=fallback_device)
    pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
    projector = _resolve_mm_projector(model)

    with torch.no_grad():
        outputs = vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = _select_transformer_layers(outputs.hidden_states)
    per_layer: List[np.ndarray] = []
    for layer in hidden_states:
        # 图像侧用 CLS patch（索引 0）代表整幅图
        cls_tokens = layer[:, 0, :]
        if projector is not None:
            cls_tokens = projector(cls_tokens)
        layer_np = cls_tokens.detach().cpu().float().numpy()
        per_layer.append(_normalize(layer_np) if normalize else layer_np)

    pooled = None
    if getattr(outputs, "pooler_output", None) is not None:
        pooled_np = outputs.pooler_output.detach().cpu().float().numpy()
        pooled = _normalize(pooled_np) if normalize else pooled_np

    return MultilayerEmbedding(per_layer=per_layer, pooled=pooled)


def _encode_texts_multilayer(
    texts: Sequence[str],
    model,
    processor,
    normalize: bool = True,
) -> MultilayerEmbedding:
    """Extract multilayer hidden states from the language model branch."""

    if not texts:
        raise ValueError("No texts provided for encoding.")

    tokenizer = _resolve_tokenizer(processor)
    tokenized = _tokenize_texts(tokenizer, texts)
    language_model = _resolve_language_model(model)

    fallback_device = _model_device(model)
    device, _ = _module_device_dtype(language_model, fallback_device=fallback_device)

    model_inputs: Dict[str, object] = {}
    for key, value in tokenized.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device=device, non_blocking=True)
        else:
            model_inputs[key] = value

    if "attention_mask" not in model_inputs:
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Tokenizer must provide attention_mask or input_ids tensors.")
        model_inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)

    attention_mask = model_inputs["attention_mask"]
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor after preprocessing.")

    with torch.no_grad():
        outputs = language_model(  # type: ignore[call-arg]
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = _select_transformer_layers(outputs.hidden_states)
    last_token_indices = _last_token_indices(attention_mask)

    per_layer: List[np.ndarray] = []
    for layer in hidden_states:
        if layer.ndim != 3:
            raise ValueError("Language model hidden states must have shape [B, T, H].")
        batch_size = layer.size(0)
        if batch_size == 0:
            continue
        gather_indices = last_token_indices.to(layer.device)
        batch_positions = torch.arange(batch_size, device=layer.device)
        token_vectors = layer[batch_positions, gather_indices, :]
        layer_np = token_vectors.detach().cpu().float().numpy()
        per_layer.append(_normalize(layer_np) if normalize else layer_np)

    pooled = per_layer[-1] if per_layer else None
    return MultilayerEmbedding(per_layer=per_layer, pooled=pooled)


def _prepare_pixel_values(images: Sequence[Image.Image], processor) -> torch.Tensor:
    """使用 image_processor 做 resize/crop/normalize 并返回像素张量。
    自动兼容单图像或多图像输入，输出标准 4D torch.Tensor [B, C, H, W]。
    """

    if not images:
        raise ValueError("No images provided for encoding.")

    image_processor = processor.image_processor

    # 自动推断目标尺寸
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

    # 构造处理参数
    kwargs = dict(
        do_resize=True,
        size={"shortest_edge": final_size},
        resample=2,  # PIL.Image.BICUBIC
        do_center_crop=True,
        crop_size=target_size,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        return_tensors="np",  # 先输出 numpy，方便统一栈处理
    )

    batch = image_processor(images=list(images), **kwargs)
    
    pixel_values = batch["pixel_values"]
    pixel_values = np.array(pixel_values, copy=True)
    
    # 自动判断类型：可能是 np.ndarray 或 list[np.ndarray]
    if isinstance(pixel_values, np.ndarray):
        final_tensor = torch.from_numpy(pixel_values)
    elif isinstance(pixel_values, (list, tuple)):
        # 处理为 list of np.ndarray -> stack
        final_tensor = torch.from_numpy(np.stack(pixel_values))
    else:
        raise TypeError(f"Unexpected pixel_values type: {type(pixel_values)}")

    # 最终保证返回形状为 [B, C, H, W]
    if final_tensor.ndim == 3:
        final_tensor = final_tensor.unsqueeze(0)

    return final_tensor


def _select_transformer_layers(hidden_states) -> List[torch.Tensor]:
    """去除 embedding 层，仅保留 transformer block 输出。

    步骤：若 hidden_states 长度>1，则跳过索引 0（embedding），返回其余层；否则直接返回原列表。
    """

    if not hidden_states:
        return []
    if len(hidden_states) > 1:
        return list(hidden_states[1:])
    return list(hidden_states)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """对批量向量执行 L2 归一化并清理 NaN/Inf。

    步骤：先用 nan_to_num 清理，再计算范数并裁剪，最后除以范数得到 unit 向量。
    """

    if not np.all(np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e5, neginf=-1e5)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    clipped_norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / clipped_norms


def _resolve_tokenizer(processor):
    """Best-effort extraction of a tokenizer from the processor object."""

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
    """Tokenize texts with sensible defaults for multilayer extraction."""

    return tokenizer(  # type: ignore[call-arg]
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute the final valid token index for each sequence based on the mask."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D [batch, seq_len].")
    mask = attention_mask.to(dtype=torch.long)
    lengths = mask.sum(dim=1) - 1
    return torch.clamp(lengths, min=0)


def _ensure_pil(image) -> Image.Image:
    """将任意图像输入安全转换为 RGB PIL。"""

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


def _resolve_vision_tower(model) -> nn.Module:
    """解析 LLaVA 中的 vision tower（CLIPVisionModel）。"""

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
    # unwrap common containers used by LLaVA
    for attr in ("vision_tower", "vision_model"):
        tower = getattr(tower, attr, tower)
    if not isinstance(tower, nn.Module):
        raise TypeError("Resolved vision tower is not a torch.nn.Module.")
    return tower


def _resolve_language_model(model) -> nn.Module:
    """解析 LLaVA 中的语言模型（LLaMA）。"""

    for attr in ("language_model", "model"):
        language_model = getattr(model, attr, None)
        if isinstance(language_model, nn.Module):
            return language_model
    if isinstance(model, nn.Module):
        return model
    raise TypeError("Unable to resolve language model from the provided LLaVA instance.")


def _resolve_mm_projector(model) -> nn.Module | None:
    """Locate the multimodal projector that maps vision tokens to text space."""

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
    """返回模块参数所在的 device/dtype。"""

    try:
        first_param = next(module.parameters())
        return first_param.device, first_param.dtype
    except StopIteration:
        device = fallback_device if fallback_device is not None else torch.device("cpu")
        dtype = torch.float32
        return device, dtype


def _model_device(model) -> torch.device:
    """推断顶层模型默认运行设备。"""

    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
