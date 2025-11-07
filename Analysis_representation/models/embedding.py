"""Lightweight helpers to extract normalised embeddings from multimodal models."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from data.schemas import MultilingualExample


EmbeddingMap = Dict[str, np.ndarray]


def encode_examples(
    examples: Iterable[MultilingualExample],
    model,
    processor,
) -> Tuple[EmbeddingMap, np.ndarray]:
    """Return language->embedding matrix and image embeddings for the provided examples."""

    texts_by_language: dict[str, list[str]] = {}
    image_inputs: List[object] = []
    example_list = list(examples)
    for example in example_list:
        # 将图像转换为 Processor 可直接消费的输入（PIL 或路径）
        image_inputs.append(example.image.to_model_input())
        for language, caption in example.captions.items():
            # 逐语言累积文本，保持与图像顺序一致
            texts_by_language.setdefault(language, []).append(caption.text)

    image_embeddings = _encode_images(image_inputs, model, processor)
    text_embeddings = {
        language: _encode_texts(texts, model, processor)
        for language, texts in texts_by_language.items()
    }
    return text_embeddings, image_embeddings


def _encode_images(
    images: Sequence[object],
    model,
    processor,
) -> np.ndarray:
    inputs = processor(images=list(images), return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    embeddings = outputs.detach().cpu().numpy()
    return _normalize(embeddings)


def _encode_texts(
    texts: Sequence[str],
    model,
    processor,
) -> np.ndarray:
    inputs = processor(text=list(texts), return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    embeddings = outputs.detach().cpu().numpy()
    return _normalize(embeddings)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)
