"""Lightweight helpers to evaluate distances under different conditions."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from analysis.metrics import cosine_distance_matrix
from data.schemas import SampleBatch
from models.embedding import MultilayerEmbedding, encode_examples

from .conditions import ConditionBuilder

DEFAULT_LAYER_MODE = "final"
VALID_LAYER_MODES = {"all", "indices", "final"}


def run_experiment(
    *,
    model,
    processor,
    batch: SampleBatch,
    languages: Sequence[str],
    conditions: Mapping[str, ConditionBuilder],
    analysis_config: Mapping[str, object] | None = None,
) -> Dict[str, Dict[str, Sequence[float]]]:
    """Return cosine distance lists for each condition/language/layer."""

    analysis_config = analysis_config or {}
    raw_layer_mode = analysis_config.get("layer_mode", DEFAULT_LAYER_MODE)
    layer_mode = str(raw_layer_mode) if raw_layer_mode is not None else DEFAULT_LAYER_MODE
    layer_indices = analysis_config.get("layer_indices", []) or []

    results: Dict[str, Dict[str, Sequence[float]]] = {}
    for name, builder in conditions.items():
        conditioned_examples = builder(batch.examples)
        print('开始获取Embedding')
        embeddings = encode_examples(conditioned_examples, model, processor)
        image_layers = _select_layers(embeddings.images, layer_mode, layer_indices)
        if not image_layers:
            raise ValueError("Image encoder did not return any hidden states.")

        distances: Dict[str, Sequence[float]] = {}
        for language in languages:
            caption_embeddings = embeddings.captions[language]
            text_layers = _select_layers(caption_embeddings, layer_mode, layer_indices)
            layer_pairs = _pair_layers(image_layers, text_layers)
            if not layer_pairs:
                continue
            for layer_label, text_vectors, image_vectors in layer_pairs:
                dist = cosine_distance_matrix(text_vectors, image_vectors)
                diag = [
                    float(dist[i, i])
                    for i in range(min(dist.shape[0], dist.shape[1]))
                ]
                metric_name = f"cosine_{language}_{layer_label}"
                distances[metric_name] = diag
        results[name] = distances
    return results


def summarise_results(
    distance_map: Mapping[str, Mapping[str, Sequence[float]]],
) -> Sequence[Mapping[str, float]]:
    """Compute mean distance per condition/language to ease reporting."""

    summary = []
    for condition, metrics in distance_map.items():
        row = {"condition": condition}
        for metric_name, values in metrics.items():
            row[metric_name] = float(np.mean(values)) if values else float("nan")
        summary.append(row)
    return summary


def _select_layers(
    embedding: MultilayerEmbedding,
    mode: str,
    indices: Sequence[int],
) -> Sequence[Tuple[str, np.ndarray]]:
    if mode not in VALID_LAYER_MODES:
        raise ValueError(f"Unsupported layer_mode '{mode}'")
    if mode == "final":
        if embedding.pooled is None:
            raise ValueError("Requested final layer but pooled embedding is missing.")
        return [("final", embedding.pooled)]

    layers = embedding.per_layer
    if not layers:
        return []

    if mode == "indices":
        if not indices:
            raise ValueError("layer_indices must be provided when layer_mode='indices'.")
        selected = []
        for idx in indices:
            if idx < 0 or idx >= len(layers):
                raise IndexError(f"Layer index {idx} out of range (total {len(layers)}).")
            selected.append((idx, layers[idx]))
    else:  # mode == "all"
        selected = list(enumerate(layers))

    return [(f"layer_{idx:02d}", layer) for idx, layer in selected]


def _pair_layers(
    image_layers: Sequence[Tuple[str, np.ndarray]],
    text_layers: Sequence[Tuple[str, np.ndarray]],
) -> Sequence[Tuple[str, np.ndarray, np.ndarray]]:
    if not image_layers or not text_layers:
        return []

    count = min(len(image_layers), len(text_layers))
    if len(image_layers) != len(text_layers):
        print(
            "[warn] Image/Text layer counts differ: "
            f"image={len(image_layers)} text={len(text_layers)}; truncating to {count}."
        )

    paired = []
    for idx in range(count):
        img_label, img_vectors = image_layers[idx]
        text_label, text_vectors = text_layers[idx]
        layer_label = text_label if text_label == img_label else f"{text_label}__img_{img_label}"
        paired.append((layer_label, text_vectors, img_vectors))
    return paired
