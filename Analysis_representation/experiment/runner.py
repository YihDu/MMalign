"""Lightweight helpers to evaluate distances under different conditions."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from analysis.metrics import cosine_distance_matrix
from data.schemas import MultilingualExample, SampleBatch
from models.embedding import encode_examples

from .conditions import ConditionBuilder


def run_experiment(
    *,
    model,
    processor,
    batch: SampleBatch,
    languages: Sequence[str],
    conditions: Mapping[str, ConditionBuilder],
) -> Dict[str, Dict[str, Sequence[float]]]:
    """Return cosine distance lists for each condition/language."""

    results: Dict[str, Dict[str, Sequence[float]]] = {}
    for name, builder in conditions.items():
        # 条件构造器会对样本顺序/内容做调整（如错位配对）
        conditioned_examples = builder(batch.examples)
        text_map, image_vectors = encode_examples(
            conditioned_examples,
            model,
            processor,
        )
        distances: Dict[str, Sequence[float]] = {}
        for language in languages:
            # 针对每种语言计算文本-图像嵌入的对角余弦距离
            text_vectors = text_map[language]
            dist = cosine_distance_matrix(text_vectors, image_vectors)
            diag = [float(dist[i, i]) for i in range(min(dist.shape[0], dist.shape[1]))]
            distances[f"cosine_{language}"] = diag
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
