"""Lightweight helpers to evaluate distances under different conditions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from pathlib import Path
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
    debug_csv_path: Path | None = None,
    micro_batch_size: int | None = None,
) -> Dict[str, Dict[str, Sequence[float]]]:
    """Return cosine distance lists for each condition/language/layer."""

    analysis_config = analysis_config or {}
    raw_layer_mode = analysis_config.get("layer_mode", DEFAULT_LAYER_MODE)
    layer_mode = str(raw_layer_mode) if raw_layer_mode is not None else DEFAULT_LAYER_MODE
    layer_indices = analysis_config.get("layer_indices", []) or []

    results: Dict[str, Dict[str, Sequence[float]]] = {}
    debug_rows: list[dict[str, object]] = []
    for name, builder in conditions.items():
        conditioned_examples = builder(batch.examples)
        print(
            f"[debug][condition:{name}] examples={len(conditioned_examples)} "
            f"layer_mode={layer_mode}"
        )
        _preview_condition_samples(name, conditioned_examples, languages)
        print("开始获取Embedding")
        fusion_overrides = analysis_config.get("fusion") if isinstance(analysis_config, Mapping) else None
        embeddings = encode_examples(
            conditioned_examples,
            model,
            processor,
            fusion_config=fusion_overrides,
            micro_batch_size=micro_batch_size,
        )

        distances: Dict[str, Sequence[float]] = {}
        text_layers_by_language: Dict[str, Sequence[Tuple[str, np.ndarray]]] = {}
        for language in languages:
            language_embeddings = embeddings.captions.get(language)
            if language_embeddings is None:
                continue
            text_layers = _select_layers(language_embeddings.text, layer_mode, layer_indices)
            image_layers = _select_layers(language_embeddings.image, layer_mode, layer_indices)
            if not text_layers or not image_layers:
                continue
            text_layers_by_language[language] = text_layers

        for language, text_layers in text_layers_by_language.items():
            language_embeddings = embeddings.captions.get(language)
            if language_embeddings is None:
                continue
            image_layers = _select_layers(language_embeddings.image, layer_mode, layer_indices)
            layer_pairs = _pair_layers(text_layers, image_layers)
            if not layer_pairs:
                continue
            for layer_label, text_vectors, image_vectors in layer_pairs:
                print(
                    f"[debug][metric-input] lang={language} layer={layer_label} "
                    f"text_shape={text_vectors.shape} image_shape={image_vectors.shape}"
                )
                dist = cosine_distance_matrix(text_vectors, image_vectors)
                diag = _diagonal(dist)
                summary = _matrix_stats(dist)
                metric_name = f"cosine_image_text_{language}_{layer_label}"
                distances[metric_name] = diag
                distances[f"{metric_name}__diag_summary"] = [summary.diag_mean, summary.diag_std]
                distances[f"{metric_name}__off_summary"] = [summary.off_mean, summary.off_std]
                debug_rows.append(
                    {
                        "condition": name,
                        "metric_type": "image_text",
                        "language": language,
                        "layer": layer_label,
                        "diag_mean": summary.diag_mean,
                        "diag_std": summary.diag_std,
                        "off_mean": summary.off_mean,
                        "off_std": summary.off_std,
                    }
                )
                _print_distance_stats(
                    condition=name,
                    metric=metric_name,
                    values=diag,
                    extra=summary.message,
                )

        for lang_a, lang_b in combinations(sorted(text_layers_by_language.keys()), 2):
            layers_a = text_layers_by_language[lang_a]
            layers_b = text_layers_by_language[lang_b]
            layer_pairs = _pair_layers(layers_a, layers_b)
            if not layer_pairs:
                continue
            for layer_label, text_vectors_a, text_vectors_b in layer_pairs:
                dist = cosine_distance_matrix(text_vectors_a, text_vectors_b)
                diag = _diagonal(dist)
                summary = _matrix_stats(dist)
                metric_name = f"cosine_{lang_a}__vs__{lang_b}_{layer_label}"
                distances[metric_name] = diag
                distances[f"{metric_name}__diag_summary"] = [summary.diag_mean, summary.diag_std]
                distances[f"{metric_name}__off_summary"] = [summary.off_mean, summary.off_std]
                debug_rows.append(
                    {
                        "condition": name,
                        "metric_type": "language_pair",
                        "language": f"{lang_a}__vs__{lang_b}",
                        "layer": layer_label,
                        "diag_mean": summary.diag_mean,
                        "diag_std": summary.diag_std,
                        "off_mean": summary.off_mean,
                        "off_std": summary.off_std,
                    }
                )
                _print_distance_stats(
                    condition=name,
                    metric=metric_name,
                    values=diag,
                    extra=summary.message,
                )

        results[name] = distances

    sanity_cfg = {}
    if isinstance(analysis_config, Mapping):
        sanity_cfg = analysis_config.get("sanity_checks", {}) or {}
    _maybe_warn_small_condition_gap(results, sanity_cfg)
    _write_debug_metrics(debug_rows, debug_csv_path)
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


def _diagonal(distances: np.ndarray) -> Sequence[float]:
    limit = min(distances.shape[0], distances.shape[1])
    return [float(distances[i, i]) for i in range(limit)]


@dataclass
class DistanceMatrixStats:
    message: str
    diag_mean: float
    diag_std: float
    off_mean: float
    off_std: float


def _matrix_stats(distances: np.ndarray) -> DistanceMatrixStats:
    diag = np.diag(distances) if distances.ndim == 2 else np.array([])
    off_mask = ~np.eye(distances.shape[0], dtype=bool) if distances.ndim == 2 else None
    off_diag = distances[off_mask] if off_mask is not None else np.array([])
    diag_mean = float(np.mean(diag)) if diag.size else float("nan")
    diag_std = float(np.std(diag)) if diag.size else float("nan")
    off_mean = float(np.mean(off_diag)) if off_diag.size else float("nan")
    off_std = float(np.std(off_diag)) if off_diag.size else float("nan")
    message = (
        f"shape={distances.shape} "
        f"diag_mean={diag_mean:.3f} diag_std={diag_std:.3f} "
        f"off_mean={off_mean:.3f} off_std={off_std:.3f}"
    )
    return DistanceMatrixStats(
        message=message,
        diag_mean=diag_mean,
        diag_std=diag_std,
        off_mean=off_mean,
        off_std=off_std,
    )


def _pair_layers(
    layers_a: Sequence[Tuple[str, np.ndarray]],
    layers_b: Sequence[Tuple[str, np.ndarray]],
) -> Sequence[Tuple[str, np.ndarray, np.ndarray]]:
    if not layers_a or not layers_b:
        return []

    count = min(len(layers_a), len(layers_b))
    if len(layers_a) != len(layers_b):
        print(
            "[warn] Layer counts differ: "
            f"a={len(layers_a)} b={len(layers_b)}; truncating to {count}."
        )

    paired = []
    for idx in range(count):
        label_a, vectors_a = layers_a[idx]
        label_b, vectors_b = layers_b[idx]
        layer_label = label_a if label_a == label_b else f"{label_a}__vs__{label_b}"
        paired.append((layer_label, vectors_a, vectors_b))
    return paired


def _preview_condition_samples(
    condition: str,
    examples,
    languages: Sequence[str],
    limit: int = 2,
) -> None:
    if not examples:
        print(f"[debug][condition:{condition}] 无样本可预览")
        return
    for idx, example in enumerate(examples[:limit]):
        image_id = example.image.image_id
        filename = example.image.filename or "N/A"
        print(
            f"[debug][condition:{condition}] sample#{idx} image_id={image_id} filename={filename}"
        )
        for language in languages:
            caption = example.captions.get(language)
            if caption is None:
                print(f"  - {language}: <缺失>")
                continue
            text = caption.text
            preview = text if len(text) <= 50 else text[:47] + "..."
            print(f"  - {language}: len={len(text)} text=\"{preview}\"")


def _print_distance_stats(
    *,
    condition: str,
    metric: str,
    values: Sequence[float],
    extra: str,
) -> None:
    if not values:
        print(f"[debug][metric] condition={condition} metric={metric} 无有效值")
        return
    arr = np.array(values, dtype=float)
    first_vals = ", ".join(f"{val:.3f}" for val in arr[:5])
    print(
        "[debug][metric] "
        f"condition={condition} metric={metric} "
        f"min={arr.min():.3f} mean={arr.mean():.3f} max={arr.max():.3f} "
        f"first={first_vals} {extra}"
    )


def _maybe_warn_small_condition_gap(
    results: Mapping[str, Mapping[str, Sequence[float]]],
    sanity_cfg: Mapping[str, object],
) -> None:
    if sanity_cfg is False:
        return
    enabled = bool(sanity_cfg.get("enabled", True))
    if not enabled:
        return

    base_condition = str(sanity_cfg.get("base_condition", "correct"))
    compare_condition = str(sanity_cfg.get("compare_condition", "mismatched"))
    mean_tol = float(sanity_cfg.get("mean_diff_tol", 0.01))
    win_tol = float(sanity_cfg.get("win_rate_tol", 0.05))
    min_metrics = int(sanity_cfg.get("min_metrics", 5))

    base_metrics = results.get(base_condition)
    compare_metrics = results.get(compare_condition)
    if not base_metrics or not compare_metrics:
        return

    flagged = []
    for metric, base_values in base_metrics.items():
        if metric.endswith("__diag_summary") or metric.endswith("__off_summary"):
            continue
        compare_values = compare_metrics.get(metric)
        if compare_values is None:
            continue
        arr_base = np.asarray(base_values, dtype=float)
        arr_comp = np.asarray(compare_values, dtype=float)
        if arr_base.size == 0 or arr_comp.size == 0 or arr_base.shape != arr_comp.shape:
            continue
        diff = arr_comp - arr_base
        mean_gap = float(np.mean(diff))
        win_rate = float(np.mean(diff > 0))
        if abs(mean_gap) < mean_tol and abs(win_rate - 0.5) < win_tol:
            flagged.append((metric, mean_gap, win_rate))

    if len(flagged) >= max(1, min_metrics):
        print(
            "[warn][sanity] 多个 metric 在 "
            f"{compare_condition} vs {base_condition} 下差异过小 "
            f"(mean_tol={mean_tol}, win_tol={win_tol})."
        )
        for metric, gap, win in flagged[:8]:
            print(f"  - {metric}: mean_gap={gap:.5f} win_rate={win:.3f}")


def _write_debug_metrics(rows: Sequence[Mapping[str, object]], debug_path: Path | None) -> None:
    if not rows:
        return
    import csv
    output_path = debug_path or Path("results/debug_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "metric_type",
        "language",
        "layer",
        "diag_mean",
        "diag_std",
        "off_mean",
        "off_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[debug] metric summaries written to {output_path}")
