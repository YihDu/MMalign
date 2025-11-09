#!/usr/bin/env python3
"""Layer/condition plots for language↔image 与 语言↔语言指标，附带更丰富统计。"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MetricMap = Mapping[str, Mapping[str, Sequence[float]]]

LANG_IMAGE_PATTERN = re.compile(
    r"^cosine_image_text_(?P<language>.+?)_(?P<label>(?:layer|final).*)$"
)
LANG_PAIR_PATTERN = re.compile(
    r"^cosine_(?P<lang_a>.+?)__vs__(?P<lang_b>.+?)_(?P<label>(?:layer|final).*)$"
)


@dataclass
class MetricSummary:
    mean: float
    std: float
    median: float
    values: np.ndarray


LanguageImageStats = Dict[str, Dict[str, Dict[str, MetricSummary]]]
LanguagePairStats = Dict[str, Dict[str, Dict[str, MetricSummary]]]

DEFAULT_BASE_CONDITION = "correct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 distance_map JSON 生成两套图："
            "语言↔图像 与 语言↔语言 层趋势，并标注 mean/std/margin/win-rate。"
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="run_experiment 生成的 distance_map JSON。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出文件前缀；若提供 .png/.pdf 后缀会自动派生 *_lang_image 与 *_language_pairs。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="可选全局标题（默认使用 results 文件名）。",
    )
    parser.add_argument(
        "--base-condition",
        type=str,
        default=DEFAULT_BASE_CONDITION,
        help="用于 margin / win-rate 参考的条件（默认 correct）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distance_map = _load_distance_map(args.results)
    language_image = _group_language_image(distance_map)
    language_pairs = _group_language_pairs(distance_map)

    if not language_image and not language_pairs:
        raise SystemExit("未找到 cosine_image_text_* 或 cosine_*__vs__* 指标。")

    base_title = args.title or args.results.stem
    if language_image:
        output_path = _derive_output(args.output, suffix="_lang_image")
        plot_language_category(
            stats=language_image,
            title=f"{base_title} · 语言↔图像",
            output_path=output_path,
            base_condition=args.base_condition,
            section_label="语言-图像",
        )

    if language_pairs:
        output_path = _derive_output(args.output, suffix="_language_pairs")
        plot_language_category(
            stats=language_pairs,
            title=f"{base_title} · 语言↔语言",
            output_path=output_path,
            base_condition=args.base_condition,
            section_label="多语言",
        )


def _load_distance_map(path: Path) -> MetricMap:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_language_image(distance_map: MetricMap) -> LanguageImageStats:
    grouped: LanguageImageStats = {}
    for condition, metrics in distance_map.items():
        for metric_name, values in metrics.items():
            match = LANG_IMAGE_PATTERN.match(metric_name)
            if not match:
                continue
            language = match.group("language")
            label = match.group("label")
            grouped.setdefault(language, {}).setdefault(condition, {})[label] = _summarize(values)
    return grouped


def _group_language_pairs(distance_map: MetricMap) -> LanguagePairStats:
    grouped: LanguagePairStats = {}
    for condition, metrics in distance_map.items():
        for metric_name, values in metrics.items():
            match = LANG_PAIR_PATTERN.match(metric_name)
            if not match:
                continue
            lang_a = match.group("lang_a")
            lang_b = match.group("lang_b")
            label = match.group("label")
            pair_label = f"{lang_a} ↔ {lang_b}"
            grouped.setdefault(pair_label, {}).setdefault(condition, {})[label] = _summarize(values)
    return grouped


def _summarize(values: Sequence[float]) -> MetricSummary:
    arr = np.asarray(values, dtype=float)
    return MetricSummary(
        mean=float(np.nanmean(arr)) if arr.size else float("nan"),
        std=float(np.nanstd(arr)) if arr.size else float("nan"),
        median=float(np.nanmedian(arr)) if arr.size else float("nan"),
        values=arr,
    )


def plot_language_category(
    *,
    stats: Mapping[str, Mapping[str, Mapping[str, MetricSummary]]],
    title: str,
    output_path: Path,
    base_condition: str,
    section_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    languages = sorted(stats.keys())
    n_langs = len(languages)
    ncols = min(3, n_langs)
    nrows = math.ceil(n_langs / ncols)
    condition_order = _collect_conditions(stats)
    color_map = _build_condition_colors(condition_order)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.4 * ncols, 4.6 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.flatten()

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[condition],
            marker="o",
            linewidth=2,
            markersize=5,
            label=condition,
        )
        for condition in condition_order
    ]

    for idx, language in enumerate(languages):
        ax = axes_flat[idx]
        _apply_panel_style(ax)
        condition_map = stats[language]
        label_order = _sorted_labels(condition_map)
        x_positions = np.arange(len(label_order))

        for condition in condition_order:
            label_values = condition_map.get(condition)
            if not label_values:
                continue
            means = np.array([label_values[label].mean for label in label_order], dtype=float)
            stds = np.array([label_values[label].std for label in label_order], dtype=float)
            ax.plot(
                x_positions,
                means,
                marker="o",
                linewidth=2,
                markersize=4.5,
                label=condition,
                color=color_map[condition],
            )
            if np.any(np.isfinite(stds)):
                lower = means - stds
                upper = means + stds
                ax.fill_between(
                    x_positions,
                    lower,
                    upper,
                    color=color_map[condition],
                    alpha=0.12,
                    linewidth=0,
                )

        ax.set_title(f"{section_label}: {language}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_format_label(label) for label in label_order],
            rotation=32,
            ha="right",
        )
        ax.set_ylabel("Cosine Distance")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        summary_lines = _format_summary_lines(condition_map, base_condition)
        if summary_lines:
            ax.text(
                0.02,
                0.02,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                fontsize=8.5,
                va="bottom",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

    for extra_ax in axes_flat[n_langs:]:
        extra_ax.axis("off")

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(len(legend_handles), 3),
            frameon=False,
        )

    fig.suptitle(title, fontsize=15, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _derive_output(base_path: Path, *, suffix: str) -> Path:
    if base_path.suffix:
        stem = base_path.with_suffix("")
        ext = base_path.suffix
    else:
        stem = base_path
        ext = ".png"
    return stem.with_name(stem.name + suffix + ext)


def _sorted_labels(
    condition_map: Mapping[str, Mapping[str, MetricSummary]]
) -> Sequence[str]:
    labels = {label for metrics in condition_map.values() for label in metrics.keys()}

    def key(label: str) -> tuple[int, float, str]:
        base = label.split("__img_", 1)[0]
        layer_match = re.search(r"layer_(\d+)", base)
        if layer_match:
            return (0, float(layer_match.group(1)), label)
        if base.startswith("final"):
            return (1, float("inf"), label)
        return (2, float("inf"), label)

    return sorted(labels, key=key)


def _collect_conditions(
    stats: Mapping[str, Mapping[str, Mapping[str, MetricSummary]]]
) -> Sequence[str]:
    conditions = set()
    for condition_map in stats.values():
        conditions.update(condition_map.keys())
    return sorted(conditions)


def _build_condition_colors(conditions: Sequence[str]) -> Mapping[str, str]:
    if not conditions:
        return {}
    cmap = plt.get_cmap("tab10")
    denom = max(1, len(conditions) - 1)
    return {condition: cmap(idx / denom) for idx, condition in enumerate(conditions)}


def _apply_panel_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#f9f9fb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_label(label: str) -> str:
    label = label.replace("layer_", "L")
    label = label.replace("__img_", "+img ")
    return label


def _format_summary_lines(
    condition_map: Mapping[str, Mapping[str, MetricSummary]],
    base_condition: str,
) -> Sequence[str]:
    lines = []
    base = condition_map.get(base_condition)
    for condition, metrics in sorted(condition_map.items()):
        means = [summary.mean for summary in metrics.values() if np.isfinite(summary.mean)]
        if not means:
            continue
        text = f"{condition}: mean={np.mean(means):.3f}"
        if base and condition != base_condition:
            margin, win = _condition_gap(metrics, base)
            if margin is not None and win is not None:
                text += f" · Δ={margin:.3f} · win={win:.2f}"
        lines.append(text)
    return lines


def _condition_gap(
    metrics: Mapping[str, MetricSummary],
    base_metrics: Mapping[str, MetricSummary],
) -> tuple[float | None, float | None]:
    diffs = []
    for label, summary in metrics.items():
        base = base_metrics.get(label)
        if base is None or summary.values.size == 0 or base.values.size == 0:
            continue
        if summary.values.shape != base.values.shape:
            continue
        diffs.append(summary.values - base.values)
    if not diffs:
        return None, None
    merged = np.concatenate(diffs)
    return float(np.mean(merged)), float(np.mean(merged > 0))


if __name__ == "__main__":
    main()
