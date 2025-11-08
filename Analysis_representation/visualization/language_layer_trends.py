#!/usr/bin/env python3
"""Generate generic layer-wise plots from experiment distance maps."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

MetricMap = Mapping[str, Mapping[str, Sequence[float]]]
LanguageStats = Dict[str, Dict[str, Dict[str, float]]]

METRIC_PATTERN = re.compile(r"^cosine_(?P<language>.+?)_(?P<label>(?:layer|final).*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create generic layer-wise plots from experiment results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to a JSON file produced by run_experiment (distance_map).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the resulting plot (PNG, PDF, etc.).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title. Defaults to a name derived from the results file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distance_map = _load_distance_map(args.results)
    language_stats = _group_by_language(distance_map)
    if not language_stats:
        raise SystemExit("No metrics matched the expected naming scheme.")

    default_title = f"Layer-wise Cosine Distance Trends ({args.results.stem})"
    plot_language_layer_trends(
        language_stats=language_stats,
        title=args.title or default_title,
        output_path=args.output,
    )


def _load_distance_map(path: Path) -> MetricMap:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_by_language(distance_map: MetricMap) -> LanguageStats:
    grouped: LanguageStats = {}
    for condition, metrics in distance_map.items():
        for metric_name, values in metrics.items():
            match = METRIC_PATTERN.match(metric_name)
            if not match:
                continue
            language = match.group("language")
            label = match.group("label")
            mean_value = float(np.mean(values)) if values else float("nan")
            language_dict = grouped.setdefault(language, {})
            condition_dict: MutableMapping[str, float] = language_dict.setdefault(
                condition, {}
            )
            condition_dict[label] = mean_value
    return grouped


def plot_language_layer_trends(
    *,
    language_stats: LanguageStats,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    languages = sorted(language_stats.keys())
    n_languages = len(languages)
    ncols = min(3, n_languages)
    nrows = math.ceil(n_languages / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = None

    for idx, language in enumerate(languages):
        ax = axes_flat[idx]
        condition_map = language_stats[language]
        label_order = _sorted_labels(condition_map)
        x_positions = np.arange(len(label_order))

        for condition, label_values in sorted(condition_map.items()):
            y_values = [label_values.get(label, np.nan) for label in label_order]
            ax.plot(
                x_positions,
                y_values,
                marker="o",
                label=condition,
            )

        ax.set_title(language)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_format_label(label) for label in label_order],
            rotation=40,
            ha="right",
        )
        ax.set_ylabel("Cosine Distance")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for extra_ax in axes_flat[n_languages:]:
        extra_ax.axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 3),
            frameon=False,
        )

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _sorted_labels(condition_map: Mapping[str, Mapping[str, float]]) -> Sequence[str]:
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


def _format_label(label: str) -> str:
    parts = label.split("__img_")
    text_part = _format_layer_token(parts[0])
    if len(parts) == 1:
        return text_part
    image_part = _format_layer_token(parts[1])
    return f"{text_part} / Img {image_part}"


def _format_layer_token(token: str) -> str:
    if token.startswith("layer_"):
        suffix = token.split("layer_", maxsplit=1)[1]
        return f"L{suffix}"
    if token.startswith("final"):
        return "Final"
    return token


if __name__ == "__main__":
    main()
