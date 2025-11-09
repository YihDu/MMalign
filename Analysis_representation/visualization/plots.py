"""Matplotlib helper functions for analysing experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_BOX_RC = {
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
}


def plot_distance_distributions(
    distances: Mapping[str, Sequence[float]],
    title: str,
    output_path: Path,
) -> None:
    """Save a readability-focused summary of distance distributions."""

    if not distances:
        raise ValueError("No distance data provided for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[Tuple[str, np.ndarray]] = []
    for label, values in distances.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            array = array.ravel()
        summaries.append((label, array))

    # 排序后更容易比较不同语言/条件
    summaries.sort(key=lambda item: np.nanmedian(item[1]))
    labels = [label for label, _ in summaries]
    data = [arr for _, arr in summaries]

    palette = plt.get_cmap("tab10")
    fig_height = max(4.5, 0.6 * len(labels) + 1.5)
    fig = plt.figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(1, 2, width_ratios=(3.25, 1), wspace=0.12)

    with plt.rc_context(_BOX_RC):
        ax_box = fig.add_subplot(gs[0, 0])
        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis("off")

        box = ax_box.boxplot(
            data,
            vert=False,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            widths=0.65,
        )

        for idx, patch in enumerate(box["boxes"]):
            color = palette(idx / max(1, len(labels) - 1))
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            box["medians"][idx].set_color("#333333")
            box["means"][idx].set_color("#1f77b4")
            box["means"][idx].set_linestyle("--")

        ax_box.set_title(title)
        ax_box.set_xlabel("Cosine Distance")
        ax_box.grid(axis="x", linestyle="--", alpha=0.35)
        ax_box.set_facecolor("#fbfbfd")
        for spine in ("top", "right"):
            ax_box.spines[spine].set_visible(False)

        # 在图中标注中位数，提升可读性
        for idx, arr in enumerate(data, start=1):
            median = float(np.nanmedian(arr))
            ax_box.annotate(
                f"median={median:.3f}",
                xy=(median, idx),
                xytext=(4, -4),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )

        table_data = []
        for label, arr in summaries:
            table_data.append(
                [
                    label,
                    f"{np.nanmean(arr):.3f}",
                    f"{np.nanmedian(arr):.3f}",
                    f"{np.nanstd(arr):.3f}",
                ]
            )

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Metric", "Mean", "Median", "Std"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=250)
        plt.close(fig)
