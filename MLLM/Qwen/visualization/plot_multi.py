#!/usr/bin/env python3
import os
import pandas as pd
import argparse
from pathlib import Path

os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
os.makedirs('/tmp/mpl_cache', exist_ok=True)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_metric_for_models(model_dirs, labels, metric, save_path):
    """
    For a given metric (e.g., 'cka'), load that metric from each model directory,
    and plot them in one figure.
    """

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#CA3519", "#E85D45", "#F2A497", "#C71585", "#FFD700", "#9370DB"]

    # ==== 自动设置 y 轴标签 ====
    ylabels = {
        "cka": "CKA",
        "cosine": "Cosine Similarity",
        "cosine_norm": "Cosine Similarity (Normalized)",
    }
    ylabel = ylabels.get(metric.lower(), metric.upper()) 

    # ====================================================

    for idx, (model_dir, label) in enumerate(zip(model_dirs, labels)):
        metric_file = Path(model_dir) / f"results_{metric}.csv"
        if not metric_file.exists():
            print(f"[WARN] {metric_file} not found, skip this model.")
            continue

        df = pd.read_csv(metric_file)

        grouped = df.groupby("layer")["score"]
        mean_scores = grouped.mean()
        std_scores = grouped.std()

        layers = mean_scores.index.values.astype(float)
        norm_layers = layers / layers.max()

        mean_vals = mean_scores.values
        std_vals = std_scores.values

        color = colors[idx % len(colors)]

        ax.plot(
            norm_layers,
            mean_vals,
            linestyle="--",
            marker='.',
            markersize=4,
            color=color,
            label=label
        )

        ax.fill_between(
            norm_layers,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.1,
        )

    # ---- Axis settings ----
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.2, 1.2)
    ax.set_xlabel("Layer (Normalized)", fontfamily='Arial')
    ax.set_ylabel(ylabel, fontfamily='Arial')   # ⭐ 使用自动 y 轴标签

    ax.set_title(f"PM4Bench-MDUR ({metric.upper()})", fontfamily='Arial')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
    prop={"family": "Arial"},
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot metrics for multiple models")
    parser.add_argument("--model_dirs", nargs="+", required=True, help="Model directories")
    parser.add_argument("--labels", nargs="+", required=True, help="Label for each model")
    parser.add_argument("--metrics", nargs="+", required=True, help="Metrics to plot")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if len(args.model_dirs) != len(args.labels):
        raise ValueError("Number of model_dirs must match number of labels")

    for metric in args.metrics:
        save_path = f"{args.output_dir}/results_{metric}.png"
        plot_metric_for_models(args.model_dirs, args.labels, metric, save_path)


if __name__ == "__main__":
    main()
