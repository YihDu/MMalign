# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.config_loader import load_config


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric(ax, steps, values, title, ylabel, color):
    """
    绘制单个谱指标的演化曲线。
    TODO: 支持置信区间或多曲线对比。
    """
    ax.plot(steps, values, marker="o", color=color, linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step / Checkpoint")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)


def extract_step(ckpt_name):
    """
    从 checkpoint 命名中提取数值步数，默认取最后一段数字。
    """
    import re
    digits = re.findall(r"\d+", ckpt_name)
    return int(digits[-1]) if digits else 0


def main(config_path: str):
    cfg = load_config(config_path)
    feature_dir = cfg.paths.activation_save_dir
    metric_dir = os.path.join(feature_dir, "metrics")
    summary_path = os.path.join(metric_dir, "summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"汇总文件不存在: {summary_path}")

    summary = load_summary(summary_path)
    if len(summary) == 0:
        print("⚠️ summary.json 为空，无法绘制谱指标。")
        return

    summary.sort(key=lambda x: extract_step(x["ckpt"]))
    steps = [extract_step(s["ckpt"]) for s in summary]

    rankme = [s.get("rankme", np.nan) for s in summary]
    alpha = [s.get("alpha_req", np.nan) for s in summary]
    erank = [s.get("effective_rank", np.nan) for s in summary]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    plot_metric(ax[0], steps, rankme, "RankMe 演化", "RankMe ↑", "tab:blue")
    plot_metric(ax[1], steps, alpha, "αReQ 演化", "αReQ ↓", "tab:orange")
    plot_metric(ax[2], steps, erank, "Effective Rank 演化", "Effective Rank ↑", "tab:green")

    layer_set = sorted({entry.get("layer") for entry in summary if entry.get("layer") is not None})
    if len(layer_set) == 1:
        layer_caption = f"layer={layer_set[0]}"
    else:
        layer_caption = f"layers={layer_set[:3]}{'...' if len(layer_set) > 3 else ''}"
    fig.suptitle(f"Spectral Geometry Evolution ({layer_caption})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(metric_dir, "geometry_evolution.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ 已保存谱指标曲线: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
