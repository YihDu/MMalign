# -*- coding: utf-8 -*-

import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scripts.utils.config_loader import load_config


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_step(ckpt_name):
    """
    从 checkpoint 名中提取 step 数值，例如 checkpoint-700 -> 700
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
        raise FileNotFoundError(f"❌ 未找到 summary.json: {summary_path}")

    summary = load_summary(summary_path)
    if len(summary) == 0:
        print("⚠️ summary.json 为空，无法绘制谱指标。")
        return

    # === 按 layer 分组 ===
    layer_groups = {}
    for entry in summary:
        layer = entry.get("layer", None)
        if layer is None:
            continue
        layer_groups.setdefault(layer, []).append(entry)

    if not layer_groups:
        print("⚠️ 未检测到 layer 字段，将绘制整体平均趋势。")
        layer_groups = {"all": summary}

    # === 创建画布 ===
    fig, axes = plt.subplots(1, 3, figsize=(50, 40))
    colors = plt.cm.tab10.colors

    # === 遍历每个 layer 绘制 ===
    for i, (layer, entries) in enumerate(sorted(layer_groups.items())):
        entries.sort(key=lambda x: extract_step(x["ckpt"]))
        steps = [extract_step(e["ckpt"]) for e in entries]
        rankme = [e.get("rankme", np.nan) for e in entries]
        alpha = [e.get("alpha_req", np.nan) for e in entries]
        erank = [e.get("effective_rank", np.nan) for e in entries]

        axes[0].plot(steps, rankme, marker="o", linewidth=2,
                     color=colors[i % len(colors)], label=f"Layer {layer}")
        axes[1].plot(steps, alpha, marker="o", linewidth=2,
                     color=colors[i % len(colors)], label=f"Layer {layer}")
        axes[2].plot(steps, erank, marker="o", linewidth=2,
                     color=colors[i % len(colors)], label=f"Layer {layer}")

    # === 图像格式设置 ===
    titles = ["RankMe 演化", "αReQ 演化", "Effective Rank 演化"]
    ylabels = ["RankMe ↑", "αReQ ↓", "Effective Rank ↑"]

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Step / Checkpoint")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        

    # === 总标题 ===
    layers_str = ", ".join(map(str, sorted(layer_groups.keys())))
    fig.suptitle(f"Spectral Geometry Evolution across Layers ({layers_str})",
                 fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # === 保存结果 ===
    out_path = os.path.join(metric_dir, "geometry_evolution_multilayer.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"✅ 已保存谱指标曲线（多层对比版）: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize spectral geometry evolution across layers.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    main(args.config)