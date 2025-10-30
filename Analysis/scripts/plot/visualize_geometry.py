import os, json
import numpy as np
import matplotlib.pyplot as plt
from utils.config_loader import load_config


def load_summary(summary_path):
    with open(summary_path, "r") as f:
        return json.load(f)


def plot_metric(ax, steps, values, title, ylabel, color):
    ax.plot(steps, values, marker="o", color=color, linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Step / Checkpoint")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)


def main(config_path: str):
    cfg = load_config(config_path)
    metric_dir = os.path.join(cfg.paths.output_dir, "metrics")
    summary_path = os.path.join(metric_dir, "summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = load_summary(summary_path)
    if len(summary) == 0:
        print("⚠️ No metrics found in summary.json")
        return

    # 按 step 排序
    def extract_step(ckpt_name):
        import re
        digits = re.findall(r"\d+", ckpt_name)
        return int(digits[-1]) if digits else 0

    summary.sort(key=lambda x: extract_step(x["ckpt"]))
    steps = [extract_step(s["ckpt"]) for s in summary]
    rankme = [s["rankme"] for s in summary]
    alpha = [s["alpha_req"] for s in summary]

    # 绘图
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_metric(ax[0], steps, rankme, "RankMe Evolution", "RankMe ↑", "tab:blue")
    plot_metric(ax[1], steps, alpha, "αReQ Evolution", "αReQ ↓", "tab:orange")

    fig.suptitle(f"Spectral Geometry Evolution ({cfg.model.target_layer})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(metric_dir, "geometry_evolution.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Saved plot to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
