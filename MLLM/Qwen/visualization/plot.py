#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import argparse
from pathlib import Path

os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
os.makedirs('/tmp/mpl_cache', exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'stix'

def plot_alignment(df, save_path=None):
    
    plt.rcParams['font.family'] = 'Arial' 
    plt.rcParams['font.size'] = 14 
    fig, ax = plt.subplots(figsize=(8, 5))

    # ---- 1️⃣ 按 layer 求全体语言 pair 平均与标准差 ----
    grouped = df.groupby("layer")["score"]
    mean_scores = grouped.mean()
    std_scores = grouped.std()

    layers = mean_scores.index.values
    mean_vals = mean_scores.values
    std_vals = std_scores.values

    # ---- 2️⃣ 画平均曲线 ----
    ax.plot(layers, mean_vals, linestyle="--",marker = 'o',markersize=4, color="#51A2FF",label = 'Qwen2.5-VL-7B')

    # ---- 3️⃣ 画标准差阴影 ----
    ax.fill_between(
        layers,
        mean_vals - std_vals,
        mean_vals + std_vals,
        color="#51A2FF",
        alpha=0.1,
    )

    # ---- 4️⃣ 图形设置 ----
    ax.set_ylim(0.2, 1.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(df["metric"].iloc[0].upper())
    ax.set_title(f"PM4Bench-MDUR Task")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # ---- 5️⃣ 保存或展示 ----
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="PM4Bench MDUR Task")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    plot_alignment(df, args.save_path)


if __name__ == "__main__":
    main()


# {df['metric'].iloc[0].upper()})