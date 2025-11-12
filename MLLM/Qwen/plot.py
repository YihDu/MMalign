#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_alignment(df, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (lang1, lang2), subdf in df.groupby(["lang1", "lang2"]):
        grouped = subdf.groupby("layer")["score"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=f"{lang1}-{lang2}")

    ax.set_ylim(0.85, 1.1)
    ax.set_xlabel("Layer")
    ax.set_ylabel(df["metric"].iloc[0].upper())
    ax.set_title(f"Cross-Lingual Alignment ({df['metric'].iloc[0].upper()})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot layer-wise cross-lingual alignment.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    plot_alignment(df, args.save_path)


if __name__ == "__main__":
    main()
