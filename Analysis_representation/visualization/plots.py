"""Matplotlib helper functions for analysing experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_distance_distributions(
    distances: Mapping[str, Sequence[float]],
    title: str,
    output_path: Path,
) -> None:
    """Save a boxplot comparing distance distributions for each language."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(distances.keys())
    data = [np.asarray(values) for values in distances.values()]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel("Cosine Distance")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
