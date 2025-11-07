"""Distance and similarity metrics used in evaluation."""

from __future__ import annotations

import numpy as np


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""

    numerator = float(np.dot(vec_a, vec_b))
    denominator = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denominator == 0:
        return 1.0
    cosine_similarity = numerator / denominator
    return 1.0 - cosine_similarity


def cosine_distance_matrix(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine distances between two matrices."""

    norms_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    safe_norms_a = np.clip(norms_a, a_min=1e-12, a_max=None)
    safe_norms_b = np.clip(norms_b, a_min=1e-12, a_max=None)
    normalized_a = matrix_a / safe_norms_a
    normalized_b = matrix_b / safe_norms_b
    cosine_similarity = normalized_a @ normalized_b.T
    return 1.0 - cosine_similarity
