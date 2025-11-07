import numpy as np

from models import embedding


def test_normalize_vectors() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 5.0]])
    normalised = embedding._normalize(vectors)
    assert np.allclose(normalised[0], np.array([0.6, 0.8]))
    assert np.allclose(normalised[1], np.array([0.0, 1.0]))
