"""Statistical significance tests for comparing condition outcomes."""

from __future__ import annotations

from typing import Mapping

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None  # type: ignore


def t_test_independent(
    group_a: np.ndarray,
    group_b: np.ndarray,
    equal_var: bool = False,
) -> Mapping[str, float]:
    """Perform an independent t-test between two groups."""

    if stats is None:
        raise ImportError(
            "SciPy is required for statistical testing. Install via `pip install scipy`."
        )
    result = stats.ttest_ind(group_a, group_b, equal_var=equal_var, nan_policy="omit")
    return {"t_stat": float(result.statistic), "p_value": float(result.pvalue)}
