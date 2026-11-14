"""Tiny helpers used by the dashboard (kept separate so the app does not import the training code)."""
import numpy as np


def sen_slope_per_decade(values: np.ndarray) -> float:
    """Sen's slope (median of all pairwise slopes) of an annual series, in units per decade."""
    v = np.asarray(values, dtype=float)
    i, j = np.triu_indices(len(v), 1)
    return float(np.median((v[j] - v[i]) / (j - i)) * 10)
