"""Cross-correlation utilities used to align two 1-D signals in time."""

from __future__ import annotations

import numpy as np


def _as_centered(x: np.ndarray) -> np.ndarray:
    """Flatten to 1-D float and remove the mean."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr - arr.mean()


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lags, corr)`` for the normalized cross-correlation of ``a`` and ``b``.

    The signals are mean-removed and divided by the product of their norms, so
    ``corr`` lies in ``[-1, 1]`` and ``corr[k]`` is the correlation at ``lags[k]``.
    """
    ca = _as_centered(a)
    cb = _as_centered(b)
    full = np.correlate(ca, cb, mode="full")
    denom = np.sqrt(np.sum(ca**2) * np.sum(cb**2))
    full = full / denom
    lags = np.arange(-(cb.size - 1), ca.size)
    return lags, full
