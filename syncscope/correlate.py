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


def parabolic_interpolation(corr: np.ndarray, peak_index: int) -> tuple[float, float]:
    """Refine a discrete peak to sub-sample resolution.

    Fits a parabola through the peak sample and its two neighbours and returns
    ``(delta, value)`` where ``delta`` is the fractional offset from
    ``peak_index`` to the true maximum (in ``[-0.5, 0.5]``) and ``value`` is the
    interpolated peak height. Falls back to the discrete peak at the edges.
    """
    if peak_index <= 0 or peak_index >= len(corr) - 1:
        return 0.0, float(corr[peak_index])
    y0 = float(corr[peak_index - 1])
    y1 = float(corr[peak_index])
    y2 = float(corr[peak_index + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return 0.0, y1
    delta = 0.5 * (y0 - y2) / denom
    value = y1 - 0.25 * (y0 - y2) * delta
    return float(delta), float(value)
