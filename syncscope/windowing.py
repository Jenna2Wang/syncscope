"""Sliding-window helpers shared by the sync and ASD code paths."""

from __future__ import annotations

import numpy as np


def window_starts(n: int, window: int, hop: int) -> np.ndarray:
    """Start indices of windows of length ``window`` stepped by ``hop`` over ``n`` samples.

    Returns an empty array when the signal is shorter than a single window.
    """
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    if n < window:
        return np.zeros(0, dtype=int)
    return np.arange(0, n - window + 1, hop)


def window_centers(n: int, window: int, hop: int, rate: float) -> np.ndarray:
    """Centre time, in seconds, of each window produced by :func:`window_starts`."""
    return (window_starts(n, window, hop) + window / 2) / rate
