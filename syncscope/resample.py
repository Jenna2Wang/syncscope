"""Resample 1-D signals so the audio envelope and visual motion share a rate."""

from __future__ import annotations

import numpy as np


def resample_signal(x: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    """Resample ``x`` from ``src_rate`` to ``dst_rate`` Hz by linear interpolation."""
    x = np.asarray(x, dtype=float).ravel()
    duration = x.size / src_rate
    n_dst = max(1, int(round(duration * dst_rate)))
    src_t = np.arange(x.size) / src_rate
    dst_t = np.arange(n_dst) / dst_rate
    return np.interp(dst_t, src_t, x)
