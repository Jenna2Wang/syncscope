"""Cross-correlation utilities used to align two 1-D signals in time."""

from __future__ import annotations

import numpy as np


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lags, corr)`` for the normalized cross-correlation of ``a`` and ``b``.

    TODO: implement. Will return the per-lag Pearson-style correlation so the
    peak location gives the best integer alignment between the two signals.
    """
    raise NotImplementedError
