"""Estimate the temporal offset between an audio envelope and visual motion."""

from __future__ import annotations

import numpy as np

from .correlate import normalized_cross_correlation
from .types import SyncResult


def estimate_offset(reference: np.ndarray, target: np.ndarray, rate: float) -> SyncResult:
    """Estimate the offset (seconds) that best aligns ``target`` onto ``reference``.

    Both signals must already be sampled at ``rate`` Hz. A **positive**
    ``offset_seconds`` in the result means ``target`` lags ``reference`` (it is
    later in time and would need to be advanced to line up). When ``reference`` is
    the audio envelope and ``target`` the visual motion, a positive value means
    the sound runs ahead of the picture.
    """
    lags, corr = normalized_cross_correlation(reference, target)
    peak = int(np.argmax(corr))
    offset_samples = -float(lags[peak])
    peak_corr = float(corr[peak])
    return SyncResult(
        offset_seconds=offset_samples / rate,
        offset_samples=offset_samples,
        confidence=peak_corr,
        peak_correlation=peak_corr,
        rate=rate,
    )
