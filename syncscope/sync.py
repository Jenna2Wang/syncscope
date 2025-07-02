"""Estimate the temporal offset between an audio envelope and visual motion."""

from __future__ import annotations

import numpy as np

from .correlate import normalized_cross_correlation, parabolic_interpolation
from .types import SyncResult


def estimate_offset(
    reference: np.ndarray,
    target: np.ndarray,
    rate: float,
    max_offset_seconds: float | None = None,
) -> SyncResult:
    """Estimate the offset (seconds) that best aligns ``target`` onto ``reference``.

    Both signals must already be sampled at ``rate`` Hz. A **positive**
    ``offset_seconds`` in the result means ``target`` lags ``reference`` (it is
    later in time and would need to be advanced to line up). When ``reference`` is
    the audio envelope and ``target`` the visual motion, a positive value means
    the sound runs ahead of the picture.

    ``max_offset_seconds`` restricts the lag search to ``+/- max_offset_seconds``,
    which both speeds things up and avoids spurious matches at implausible shifts.
    """
    lags, corr = normalized_cross_correlation(reference, target)
    if max_offset_seconds is not None:
        max_lag = int(round(max_offset_seconds * rate))
        # Never search beyond the lag range the signals actually support, otherwise
        # we would key off correlations backed by only a handful of samples.
        max_lag = min(max_lag, int(lags.max()))
        keep = np.abs(lags) <= max_lag
        lags, corr = lags[keep], corr[keep]
    peak = int(np.argmax(corr))
    delta, value = parabolic_interpolation(corr, peak)
    # Sub-sample lag: the lag axis is unit-spaced, so the interpolation offset in
    # samples is just ``delta``. Negate to follow the result's sign convention.
    offset_samples = -(float(lags[peak]) + delta)
    # Confidence is how far the peak stands out above the median correlation -- a
    # sharp, isolated peak is trustworthy, a noisy flat surface is not.
    confidence = float(value - np.median(corr))
    return SyncResult(
        offset_seconds=offset_samples / rate,
        offset_samples=offset_samples,
        confidence=confidence,
        peak_correlation=float(value),
        rate=rate,
    )


def align_signals(
    a: np.ndarray, b: np.ndarray, offset_samples: float
) -> tuple[np.ndarray, np.ndarray]:
    """Trim ``a`` and ``b`` so they overlap after removing an integer offset.

    ``offset_samples`` follows the :class:`SyncResult` convention (positive means
    ``b`` lags ``a``). The returned arrays have equal length and are aligned in
    time, ready to be overlaid or scored.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = int(round(offset_samples))
    if k > 0:
        b = b[k:]
    elif k < 0:
        a = a[-k:]
    n = min(a.size, b.size)
    return a[:n], b[:n]


def estimate_offset_track(
    reference: np.ndarray,
    target: np.ndarray,
    rate: float,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    max_offset_seconds: float | None = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate how the offset drifts across a clip.

    Runs :func:`estimate_offset` on overlapping windows and returns
    ``(centers, offsets, confidences)`` with one entry per window. Useful for
    spotting clips whose sync wanders, e.g. variable-frame-rate captures or
    concatenated takes.
    """
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(reference.size, target.size)
    win = max(2, int(round(window_seconds * rate)))
    hop = max(1, int(round(hop_seconds * rate)))
    centers: list[float] = []
    offsets: list[float] = []
    confidences: list[float] = []
    for start in range(0, max(1, n - win + 1), hop):
        end = min(n, start + win)
        res = estimate_offset(
            reference[start:end], target[start:end], rate, max_offset_seconds=max_offset_seconds
        )
        centers.append((start + (end - start) / 2) / rate)
        offsets.append(res.offset_seconds)
        confidences.append(res.confidence)
    return np.asarray(centers), np.asarray(offsets), np.asarray(confidences)
