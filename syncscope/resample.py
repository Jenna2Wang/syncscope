"""Resample 1-D signals so the audio envelope and visual motion share a rate."""

from __future__ import annotations

import numpy as np


def resample_signal(x: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    """Resample ``x`` from ``src_rate`` to ``dst_rate`` Hz by linear interpolation."""
    x = np.asarray(x, dtype=float).ravel()
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("rates must be positive")
    if x.size == 0 or src_rate == dst_rate:
        return x.copy()
    duration = x.size / src_rate
    n_dst = max(1, int(round(duration * dst_rate)))
    src_t = np.arange(x.size) / src_rate
    dst_t = np.arange(n_dst) / dst_rate
    return np.interp(dst_t, src_t, x)


def to_common_rate(
    audio_env: np.ndarray,
    audio_rate: float,
    motion: np.ndarray,
    motion_rate: float,
    target_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bring ``audio_env`` and ``motion`` onto a single rate and equal length.

    Returns ``(audio_resampled, motion_resampled, target_rate)``. When
    ``target_rate`` is ``None`` the faster of the two input rates is used so no
    detail is thrown away. The signals are truncated to the shorter length so they
    can be cross-correlated directly.
    """
    target = target_rate if target_rate is not None else max(audio_rate, motion_rate)
    audio_resampled = resample_signal(audio_env, audio_rate, target)
    motion_resampled = resample_signal(motion, motion_rate, target)
    n = min(audio_resampled.size, motion_resampled.size)
    return audio_resampled[:n], motion_resampled[:n], target
