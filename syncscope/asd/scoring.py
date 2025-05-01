"""Windowed cross-modal scoring for active-speaker detection.

The idea mirrors :func:`syncscope.sync.estimate_offset` but applied locally: slide
a short window over a face's motion and the audio envelope and measure how well
they correlate. A face that is speaking lights up; a face that is merely moving
does not.
"""

from __future__ import annotations

import numpy as np

from ..correlate import normalized_cross_correlation


def window_correlation_score(
    motion_window: np.ndarray, audio_window: np.ndarray, max_lag: int
) -> float:
    """Best correlation between a motion window and an audio window within ``+/-max_lag``."""
    if motion_window.size < 2 or audio_window.size < 2:
        return 0.0
    lags, corr = normalized_cross_correlation(audio_window, motion_window)
    keep = np.abs(lags) <= max_lag
    return float(np.max(corr[keep])) if np.any(keep) else float(np.max(corr))


def windowed_scores(
    motion: np.ndarray,
    audio_env: np.ndarray,
    rate: float,
    window_seconds: float = 0.5,
    hop_seconds: float = 0.2,
    max_offset_seconds: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over ``motion`` and ``audio_env`` (both at ``rate``) and score each.

    Returns ``(centers, scores)`` where ``centers`` are window-centre times in
    seconds and ``scores`` are the peak cross-correlation in each window.
    """
    motion = np.asarray(motion, dtype=float)
    audio_env = np.asarray(audio_env, dtype=float)
    n = min(motion.size, audio_env.size)
    motion, audio_env = motion[:n], audio_env[:n]
    win = max(2, int(round(window_seconds * rate)))
    hop = max(1, int(round(hop_seconds * rate)))
    max_lag = int(round(max_offset_seconds * rate))
    centers: list[float] = []
    scores: list[float] = []
    for start in range(0, n - win + 1, hop):
        score = window_correlation_score(
            motion[start : start + win], audio_env[start : start + win], max_lag
        )
        scores.append(score)
        centers.append((start + win / 2) / rate)
    return np.asarray(centers), np.asarray(scores)
