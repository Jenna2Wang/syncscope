"""Low-level DSP helpers (signal framing)."""

from __future__ import annotations

import numpy as np


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice ``x`` into overlapping frames of shape ``(n_frames, frame_length)``.

    Uses a strided view (copied once, so the result is safe to mutate) instead of
    a Python loop. Any tail shorter than ``frame_length`` is dropped.
    """
    x = np.asarray(x, dtype=float).ravel()
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if x.size < frame_length:
        return np.empty((0, frame_length), dtype=float)
    n_frames = 1 + (x.size - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    view = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return view.copy()
