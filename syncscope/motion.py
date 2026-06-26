"""Per-frame visual motion signals extracted from a sequence of video frames."""

from __future__ import annotations

import numpy as np


def visual_motion(frames: np.ndarray) -> np.ndarray:
    """Return a per-frame motion magnitude for a ``(T, H, W)`` stack of frames.

    Motion is the summed absolute frame-to-frame difference. The first element is
    0 (there is no previous frame), so the output length equals the input length
    and is sampled at the video frame rate.
    """
    f = np.asarray(frames, dtype=float)
    diff = np.abs(np.diff(f, axis=0))
    per_frame = diff.reshape(diff.shape[0], -1).sum(axis=1)
    return np.concatenate([[0.0], per_frame])
