"""Per-frame visual motion signals extracted from a sequence of video frames."""

from __future__ import annotations

import numpy as np


def visual_motion(frames: np.ndarray) -> np.ndarray:
    """Return a per-frame motion magnitude for a ``(T, H, W[, C])`` stack of frames.

    TODO: implement frame-difference motion; the result is sampled at the video
    frame rate and is what we correlate against the audio envelope.
    """
    raise NotImplementedError
