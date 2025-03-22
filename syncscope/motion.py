"""Per-frame visual motion signals extracted from a sequence of video frames."""

from __future__ import annotations

import numpy as np

# A region of interest as (top, bottom, left, right) pixel bounds.
Roi = tuple[int, int, int, int]


def visual_motion(frames: np.ndarray, roi: Roi | None = None) -> np.ndarray:
    """Return a per-frame motion magnitude for a ``(T, H, W)`` stack of frames.

    Motion is the summed absolute frame-to-frame difference over the (optional)
    region of interest ``roi = (top, bottom, left, right)``. Restricting to the
    mouth region is what turns this into a useful active-speaker cue. The first
    element is 0 (there is no previous frame), so the output length equals the
    input length and is sampled at the video frame rate.
    """
    f = np.asarray(frames, dtype=float)
    if roi is not None:
        top, bottom, left, right = roi
        f = f[:, top:bottom, left:right]
    diff = np.abs(np.diff(f, axis=0))
    per_frame = diff.reshape(diff.shape[0], -1).sum(axis=1)
    return np.concatenate([[0.0], per_frame])
