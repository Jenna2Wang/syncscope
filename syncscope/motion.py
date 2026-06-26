"""Per-frame visual motion signals extracted from a sequence of video frames."""

from __future__ import annotations

import numpy as np

# A region of interest as (top, bottom, left, right) pixel bounds.
Roi = tuple[int, int, int, int]

# Rec. 601 luma coefficients for collapsing RGB frames to a single channel.
_LUMA = np.array([0.299, 0.587, 0.114])


def to_gray(frames: np.ndarray) -> np.ndarray:
    """Collapse ``(T, H, W, C)`` frames to ``(T, H, W)``; pass 3-D input through."""
    f = np.asarray(frames, dtype=float)
    if f.ndim == 4:
        f = f @ _LUMA if f.shape[-1] == 3 else f.mean(axis=-1)
    if f.ndim != 3:
        raise ValueError("frames must have shape (T, H, W) or (T, H, W, C)")
    return f


def visual_motion(frames: np.ndarray, roi: Roi | None = None) -> np.ndarray:
    """Return a per-frame motion magnitude for a ``(T, H, W[, C])`` stack of frames.

    Motion is the summed absolute frame-to-frame difference over the (optional)
    region of interest ``roi = (top, bottom, left, right)``. Restricting to the
    mouth region is what turns this into a useful active-speaker cue. The first
    element is 0 (there is no previous frame), so the output length equals the
    input length and is sampled at the video frame rate.
    """
    f = to_gray(frames)
    if roi is not None:
        top, bottom, left, right = roi
        f = f[:, top:bottom, left:right]
    diff = np.abs(np.diff(f, axis=0))
    per_frame = diff.reshape(diff.shape[0], -1).sum(axis=1)
    return np.concatenate([[0.0], per_frame])
