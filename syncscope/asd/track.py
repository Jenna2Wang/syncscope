"""Helpers for building and slicing :class:`~syncscope.types.FaceTrack` objects."""

from __future__ import annotations

import numpy as np

from ..types import FaceTrack


def tracks_from_motion(
    motions: list[np.ndarray], fps: float, start_frame: int = 0
) -> list[FaceTrack]:
    """Wrap a list of per-frame motion signals into :class:`FaceTrack` objects.

    Track ids are assigned in order. ``start_frame`` is applied to every track,
    which is the common case where all faces are tracked over the same window.
    """
    return [
        FaceTrack(track_id=i, motion=np.asarray(m, dtype=float), fps=fps, start_frame=start_frame)
        for i, m in enumerate(motions)
    ]
