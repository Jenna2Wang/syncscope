"""Helpers for building and slicing :class:`~syncscope.types.FaceTrack` objects."""

from __future__ import annotations

import numpy as np

from ..types import FaceTrack


def tracks_from_motion(motions: list[np.ndarray], fps: float) -> list[FaceTrack]:
    """Wrap a list of per-frame motion signals into :class:`FaceTrack` objects.

    TODO: support per-track ``start_frame`` offsets for tracks that do not span
    the whole clip.
    """
    raise NotImplementedError
