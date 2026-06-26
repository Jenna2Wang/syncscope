"""Low-level DSP helpers (signal framing)."""

from __future__ import annotations

import numpy as np


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice ``x`` into overlapping frames of shape ``(n_frames, frame_length)``.

    TODO: implement using a strided view to avoid copies where possible.
    """
    raise NotImplementedError
