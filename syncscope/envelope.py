"""Audio energy and onset envelopes sampled at a fixed frame rate."""

from __future__ import annotations

import numpy as np

from .dsp import frame_signal

DEFAULT_HOP = 160  # 10 ms at 16 kHz -> a 100 Hz envelope
DEFAULT_FRAME = 400  # 25 ms at 16 kHz


def envelope_rate(sr: int, hop_length: int = DEFAULT_HOP) -> float:
    """Frame rate, in Hz, of an envelope computed with ``hop_length``."""
    return sr / hop_length


def audio_envelope(
    waveform: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP,
    frame_length: int = DEFAULT_FRAME,
) -> np.ndarray:
    """Return a frame-rate RMS energy envelope of ``waveform``.

    The envelope is sampled at ``sr / hop_length`` Hz (see :func:`envelope_rate`).
    """
    x = np.asarray(waveform, dtype=float).ravel()
    frames = frame_signal(x, frame_length, hop_length)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
