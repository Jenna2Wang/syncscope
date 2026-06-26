"""Audio energy and onset envelopes sampled at a fixed frame rate."""

from __future__ import annotations

import numpy as np

from .framing import frame_signal

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
    kind: str = "rms",
) -> np.ndarray:
    """Return a frame-rate envelope of ``waveform``.

    The envelope is sampled at ``sr / hop_length`` Hz (see :func:`envelope_rate`).
    ``kind="rms"`` gives per-frame root-mean-square energy; ``kind="onset"`` gives
    a half-wave-rectified first difference of the log-energy, which emphasises the
    onsets of speech and tends to line up more crisply with visible articulation.
    """
    x = np.asarray(waveform, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=float)
    frames = frame_signal(x, frame_length, hop_length)
    if frames.shape[0] == 0:
        # Clip shorter than a single frame: fall back to one frame over all of it.
        frames = x[np.newaxis, :]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    if kind == "rms":
        return rms
    if kind == "onset":
        log_energy = np.log(rms + 1e-8)
        flux = np.diff(log_energy, prepend=log_energy[:1])
        return np.maximum(flux, 0.0)
    raise ValueError(f"unknown envelope kind: {kind!r}")
