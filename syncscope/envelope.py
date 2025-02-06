"""Audio energy and onset envelopes sampled at a fixed frame rate."""

from __future__ import annotations

import numpy as np

DEFAULT_HOP = 160  # 10 ms at 16 kHz -> a 100 Hz envelope


def audio_envelope(waveform: np.ndarray, sr: int, hop_length: int = DEFAULT_HOP) -> np.ndarray:
    """Return a frame-rate energy envelope of ``waveform``.

    TODO: implement RMS energy per frame; the envelope rate is ``sr / hop_length``.
    """
    raise NotImplementedError
