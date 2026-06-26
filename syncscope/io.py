"""Optional loaders for real audio and video.

The heavy third-party dependencies (librosa, OpenCV) are imported lazily so the
core package stays a numpy-only install. Install the relevant extra to use them::

    pip install syncscope[audio]   # librosa + soundfile
    pip install syncscope[video]   # opencv-python
"""

from __future__ import annotations

import numpy as np


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load ``path`` as a mono waveform resampled to ``sr`` Hz.

    Returns ``(waveform, sr)``. Requires the ``audio`` extra.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise ImportError(
            "audio loading needs the 'audio' extra: pip install 'syncscope[audio]'"
        ) from exc
    waveform, sr_out = librosa.load(path, sr=sr, mono=True)
    return np.asarray(waveform, dtype=float), int(sr_out)
