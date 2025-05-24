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


def load_video_frames(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Load ``path`` as a stack of grayscale frames plus its frame rate.

    Returns ``(frames, fps)`` with ``frames`` shaped ``(T, H, W)``. Requires the
    ``video`` extra. ``max_frames`` caps how many frames are read.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise ImportError(
            "video loading needs the 'video' extra: pip install 'syncscope[video]'"
        ) from exc
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    return np.asarray(frames, dtype=float), float(fps)
