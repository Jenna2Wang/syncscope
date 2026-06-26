"""High-level pipelines that wire envelope + motion + sync/asd together."""

from __future__ import annotations

import numpy as np

from .envelope import audio_envelope, envelope_rate
from .motion import visual_motion
from .resample import resample_signal, to_common_rate
from .sync import estimate_offset
from .types import SyncResult


def sync_streams(
    waveform: np.ndarray,
    sr: int,
    frames: np.ndarray,
    fps: float,
    max_offset_seconds: float = 0.5,
) -> SyncResult:
    """Estimate the audio-visual offset between a waveform and a stack of frames.

    Convenience wrapper that builds the audio envelope and the whole-frame motion
    signal, brings them onto a common rate, and runs :func:`estimate_offset`.
    """
    env = audio_envelope(waveform, sr)
    audio_rate = envelope_rate(sr)
    motion = visual_motion(frames)
    audio_c, motion_c, rate = to_common_rate(env, audio_rate, motion, fps)
    return estimate_offset(audio_c, motion_c, rate, max_offset_seconds=max_offset_seconds)
