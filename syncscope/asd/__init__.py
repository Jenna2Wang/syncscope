"""Active-speaker detection built on cross-modal correlation."""

from __future__ import annotations

from .detector import ActiveSpeakerDetector
from .scoring import window_correlation_score, windowed_scores
from .track import tracks_from_motion

__all__ = [
    "ActiveSpeakerDetector",
    "tracks_from_motion",
    "window_correlation_score",
    "windowed_scores",
]
