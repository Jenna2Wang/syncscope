"""The active-speaker detector: attribute speech to the most correlated face."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..resample import resample_signal
from ..types import FaceTrack, SpeakerSegment
from .scoring import windowed_scores


@dataclass
class ActiveSpeakerDetector:
    """Decide which face is speaking over time from per-face motion and the audio.

    Each face's motion is resampled onto the audio-envelope rate, scored against
    the audio in short sliding windows, and the highest-scoring face above
    ``threshold`` is taken to be the active speaker in that window.
    """

    window_seconds: float = 0.5
    hop_seconds: float = 0.2
    threshold: float = 0.3
    max_offset_seconds: float = 0.2

    def score_tracks(
        self, tracks: list[FaceTrack], audio_env: np.ndarray, audio_rate: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(centers, scores)`` where ``scores[i]`` are window scores for track ``i``."""
        audio = np.asarray(audio_env, dtype=float)
        centers: np.ndarray | None = None
        per_track: list[np.ndarray] = []
        for track in tracks:
            motion = resample_signal(track.motion, track.fps, audio_rate)
            centers, scores = windowed_scores(
                motion,
                audio,
                audio_rate,
                self.window_seconds,
                self.hop_seconds,
                self.max_offset_seconds,
            )
            per_track.append(scores)
        if centers is None or centers.size == 0:
            return np.zeros(0), np.zeros((0, 0))
        width = min(len(s) for s in per_track)
        matrix = np.stack([s[:width] for s in per_track], axis=0)
        return centers[:width], matrix

    def detect(
        self, tracks: list[FaceTrack], audio_env: np.ndarray, audio_rate: float
    ) -> list[SpeakerSegment]:
        """Return one :class:`SpeakerSegment` per active window (unmerged for now)."""
        if not tracks:
            return []
        centers, matrix = self.score_tracks(tracks, audio_env, audio_rate)
        if centers.size == 0:
            return []
        track_ids = [t.track_id for t in tracks]
        best_idx = np.argmax(matrix, axis=0)
        best_score = matrix[best_idx, np.arange(matrix.shape[1])]
        half = self.hop_seconds / 2.0
        segments: list[SpeakerSegment] = []
        for i, center in enumerate(centers):
            if best_score[i] >= self.threshold:
                segments.append(
                    SpeakerSegment(
                        track_id=track_ids[int(best_idx[i])],
                        start=float(center - half),
                        end=float(center + half),
                        score=float(best_score[i]),
                    )
                )
        return segments
