"""Core data structures shared across :mod:`syncscope`.

These are intentionally tiny, dependency-light containers. Anything that needs
to cross a module boundary (a sync estimate, a face track, a detected speaking
segment) is represented here so the rest of the package can stay functional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Offsets below this are generally imperceptible to a viewer. ITU-R BT.1359
# puts the comfortable range at roughly -125 ms (audio late) to +45 ms (audio
# early); we use a symmetric, conservative threshold for the ``in_sync`` flag.
PERCEPTIBLE_OFFSET_MS = 40.0


@dataclass(frozen=True)
class SyncResult:
    """Result of estimating the audio-visual offset of a clip.

    A positive ``offset_seconds`` means the audio envelope has to be delayed to
    line up with the visual motion (i.e. the sound currently runs *ahead* of the
    picture).
    """

    offset_seconds: float
    offset_samples: float
    confidence: float
    peak_correlation: float
    rate: float

    @property
    def offset_ms(self) -> float:
        return self.offset_seconds * 1000.0

    @property
    def in_sync(self) -> bool:
        return abs(self.offset_ms) <= PERCEPTIBLE_OFFSET_MS


@dataclass
class FaceTrack:
    """A single face followed over time, with a per-frame motion signal.

    ``motion`` is sampled at ``fps`` and usually comes from
    :func:`syncscope.motion.visual_motion` applied to the mouth region.
    """

    track_id: int
    motion: np.ndarray
    fps: float
    start_frame: int = 0

    def __post_init__(self) -> None:
        self.motion = np.asarray(self.motion, dtype=float)

    @property
    def duration(self) -> float:
        return len(self.motion) / self.fps

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.motion)


@dataclass(frozen=True)
class SpeakerSegment:
    """A span of time during which a particular track is judged to be speaking."""

    track_id: int
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start
