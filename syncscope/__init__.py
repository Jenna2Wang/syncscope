"""syncscope: audio-visual synchronization and active-speaker detection."""

from __future__ import annotations

from ._version import __version__
from .asd import ActiveSpeakerDetector, tracks_from_motion
from .correlate import normalized_cross_correlation
from .envelope import audio_envelope, envelope_rate
from .metrics import average_precision, offset_error, within_tolerance
from .motion import visual_motion
from .pipeline import detect_speakers, sync_streams
from .resample import resample_signal, to_common_rate
from .sync import align_signals, estimate_offset
from .types import FaceTrack, SpeakerSegment, SyncResult

__all__ = [
    "ActiveSpeakerDetector",
    "FaceTrack",
    "SpeakerSegment",
    "SyncResult",
    "__version__",
    "align_signals",
    "audio_envelope",
    "average_precision",
    "detect_speakers",
    "envelope_rate",
    "estimate_offset",
    "normalized_cross_correlation",
    "offset_error",
    "resample_signal",
    "sync_streams",
    "to_common_rate",
    "tracks_from_motion",
    "visual_motion",
    "within_tolerance",
]
