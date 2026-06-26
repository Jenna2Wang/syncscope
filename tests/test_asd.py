import numpy as np

from syncscope.asd import ActiveSpeakerDetector, tracks_from_motion
from syncscope.synthetic import speaker_scene


def test_picks_the_talking_face():
    audio_env, tracks, active = speaker_scene(
        duration=6.0, n_faces=3, active=1, seed=3
    )
    det = ActiveSpeakerDetector(threshold=0.3)
    segments = det.detect(tracks, audio_env, audio_rate=100.0)
    assert segments, "expected at least one speaking segment"
    # the great majority of speaking time should be attributed to face 1
    spoken = {}
    for seg in segments:
        spoken[seg.track_id] = spoken.get(seg.track_id, 0.0) + seg.duration
    assert max(spoken, key=spoken.get) == active


def test_silent_scene_yields_no_speaker():
    rng = np.random.default_rng(0)
    audio_env = np.zeros(600)  # silence
    tracks = tracks_from_motion([rng.random(150) for _ in range(2)], fps=25.0)
    det = ActiveSpeakerDetector(threshold=0.5)
    segments = det.detect(tracks, audio_env, audio_rate=100.0)
    assert segments == []


def test_no_tracks_returns_empty():
    det = ActiveSpeakerDetector()
    assert det.detect([], np.ones(100), audio_rate=100.0) == []
