import numpy as np

from syncscope.synthetic import delayed_pair, speaker_scene, speech_like_envelope
from syncscope.types import FaceTrack


def test_envelope_shape_and_nonnegative():
    env = speech_like_envelope(duration=3.0, rate=100.0, seed=0)
    assert env.shape == (300,)
    assert np.all(env >= 0.0)


def test_envelope_is_deterministic():
    a = speech_like_envelope(2.0, 100.0, seed=42)
    b = speech_like_envelope(2.0, 100.0, seed=42)
    np.testing.assert_array_equal(a, b)


def test_delayed_pair_same_length():
    ref, target = delayed_pair(duration=4.0, rate=100.0, offset_seconds=0.1, seed=0)
    assert ref.shape == target.shape == (400,)


def test_speaker_scene_shapes():
    audio_env, tracks, active = speaker_scene(duration=5.0, n_faces=4, active=2, seed=0)
    assert len(tracks) == 4
    assert all(isinstance(t, FaceTrack) for t in tracks)
    assert audio_env.shape == (500,)
    assert active == 2
