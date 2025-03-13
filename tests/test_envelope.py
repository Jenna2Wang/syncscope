import numpy as np
import pytest

from syncscope.envelope import audio_envelope, envelope_rate


def test_envelope_rate():
    assert envelope_rate(16000, hop_length=160) == 100.0


def test_envelope_tracks_loudness():
    sr = 16000
    t = np.arange(sr) / sr
    quiet = 0.01 * np.sin(2 * np.pi * 200 * t[: sr // 2])
    loud = 1.0 * np.sin(2 * np.pi * 200 * t[sr // 2 :])
    env = audio_envelope(np.concatenate([quiet, loud]), sr)
    # second half should be markedly louder than the first
    assert env[: len(env) // 2].mean() < env[len(env) // 2 :].mean()


def test_envelope_length_matches_expected_rate():
    sr = 16000
    x = np.zeros(sr)  # one second
    env = audio_envelope(x, sr, hop_length=160)
    assert abs(len(env) - 100) <= 2


def test_onset_envelope_is_nonnegative():
    sr = 16000
    rng = np.random.default_rng(0)
    env = audio_envelope(rng.standard_normal(sr), sr, kind="onset")
    assert np.all(env >= 0.0)


def test_short_signal_returns_single_frame():
    env = audio_envelope(np.ones(10), sr=16000)
    assert env.shape == (1,)


def test_empty_signal_returns_empty():
    assert audio_envelope(np.array([]), sr=16000).shape == (0,)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown envelope kind"):
        audio_envelope(np.ones(800), sr=16000, kind="bogus")
