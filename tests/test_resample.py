import numpy as np
import pytest

from syncscope.resample import resample_signal, to_common_rate


def test_identity_when_rates_match():
    x = np.arange(10.0)
    out = resample_signal(x, 100, 100)
    np.testing.assert_array_equal(out, x)
    assert out is not x  # returns a copy


def test_upsampling_changes_length():
    x = np.arange(10.0)
    out = resample_signal(x, 10, 20)
    assert len(out) == 20


def test_downsampling_changes_length():
    x = np.arange(100.0)
    out = resample_signal(x, 100, 25)
    assert len(out) == 25


def test_empty_input_returns_empty():
    assert resample_signal(np.array([]), 10, 20).shape == (0,)


def test_nonpositive_rate_raises():
    with pytest.raises(ValueError, match="positive"):
        resample_signal(np.arange(5.0), 0, 10)


def test_to_common_rate_equal_length_and_target():
    audio = np.arange(100.0)  # 100 Hz, 1 s
    motion = np.arange(25.0)  # 25 Hz, 1 s
    a, m, rate = to_common_rate(audio, 100, motion, 25)
    assert rate == 100
    assert len(a) == len(m)
