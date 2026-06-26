import numpy as np
import pytest

from syncscope.framing import frame_signal


def test_frame_shapes_and_contents():
    x = np.arange(20.0)
    frames = frame_signal(x, frame_length=5, hop_length=5)
    assert frames.shape == (4, 5)
    np.testing.assert_array_equal(frames[0], np.arange(5))


def test_overlapping_frames():
    x = np.arange(10.0)
    frames = frame_signal(x, frame_length=4, hop_length=2)
    assert frames.shape == (4, 4)  # starts at 0, 2, 4, 6
    np.testing.assert_array_equal(frames[1], np.arange(2, 6))


def test_short_signal_yields_no_frames():
    assert frame_signal(np.arange(3.0), frame_length=5, hop_length=2).shape == (0, 5)


def test_invalid_params_raise():
    with pytest.raises(ValueError, match="positive"):
        frame_signal(np.arange(10.0), frame_length=0, hop_length=1)


def test_frames_are_a_safe_copy():
    x = np.arange(10.0)
    frames = frame_signal(x, frame_length=4, hop_length=2)
    frames[0, 0] = -999.0
    assert x[0] == 0.0  # mutating the frames must not touch the source
