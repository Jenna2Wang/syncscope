import numpy as np
import pytest

from syncscope.windowing import window_centers, window_starts


def test_window_starts_basic():
    starts = window_starts(100, window=20, hop=10)
    assert starts[0] == 0
    assert starts[-1] + 20 <= 100
    assert np.all(np.diff(starts) == 10)


def test_window_starts_short_signal_is_empty():
    assert window_starts(5, window=20, hop=10).size == 0


def test_window_centers_in_seconds():
    centers = window_centers(100, window=20, hop=10, rate=100.0)
    np.testing.assert_allclose(centers[0], 0.1)  # (0 + 20/2) / 100


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="positive"):
        window_starts(100, window=0, hop=10)
