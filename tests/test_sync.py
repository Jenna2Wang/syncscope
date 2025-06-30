import numpy as np
import pytest

from syncscope.sync import align_signals, estimate_offset, estimate_offset_track
from syncscope.synthetic import delayed_pair


@pytest.mark.parametrize("offset", [0.0, 0.1, 0.25, -0.15])
def test_recovers_known_offset(offset):
    rate = 100.0
    ref, target = delayed_pair(duration=6.0, rate=rate, offset_seconds=offset, seed=7)
    result = estimate_offset(ref, target, rate=rate, max_offset_seconds=0.5)
    assert result.offset_seconds == pytest.approx(offset, abs=0.02)


def test_aligned_signal_has_high_confidence():
    rate = 100.0
    ref, target = delayed_pair(duration=6.0, rate=rate, offset_seconds=0.12, seed=1)
    result = estimate_offset(ref, target, rate=rate, max_offset_seconds=0.5)
    assert result.confidence > 0.2
    assert result.peak_correlation > 0.5


def test_offset_ms_and_in_sync_flag():
    rate = 100.0
    ref, target = delayed_pair(duration=5.0, rate=rate, offset_seconds=0.0, seed=2)
    result = estimate_offset(ref, target, rate=rate, max_offset_seconds=0.3)
    assert result.offset_ms == pytest.approx(result.offset_seconds * 1000.0)
    assert result.in_sync


def test_align_signals_trims_to_overlap():
    a = np.arange(10.0)
    b = np.arange(10.0)
    a2, b2 = align_signals(a, b, offset_samples=3)
    assert len(a2) == len(b2) == 7
    # b advanced by 3 should line up with a's tail
    np.testing.assert_allclose(b2, a[3:])


def test_offset_track_is_consistent_for_constant_offset():
    rate = 100.0
    ref, target = delayed_pair(duration=10.0, rate=rate, offset_seconds=0.1, seed=9)
    centers, offsets, confidences = estimate_offset_track(
        ref, target, rate=rate, window_seconds=2.0, hop_seconds=1.0
    )
    assert len(centers) == len(offsets) == len(confidences)
    assert len(centers) > 1
    # a constant true offset should give a roughly constant estimate
    assert np.median(offsets) == pytest.approx(0.1, abs=0.03)
