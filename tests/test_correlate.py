import numpy as np
import pytest

from syncscope.correlate import (
    normalized_cross_correlation,
    parabolic_interpolation,
)


def test_autocorrelation_peaks_at_zero_lag():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    lags, corr = normalized_cross_correlation(x, x)
    assert lags[np.argmax(corr)] == 0
    assert corr.max() == np.float64(corr[np.argmax(corr)])
    np.testing.assert_allclose(corr.max(), 1.0, atol=1e-9)


def test_correlation_is_bounded():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(50)
    b = rng.standard_normal(70)
    _, corr = normalized_cross_correlation(a, b)
    assert corr.max() <= 1.0 + 1e-9
    assert corr.min() >= -1.0 - 1e-9


def test_lag_axis_length_matches_full_correlation():
    a = np.arange(10.0)
    b = np.arange(6.0)
    lags, corr = normalized_cross_correlation(a, b)
    assert len(lags) == len(corr) == len(a) + len(b) - 1


def test_fft_and_direct_methods_agree():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(200)
    b = rng.standard_normal(200)
    _, direct = normalized_cross_correlation(a, b, method="direct")
    _, fft = normalized_cross_correlation(a, b, method="fft")
    np.testing.assert_allclose(direct, fft, atol=1e-9)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        normalized_cross_correlation(np.arange(4.0), np.arange(4.0), method="nope")


def test_flat_signal_yields_zero_correlation():
    flat = np.ones(32)
    rng = np.random.default_rng(3)
    _, corr = normalized_cross_correlation(flat, rng.standard_normal(32))
    np.testing.assert_allclose(corr, 0.0)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        normalized_cross_correlation(np.array([]), np.arange(4.0))


def test_parabolic_interpolation_finds_offset_peak():
    # Parabola peaking between samples 1 and 2.
    corr = np.array([0.0, 0.9, 1.0, 0.7])
    delta, value = parabolic_interpolation(corr, 2)
    assert -0.5 <= delta <= 0.5
    assert value >= corr.max()


def test_parabolic_interpolation_at_edges_is_noop():
    corr = np.array([1.0, 0.5, 0.2])
    assert parabolic_interpolation(corr, 0) == (0.0, 1.0)
    assert parabolic_interpolation(corr, len(corr) - 1) == (0.0, 0.2)
