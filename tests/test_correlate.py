import numpy as np

from syncscope.correlate import normalized_cross_correlation


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
