import numpy as np
import pytest

from syncscope.metrics import average_precision, offset_error, within_tolerance


def test_offset_error_and_tolerance():
    assert offset_error(0.12, 0.10) == pytest.approx(0.02)
    assert within_tolerance(0.12, 0.10, tol=0.04)
    assert not within_tolerance(0.30, 0.10, tol=0.04)


def test_average_precision_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert average_precision(scores, labels) == 1.0


def test_average_precision_no_positives_is_zero():
    assert average_precision(np.array([0.5, 0.1]), np.array([0, 0])) == 0.0


def test_average_precision_is_between_zero_and_one():
    rng = np.random.default_rng(0)
    scores = rng.random(50)
    labels = (rng.random(50) > 0.5).astype(int)
    ap = average_precision(scores, labels)
    assert 0.0 <= ap <= 1.0


def test_average_precision_worst_ranking_is_low():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    labels = np.array([0, 0, 1, 1])
    assert average_precision(scores, labels) < 0.6
