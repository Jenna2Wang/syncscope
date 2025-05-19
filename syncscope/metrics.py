"""Evaluation metrics for sync-offset estimation and active-speaker detection."""

from __future__ import annotations


def offset_error(estimated: float, truth: float) -> float:
    """Absolute error between an estimated and a true offset, in the same units."""
    return abs(estimated - truth)


def within_tolerance(estimated: float, truth: float, tol: float) -> bool:
    """Whether ``estimated`` lands within ``tol`` of ``truth``.

    A tolerance of around 40 ms is a reasonable "looks in sync" threshold for the
    offset task.
    """
    return offset_error(estimated, truth) <= tol
