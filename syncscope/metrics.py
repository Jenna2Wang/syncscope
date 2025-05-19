"""Evaluation metrics for sync-offset estimation and active-speaker detection."""

from __future__ import annotations

import numpy as np


def offset_error(estimated: float, truth: float) -> float:
    """Absolute error between an estimated and a true offset, in the same units."""
    return abs(estimated - truth)


def within_tolerance(estimated: float, truth: float, tol: float) -> bool:
    """Whether ``estimated`` lands within ``tol`` of ``truth``.

    A tolerance of around 40 ms is a reasonable "looks in sync" threshold for the
    offset task.
    """
    return offset_error(estimated, truth) <= tol


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve) for binary labels.

    ``scores`` are real-valued confidences and ``labels`` are 0/1 ground truth for
    the same items (e.g. per-window "is this face speaking?"). Uses the standard
    interpolation-free definition ``AP = sum_k P(k) * dR(k)``. Returns 0.0 when
    there are no positive labels.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_positive = int(labels.sum())
    if n_positive == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_positive
    recall_step = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * recall_step))
