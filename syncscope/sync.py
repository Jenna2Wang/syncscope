"""Estimate the temporal offset between an audio envelope and visual motion."""

from __future__ import annotations

import numpy as np

from .types import SyncResult


def estimate_offset(reference: np.ndarray, target: np.ndarray, rate: float) -> SyncResult:
    """Estimate the offset (in seconds) that best aligns ``target`` onto ``reference``.

    TODO: implement via a constrained cross-correlation lag search.
    """
    raise NotImplementedError
