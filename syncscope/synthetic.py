"""Synthetic audio-visual signals for tests, demos and examples.

Everything here is deterministic given a seed so the docs and the test-suite can
agree on exact numbers without shipping any real media files.
"""

from __future__ import annotations

import numpy as np


def speech_like_envelope(duration: float, rate: float, seed: int = 0) -> np.ndarray:
    """Return a 1-D envelope with speech-like bursts of activity. TODO: implement."""
    raise NotImplementedError
