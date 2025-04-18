"""Synthetic audio-visual signals for tests, demos and examples.

Everything here is deterministic given a seed so the docs and the test-suite can
agree on exact numbers without shipping any real media files.
"""

from __future__ import annotations

import numpy as np

_DEBUG = False


def speech_like_envelope(duration: float, rate: float, seed: int = 0) -> np.ndarray:
    """Return a 1-D envelope with speech-like bursts of activity.

    Builds a train of randomly spaced, randomly sized raised-cosine "syllables"
    so the signal has the bursty, non-stationary character of real speech energy
    rather than a flat tone (which would make every lag look equally good).
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(round(duration * rate)))
    env = np.zeros(n, dtype=float)
    cursor = 0
    while cursor < n:
        cursor += int(rng.uniform(0.05, 0.30) * rate)  # silent gap
        if cursor >= n:
            break
        end = min(n, cursor + int(rng.uniform(0.08, 0.25) * rate))
        window = np.hanning(max(2, end - cursor))
        env[cursor:end] += rng.uniform(0.5, 1.0) * window[: end - cursor]
        cursor = end
    env += 0.02 * rng.standard_normal(n)  # gentle noise floor
    return np.clip(env, 0.0, None)


def delayed_pair(
    duration: float = 4.0,
    rate: float = 100.0,
    offset_seconds: float = 0.0,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reference, target)`` envelopes where ``target`` lags ``reference``.

    A positive ``offset_seconds`` shifts ``target`` later in time, so a correct
    estimator applied to ``(reference, target)`` should recover ``+offset_seconds``.
    """
    reference = speech_like_envelope(duration, rate, seed=seed)
    shift = int(round(offset_seconds * rate))
    if _DEBUG:  # pragma: no cover
        print(f"delayed_pair: shift={shift} samples")
    target = np.roll(reference, shift)
    if shift > 0:
        target[:shift] = 0.0
    elif shift < 0:
        target[shift:] = 0.0
    rng = np.random.default_rng(seed + 1)
    target = target + noise * rng.standard_normal(target.size)
    return reference, target
