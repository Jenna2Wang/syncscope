"""Estimate an audio-visual offset from synthetic signals.

Run with::

    python examples/sync_offset.py

No media files or optional dependencies are required -- the signals are generated
deterministically by :mod:`syncscope.synthetic`.
"""

from __future__ import annotations

import syncscope as ss
from syncscope.synthetic import delayed_pair


def main() -> None:
    rate = 100.0
    true_offset = 0.18  # seconds; the audio runs 180 ms ahead of the picture
    reference, target = delayed_pair(duration=6.0, rate=rate, offset_seconds=true_offset, seed=11)

    result = ss.estimate_offset(reference, target, rate=rate, max_offset_seconds=0.5)

    print(f"true offset:      {true_offset * 1000:+.0f} ms")
    print(f"estimated offset: {result.offset_ms:+.0f} ms")
    print(f"confidence:       {result.confidence:.3f}")
    print("verdict:", "in sync" if result.in_sync else "out of sync")


if __name__ == "__main__":
    main()
