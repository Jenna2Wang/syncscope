"""Find the active speaker in a synthetic multi-face scene.

Run with::

    python examples/active_speaker.py

One of the faces "speaks" (its motion follows the audio); the others just move
around. The detector should attribute the speaking segments to the right face.
"""

from __future__ import annotations

import syncscope as ss
from syncscope.synthetic import speaker_scene


def main() -> None:
    audio_env, tracks, active = speaker_scene(duration=6.0, n_faces=3, active=1, seed=21)
    detector = ss.ActiveSpeakerDetector(threshold=0.3)
    segments = detector.detect(tracks, audio_env, audio_rate=100.0)

    print(f"faces in scene:       {[t.track_id for t in tracks]}")
    print(f"ground-truth speaker: {active}")
    print("detected speaking segments:")
    for seg in segments:
        print(
            f"  face {seg.track_id}: {seg.start:5.2f}s -> {seg.end:5.2f}s  (score {seg.score:.2f})"
        )


if __name__ == "__main__":
    main()
