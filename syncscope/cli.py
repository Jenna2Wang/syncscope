"""Command-line interface for syncscope."""

from __future__ import annotations

import argparse

from . import __version__


def cmd_sync(args: argparse.Namespace) -> int:
    """Estimate and report the audio-visual offset of a media file."""
    from .io import load_audio, load_video_frames
    from .pipeline import sync_streams

    try:
        waveform, sr = load_audio(args.media)
        frames, fps = load_video_frames(args.media, max_frames=args.max_frames)
    except ImportError as exc:
        print(exc)
        return 1
    result = sync_streams(waveform, sr, frames, fps, max_offset_seconds=args.max_offset)
    print(f"offset: {result.offset_ms:+.1f} ms  (confidence {result.confidence:.3f})")
    print("in sync" if result.in_sync else "out of sync")
    return 0


def cmd_asd(args: argparse.Namespace) -> int:
    """Gate the on-screen motion of a media file against its audio.

    This treats the whole frame as one track; for multi-face scenes, supply
    explicit per-face motion via the Python API (:mod:`syncscope.asd`).
    """
    from .asd import ActiveSpeakerDetector
    from .envelope import audio_envelope, envelope_rate
    from .io import load_audio, load_video_frames
    from .motion import visual_motion
    from .types import FaceTrack

    try:
        waveform, sr = load_audio(args.media)
        frames, fps = load_video_frames(args.media, max_frames=args.max_frames)
    except ImportError as exc:
        print(exc)
        return 1
    env = audio_envelope(waveform, sr)
    track = FaceTrack(track_id=0, motion=visual_motion(frames), fps=fps)
    detector = ActiveSpeakerDetector(threshold=args.threshold)
    segments = detector.detect([track], env, envelope_rate(sr))
    if not segments:
        print("no speaking activity detected")
        return 0
    for seg in segments:
        print(f"[{seg.start:6.2f}s - {seg.end:6.2f}s] track {seg.track_id} (score {seg.score:.2f})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run sync + ASD on built-in synthetic signals (no extra dependencies)."""
    from .asd import ActiveSpeakerDetector
    from .sync import estimate_offset
    from .synthetic import delayed_pair, speaker_scene

    truth_ms = 120.0
    ref, target = delayed_pair(duration=5.0, rate=100.0, offset_seconds=truth_ms / 1000.0, seed=0)
    result = estimate_offset(ref, target, rate=100.0, max_offset_seconds=0.5)
    print(
        f"sync:  estimated {result.offset_ms:+.1f} ms "
        f"(truth {truth_ms:+.1f} ms), confidence {result.confidence:.2f}"
    )

    audio_env, tracks, active = speaker_scene(duration=5.0, n_faces=3, active=2, seed=0)
    segments = ActiveSpeakerDetector().detect(tracks, audio_env, audio_rate=100.0)
    if segments:
        winner = max(segments, key=lambda s: s.duration)
        print(f"asd:   most active face = {winner.track_id} (truth {active})")
    else:
        print("asd:   no active speaker found")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syncscope",
        description="Audio-visual synchronization and active-speaker detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_sync = sub.add_parser("sync", help="estimate the audio-visual offset of a media file")
    p_sync.add_argument("media", help="path to a video (or audio) file")
    p_sync.add_argument("--max-offset", type=float, default=0.5, help="max |offset| searched, s")
    p_sync.add_argument("--max-frames", type=int, default=None, help="cap frames read")
    p_sync.set_defaults(func=cmd_sync)

    p_asd = sub.add_parser("asd", help="gate on-screen motion against the audio")
    p_asd.add_argument("media", help="path to a video file")
    p_asd.add_argument("--threshold", type=float, default=0.3, help="min correlation to speak")
    p_asd.add_argument("--max-frames", type=int, default=None, help="cap frames read")
    p_asd.set_defaults(func=cmd_asd)

    p_demo = sub.add_parser("demo", help="run on built-in synthetic signals (no extra deps)")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
