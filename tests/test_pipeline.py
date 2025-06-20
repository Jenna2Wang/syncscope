import numpy as np

import syncscope as ss


def _talking_frames(motion_1d, size=8):
    """Expand a 1-D motion signal into a tiny (T, H, W) stack with that energy."""
    frames = np.zeros((len(motion_1d), size, size))
    for i, m in enumerate(motion_1d):
        frames[i] = m
    return frames


def test_sync_streams_recovers_offset():
    from syncscope.synthetic import speech_like_envelope

    # Build an audio waveform whose envelope matches a moving frame stack.
    env = speech_like_envelope(duration=5.0, rate=200.0, seed=4)
    # waveform sampled at 16 kHz that carries this envelope as amplitude modulation
    sr = 16000
    t = np.arange(int(5.0 * sr)) / sr
    carrier = np.sin(2 * np.pi * 220 * t)
    amp = np.interp(t, np.arange(env.size) / 200.0, env)
    waveform = carrier * amp
    # frames whose motion follows the same envelope at 25 fps, delayed slightly
    fps = 25.0
    motion = np.interp(np.arange(int(5.0 * fps)) / fps, np.arange(env.size) / 200.0, env)
    frames = _talking_frames(np.roll(motion, 3))
    result = ss.sync_streams(waveform, sr, frames, fps, max_offset_seconds=0.5)
    assert isinstance(result, ss.SyncResult)
    assert abs(result.offset_seconds) < 0.5


def test_detect_speakers_end_to_end():
    from syncscope.synthetic import speaker_scene

    audio_env, tracks, active = speaker_scene(duration=6.0, n_faces=3, active=0, seed=5)
    # turn the envelope back into a waveform for the public entry point
    sr = 16000
    t = np.arange(int(6.0 * sr)) / sr
    waveform = np.sin(2 * np.pi * 200 * t) * np.interp(
        t, np.arange(audio_env.size) / 100.0, audio_env
    )
    segments = ss.detect_speakers(waveform, sr, tracks)
    assert segments
    spoken: dict[int, float] = {}
    for seg in segments:
        spoken[seg.track_id] = spoken.get(seg.track_id, 0.0) + seg.duration
    assert max(spoken, key=spoken.get) == active
