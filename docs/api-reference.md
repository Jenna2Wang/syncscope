# API reference

Everything in this list is importable from the top-level `syncscope` package.

## Synchronization

### `estimate_offset(reference, target, rate, max_offset_seconds=None) -> SyncResult`
Estimate the offset (seconds) that best aligns `target` onto `reference`. A
positive result means `target` lags `reference`. `max_offset_seconds` restricts
the lag search.

### `estimate_offset_track(reference, target, rate, window_seconds=2.0, hop_seconds=1.0, max_offset_seconds=0.5) -> (centers, offsets, confidences)`
Offset estimated over sliding windows, for spotting drift.

### `align_signals(a, b, offset_samples) -> (a, b)`
Trim two signals so they overlap after removing an integer offset.

### `SyncResult`
Frozen dataclass: `offset_seconds`, `offset_samples`, `confidence`,
`peak_correlation`, `rate`. Properties: `offset_ms`, `in_sync`.

## Active-speaker detection

### `ActiveSpeakerDetector(window_seconds=0.5, hop_seconds=0.2, threshold=0.3, max_offset_seconds=0.2)`
- `detect(tracks, audio_env, audio_rate) -> list[SpeakerSegment]`
- `score_tracks(tracks, audio_env, audio_rate) -> (centers, scores)`

### `tracks_from_motion(motions, fps, start_frame=0) -> list[FaceTrack]`
Wrap per-frame motion signals into `FaceTrack` objects.

### `FaceTrack`
`track_id`, `motion`, `fps`, `start_frame`. Properties: `duration`, `end_frame`.

### `SpeakerSegment`
Frozen dataclass: `track_id`, `start`, `end`, `score`. Property: `duration`.

## Features

### `audio_envelope(waveform, sr, hop_length=160, frame_length=400, kind="rms") -> ndarray`
Frame-rate energy envelope. `kind` is `"rms"` or `"onset"`.

### `envelope_rate(sr, hop_length=160) -> float`
The envelope's sampling rate in Hz.

### `visual_motion(frames, roi=None) -> ndarray`
Per-frame motion magnitude for a `(T, H, W[, C])` stack.

## Signal utilities

### `normalized_cross_correlation(a, b, method="auto") -> (lags, corr)`
Mean-removed, norm-normalized correlation. `method` is `"direct"`, `"fft"` or
`"auto"`.

### `resample_signal(x, src_rate, dst_rate) -> ndarray`
### `to_common_rate(audio_env, audio_rate, motion, motion_rate, target_rate=None) -> (audio, motion, rate)`

## Pipelines

### `sync_streams(waveform, sr, frames, fps, max_offset_seconds=0.5) -> SyncResult`
### `detect_speakers(waveform, sr, tracks, detector=None) -> list[SpeakerSegment]`

## Metrics

### `offset_error(estimated, truth) -> float`
### `within_tolerance(estimated, truth, tol) -> bool`
### `average_precision(scores, labels) -> float`
