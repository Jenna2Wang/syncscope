# Architecture

`syncscope` is a small pipeline of pure functions over 1-D signals, with a couple
of thin convenience wrappers on top. There is no global state and no framework;
everything operates on NumPy arrays.

## Data flow

```
 waveform ──▶ envelope.audio_envelope ──┐
                                         ├─▶ resample.to_common_rate ─▶ correlate ─▶ sync.estimate_offset ─▶ SyncResult
 frames  ──▶ motion.visual_motion ──────┘                                   │
                                                                            └─▶ asd.scoring.windowed_scores ─▶ asd.ActiveSpeakerDetector ─▶ [SpeakerSegment]
```

## Modules

| Module | Responsibility |
| --- | --- |
| `syncscope.framing` | Slice a signal into overlapping frames (strided view). |
| `syncscope.envelope` | RMS / onset energy envelopes from a waveform. |
| `syncscope.motion` | Per-frame visual motion, optional ROI and grayscale reduction. |
| `syncscope.resample` | Linear resampling and a common-rate aligner. |
| `syncscope.correlate` | Normalized cross-correlation (direct + FFT) and sub-sample peak refinement. |
| `syncscope.sync` | Offset estimation, signal alignment, drift tracking. |
| `syncscope.asd` | Windowed scoring and the `ActiveSpeakerDetector`. |
| `syncscope.metrics` | Offset error and average precision for evaluation. |
| `syncscope.windowing` | Shared sliding-window index helpers. |
| `syncscope.synthetic` | Deterministic signal generators for tests and demos. |
| `syncscope.io` | Optional media loaders (librosa / OpenCV), imported lazily. |
| `syncscope.pipeline` | High-level `sync_streams` / `detect_speakers` wrappers. |
| `syncscope.cli` | `syncscope sync | asd | demo`. |

## Why correlation, not a neural network

Learned models such as SyncNet and TalkNet are more accurate on hard,
in-the-wild footage, but they require weights, a GPU and a face pipeline.
`syncscope` keeps the dependency surface tiny and the behaviour explainable: the
correspondence cue is a plain correlation you can inspect and reason about. The
public API is structured so a learned feature extractor could be dropped in
later without changing the offset/ASD logic.
