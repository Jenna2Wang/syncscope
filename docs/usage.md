# Usage guide

All of the examples below run on synthetic signals, so you can paste them into a
REPL without any media files.

## Estimating an offset

`estimate_offset(reference, target, rate, max_offset_seconds=None)` returns a
`SyncResult`. Both inputs must already be sampled at `rate` (Hz).

```python
import syncscope as ss
from syncscope.synthetic import delayed_pair

reference, target = delayed_pair(duration=8.0, rate=100.0, offset_seconds=-0.08)
result = ss.estimate_offset(reference, target, rate=100.0, max_offset_seconds=0.4)

result.offset_seconds   # estimated offset (negative => target leads reference)
result.offset_ms        # the same value in milliseconds
result.confidence       # peak height above the median correlation
result.in_sync          # |offset| within the perceptible threshold (40 ms)
```

**Sign convention.** A positive `offset_seconds` means `target` *lags*
`reference`. When `reference` is the audio envelope and `target` is the visual
motion, a positive value means the sound runs ahead of the picture.

### From raw streams

`pipeline.sync_streams` does the envelope/motion/resample steps for you:

```python
from syncscope.pipeline import sync_streams

result = sync_streams(waveform, sr=16000, frames=frames, fps=25.0)
```

### Tracking drift

For long clips whose sync may wander, `estimate_offset_track` returns the offset
estimated over sliding windows:

```python
from syncscope.sync import estimate_offset_track

centers, offsets, confidences = estimate_offset_track(reference, target, rate=100.0)
```

## Active-speaker detection

Wrap each face's per-frame motion in a `FaceTrack` (or use
`tracks_from_motion`), then call `ActiveSpeakerDetector.detect`:

```python
import syncscope as ss
from syncscope.synthetic import speaker_scene

audio_env, tracks, _ = speaker_scene(n_faces=4, active=2)
detector = ss.ActiveSpeakerDetector(window_seconds=0.5, hop_seconds=0.2, threshold=0.3)
segments = detector.detect(tracks, audio_env, audio_rate=100.0)
```

Each `SpeakerSegment` carries `track_id`, `start`, `end` and a mean `score`.

## Evaluation

```python
from syncscope.metrics import within_tolerance, average_precision

within_tolerance(result.offset_seconds, truth=-0.08, tol=0.04)
average_precision(scores, labels)   # for per-window ASD predictions
```
