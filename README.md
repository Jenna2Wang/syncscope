# syncscope

**Audio-visual synchronization and active-speaker detection in Python.**

[![CI](https://github.com/stella-sage553/syncscope/actions/workflows/ci.yml/badge.svg)](https://github.com/stella-sage553/syncscope/actions/workflows/ci.yml)
[![CodeQL](https://github.com/stella-sage553/syncscope/actions/workflows/codeql.yml/badge.svg)](https://github.com/stella-sage553/syncscope/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`syncscope` measures how well an audio track and a video line up in time, and
uses the same machinery to work out *which* face on screen is doing the talking.
It is built around one idea: when someone speaks, the energy in the audio rises
and falls together with the motion of their mouth. Cross-correlate the two and
the peak tells you both the **offset** (for lip-sync) and, per face, the
**correspondence** (for active-speaker detection).

The core is deliberately small and depends only on NumPy, so it installs
anywhere and runs without a GPU. Reading real media files (via librosa / OpenCV)
is an optional extra.

## Install

```bash
pip install syncscope            # core (NumPy only)
pip install "syncscope[audio]"   # + librosa/soundfile for loading audio
pip install "syncscope[video]"   # + OpenCV for loading video frames
```

## Quickstart

### Estimate an audio-visual offset

```python
import syncscope as ss
from syncscope.synthetic import delayed_pair

# two envelopes sampled at 100 Hz; `target` lags `reference` by 120 ms
reference, target = delayed_pair(duration=6.0, rate=100.0, offset_seconds=0.12)

result = ss.estimate_offset(reference, target, rate=100.0, max_offset_seconds=0.5)
print(result.offset_ms)     # ~ +120.0
print(result.confidence)    # how sharply the peak stands out
print(result.in_sync)       # False (120 ms is past the perceptible threshold)
```

### Detect the active speaker

```python
import syncscope as ss
from syncscope.synthetic import speaker_scene

audio_env, tracks, _ = speaker_scene(n_faces=3, active=1)
detector = ss.ActiveSpeakerDetector(threshold=0.3)
for segment in detector.detect(tracks, audio_env, audio_rate=100.0):
    print(segment.track_id, segment.start, segment.end, segment.score)
```

### Command line

```bash
syncscope demo                 # runs sync + ASD on built-in synthetic signals
syncscope sync clip.mp4        # needs the [audio] and [video] extras
syncscope asd  clip.mp4
```

## How it works

1. **Audio envelope** — frame-rate RMS (or onset) energy of the waveform.
2. **Visual motion** — mean absolute frame-to-frame difference, optionally
   restricted to a mouth region of interest.
3. **Common rate** — both signals are resampled onto a shared rate.
4. **Cross-correlation** — a normalized correlation over candidate lags; the peak
   is the offset and its prominence is the confidence. Active-speaker detection
   runs the same correlation in short sliding windows, per face.

See [`docs/`](docs/) for the architecture, a usage guide, design notes and the
API reference.

## License

MIT — see [LICENSE](LICENSE).

## Update 2026-09-27 22:51:56
Refactored code following security guidelines - ID: uk124gp5

