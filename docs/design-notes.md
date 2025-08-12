# Design notes

A few decisions that are not obvious from the code.

## Normalized correlation, mean-removed

The cross-correlation subtracts the mean of each signal and divides by the
product of the norms. This makes the score a Pearson-style correlation in
`[-1, 1]` that is invariant to overall loudness and to a constant brightness
offset in the video. Without mean removal a bright but static frame region would
swamp the (small) speaking signal.

## Confidence = peak prominence

The confidence is `peak - median(correlation)` rather than the raw peak value.
SyncNet uses the analogous `median(distance) - min(distance)`. The intuition is
that a *trustworthy* match is not just high, it is **isolated**: a sharp peak
sitting well above the surrounding surface. A signal that correlates moderately
at every lag (e.g. two slow ramps) gets a high raw peak but a low prominence,
which is the honest answer.

## Sub-sample interpolation

Envelopes are coarse (often 100 Hz), so the integer-lag peak quantizes the offset
to 10 ms steps. Fitting a parabola through the peak and its neighbours recovers a
fractional lag, which matters when comparing against the ~40 ms perceptibility
threshold.

## FFT vs direct correlation

For short windows the direct `numpy.correlate` is fine; for long signals it is
`O(n*m)` and dominates runtime. `normalized_cross_correlation(..., method="auto")`
switches to an FFT-based path once the inputs are large. Both paths are tested to
agree to floating-point tolerance.

## Active-speaker detection is just windowed sync

There is no separate model for ASD: it reuses the correlation cue inside short
sliding windows, scores every face, and assigns each window to the highest-scoring
face above a threshold. Contiguous windows for the same face are merged into
segments. This keeps the mental model to one concept.

## Known limitations

- The motion cue is whole-region frame differencing; it does not isolate lips
  from head motion. Supplying a tight mouth ROI helps a lot.
- There is no face detector or tracker in the box — you bring the tracks. The CLI
  `asd` command therefore treats the whole frame as a single track.
- Very low frame rates (< ~12 fps) leave too few motion samples per window for a
  reliable correlation.
