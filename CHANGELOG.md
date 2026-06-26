# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Normalized cross-correlation with direct and FFT backends and sub-sample peak
  refinement.
- Audio energy/onset envelopes and frame-difference visual motion (with optional
  ROI and grayscale reduction).
- `estimate_offset` with a constrained lag search and prominence-based confidence.
- `estimate_offset_track` for sliding-window drift estimation.
- `ActiveSpeakerDetector` that attributes speech to faces via windowed correlation.
- Evaluation metrics: offset error, tolerance check and average precision.
- A `syncscope` command line with `sync`, `asd` and `demo` subcommands.
- Optional media loaders behind the `audio` and `video` extras.
