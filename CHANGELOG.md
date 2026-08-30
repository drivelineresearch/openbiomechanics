# Changelog

Notable repository and dataset changes are recorded here. Dataset release assets remain attached to their named GitHub Release tags.

## Unreleased

- Improved contributor onboarding, repository navigation, runnable examples, loader validation, download integrity, and repository-wide quality checks.

## 2026-07-19 — Validated provisional OBP-CV calibration

- Added cross-validated, full-frame-monotonic intrinsics for eight OptiTrack, four Edgertronic, and one iPhone feed.
- Added a held-out-qualified eight-camera OptiTrack pose graph with independent loop closure of 0.072° and 3.27 mm in camera-19-relative coordinates.
- Added exact source hashes, timing diagnostics, reproducible scripts, automated safeguards, and documented rejected views and camera-pair fits.
- Added the [cube, CS-200, epipolar, and undistortion-grid annotation plan](computer_vision/calibration/ANNOTATION_PLAN.md) for reaching final lab-frame calibration.

These values are reproducible provisional results, not final lab-frame calibration. See [`computer_vision/calibration/`](computer_vision/calibration/README.md).

## 2026-07-04 — Repository overhaul and history rewrite

> [!WARNING]
> Git history was rewritten to remove about 2 GB of large binaries. Existing clones and forks from before this date must be deleted and re-cloned; old commit hashes no longer exist and `git pull` cannot reconcile the histories.

- Moved raw C3Ds, processed full-signal archives, computer-vision demo media, and Mokka installers from Git history to GitHub Releases.
- Reduced a fresh clone to roughly 28 MB.
- Split code licensing into [`LICENSE-CODE.md`](LICENSE-CODE.md) while retaining the data and biomechanics-documentation terms in [`LICENSE-DATA.md`](LICENSE-DATA.md).
- Added the repository-local [`obp`](obp/README.md) loader, runnable [`examples`](examples/README.md), generated data dictionaries, [`DATASHEET.md`](DATASHEET.md), and [`CITATION.cff`](CITATION.cff).
- Corrected POI/metadata documentation, removed redistributed textbook PDFs in favor of cited references, consolidated shared computer-vision helpers, and removed committed machine-local cruft.

Release tags introduced:

- `dataset-v1` — pitching/hitting C3D and full-signal archives
- `cv-media-v1` — computer-vision demo media
- `tools-mokka-0.6.2` — mirrored Mokka installers

## 2024-07-30 — High Performance module

- Added the [`high_performance/`](high_performance/README.md) assessment dataset and protocol documentation.

## 2023-11-02 — OBP Computer Vision

- Added the [`computer_vision/`](computer_vision/README.md) examples and supporting material.
