# Validated provisional camera calibration

This directory contains a reproducible, **provisional** calibration audit for
the OpenBiomechanics Computer Vision (OBP-CV) videos. It establishes which
camera parameters can be recovered from the published calibration recordings,
records the numerical starting estimates, and documents where more work is
required before treating a result as released ground truth.

The source videos remain in the public
[OBP-CV Google Drive collection](https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q/edit?usp=sharing).
They are not duplicated in this repository.

## Status at a glance

| Deliverable | Current status | Evidence |
| --- | --- | --- |
| OptiTrack intrinsics, cameras 15–22 | Provisional, physically valid estimates | Three-fold held-out median errors: 0.167–0.274 px; all models monotonic over the full frame |
| OptiTrack relative extrinsics | Provisional connected rig | 8 held-out-qualified edges connect all 8 cameras |
| OptiTrack lab/world coordinates | Not complete | Cube/CS-200 correspondences and final bundle adjustment still required |
| Edgertronic intrinsics, cameras 1–4 | Provisional, physically valid estimates | Held-out medians: 0.183–0.204 px after deterministic bad-view rejection |
| Edgertronic relative extrinsics | Not solved from checkerboard alone | Simultaneous checkerboard overlap graph is disconnected |
| iPhone intrinsics | Provisional `k1` estimate | 0.146 px held-out median; interleaved focal length differs by 0.017% |
| iPhone-to-fixed-rig extrinsics | Not complete | No genlock; rolling shutter and variable presentation timestamps |

Do not advertise the matrices in `results/` as a final lab calibration. They
are intended to seed the next orientation-aware pose-graph and bundle-adjustment
pass.

## Published files

- `results/calibration_results.json` contains input SHA-256 hashes and software
  versions, every candidate intrinsic model and fold, coverage and physical
  validity checks, rejected views, and every attempted synchronized pair fit.
  Pair entries retain raw overlap counts, time-stratified frame selections,
  full transforms, stereo RMS, and held-out per-frame diagnostics.
- `results/intrinsics_summary.csv` is a compact per-camera table.
- `results/extrinsics_summary.csv` is a compact camera-pair quality table.
- `results/optitrack_rig_provisional.json` contains the 8 strictly accepted
  OptiTrack pair transforms, all rejection reasons, and a camera-19-relative
  shortest-path pose graph.
- `results/iphone_split_stability.json` records the interleaved iPhone fit
  stability check.
- `results/timing_results.json` records presentation-timestamp interval
  distributions for the iPhone checkerboard and SpyderCHECKR recordings.
- `scripts/` contains the programs that generated the results.
- [`ANNOTATION_PLAN.md`](ANNOTATION_PLAN.md) specifies the next cube, CS-200,
  epipolar-overlay, undistortion-grid, and bundle-adjustment work.

## Coordinate conventions

Checkerboard object points use 7 × 4 internal corners from the published 5 × 8
square target. Adjacent corners are 100 mm apart. OpenCV camera coordinates are
used throughout.

For a pair named `a_to_b`, the stored transform follows:

```text
X_b = R_ba @ X_a + t_ba
```

Translations are millimetres. In `optitrack_rig_provisional.json`, camera 19 is
the arbitrary reference frame. A point in camera-19 coordinates maps into
camera `i` coordinates as:

```text
X_i = R_i19 @ X_19 + t_i19
```

This reference is not the laboratory origin and does not encode a pitching or
anatomical axis convention.

## Method

1. Decode deterministic frame samples from the eight 1280 × 720 OptiTrack
   checkerboard videos, four 1280 × 1024 Edgertronic videos, and the iPhone MOV.
2. Detect the 7 × 4 internal-corner grid with OpenCV
   `findChessboardCornersSB` and refine corners at native resolution.
3. Select up to 120 diverse views by board position, projected area, and
   circular orientation features.
4. Fit a square-pixel `k1` pilot model and reject only gross view outliers above
   the larger of 0.75 px or four robust standard deviations from the median.
5. Compare zero-distortion, `k1`, and `k1+k2` square-pixel Brown models with
   three-fold held-out validation. Tangential distortion and `k3` remain fixed
   at zero because the target does not cover enough of every frame to identify
   them reliably.
6. Reject any candidate whose radial mapping is non-positive or non-monotonic
   anywhere in the full image. Choose the simplest physically valid model within
   a small held-out-error tolerance of the best candidate, then refit it on all
   retained views.
7. Intersect the views retained by both intrinsic fits, select synchronized pair
   frames uniformly over that qualified overlap, fit on four of every five
   frames, and evaluate the transform on the held-out fifth before refitting the
   published transform on all selected frames. Both raw and qualified overlap
   counts remain in the output.
8. Accept an OptiTrack edge only when all of the following hold:
   - stereo RMS < 0.5 px;
   - held-out rotation median < 1° and p95 < 1.5°;
   - held-out translation median and p95 < 30 mm;
   - at least 4 held-out frames.
9. Build a validation-weighted camera-19-relative pose graph from accepted edges
   and report non-tree loop closure without claiming bundle adjustment.

The independent non-tree accepted edge closes within 0.072° and 3.27 mm. That
is encouraging, but it is not a substitute for joint all-camera optimization.

## Reproducing the audit

Python 3 and FFmpeg/FFprobe are required. Install the exact NumPy and headless
OpenCV versions used for the committed results:

```bash
python3 -m pip install -r requirements.txt
```

Download the public files into `/tmp/obp-calibration-assets` using these names:

```text
opti_dynamic_checkerboard_cam15.mp4 ... cam22.mp4
edge_dynamic_checkerboard_cam1.mp4 ... cam4.mp4
iphone_dynamic_checkerboard.MOV
iphone_color_palette.MOV
iphone_grayscale_palette.MOV
```

Then run from this directory. Use `--assets-dir PATH` on each command if the
videos live elsewhere:

```bash
python3 scripts/analyze_calibration.py
python3 scripts/build_pose_graph.py
python3 scripts/iphone_stability.py
python3 scripts/timing_audit.py
python3 -m unittest discover -s tests -v
```

After changing only stereo or pose-graph logic, pass `--reuse-intrinsics` to
`analyze_calibration.py`. Reuse is refused unless the schema and all source
SHA-256 hashes match the existing results.

The detector and cross-validation sweep can take several minutes. Each output
records exact input hashes and software versions. Regeneration is clean at the
JSON/CSV level and the deterministic pose graph is rebuilt in CI.

## Known limitations

- A planar checkerboard has a 180° orientation ambiguity. The blue orientation
  marks should be incorporated into the final OptiTrack solver rather than
  trusting corner order for every shared frame.
- Several camera pairs have low intrinsic reprojection error but poor stereo
  consistency. The complete pair table is retained so failed links are not
  hidden.
- Edgertronic checkerboard overlap does not connect all four cameras. The large
  3-D cube and the CS-200 ground plane are visible in all feeds and are the
  preferred route for its rig calibration.
- The cube topology/joint correspondences and CS-200 marker centers have not
  yet been annotated in the video frames.
- The iPhone has rolling shutter and variable frame intervals. Use presentation
  timestamps rather than `frame_index / nominal_fps`.
- Intrinsic validation covers the observed checkerboard distribution. More
  edge-of-frame observations would improve distortion identifiability. The
  output reports both observed and full-frame normalized radii so this gap is
  explicit; most feeds conservatively select zero distortion.

## SpyderCHECKR/color correction scope

The color and grayscale targets are visible, but small. They support coarse
chart rectification, robust patch medians, neutral-ramp white balance,
exposure/tone tests, and inter-camera matching. They do not support a defensible
claim of absolute sensor colorimetry because the recordings are processed 8-bit
video, target patches can be only a few pixels across, and no measured target
Lab/spectral reference file is supplied.

See [`ANNOTATION_PLAN.md`](ANNOTATION_PLAN.md) for the annotation schemas and
visual-QA workflow, and [`NEXT_AGENT.md`](NEXT_AGENT.md) for the complete
remaining color, geometry, and bundle-adjustment handoff.

## License

The scripts are covered by [`../../LICENSE-CODE.md`](../../LICENSE-CODE.md).
Calibration videos, derived values, and documentation are covered by
[`../../LICENSE-DATA.md`](../../LICENSE-DATA.md), including its additional usage
exclusions.
