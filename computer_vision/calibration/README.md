# Experimental camera calibration audit

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
| OptiTrack intrinsics, cameras 15–22 | Provisional numeric estimates | Held-out median corner errors: 0.154–0.287 px |
| OptiTrack relative extrinsics | Provisional connected rig | 9 accepted pair edges connect all 8 cameras |
| OptiTrack lab/world coordinates | Not complete | Cube/CS-200 correspondences and final bundle adjustment still required |
| Edgertronic intrinsics, cameras 1–4 | Cameras 2–4 usable; camera 1 weak | Camera 1 held-out median 1.050 px and p95 4.492 px |
| Edgertronic relative extrinsics | Not solved from checkerboard alone | Simultaneous checkerboard overlap graph is disconnected |
| iPhone intrinsics | Provisional numeric estimate | 0.146 px held-out median; interleaved fx differs by 0.043% |
| iPhone-to-fixed-rig extrinsics | Not complete | No genlock; rolling shutter and variable presentation timestamps |

Do not advertise the matrices in `results/` as a final lab calibration. They
are intended to seed the next orientation-aware pose-graph and bundle-adjustment
pass.

## Published files

- `results/calibration_results.json` contains all intrinsic fits and every
  attempted synchronized camera-pair fit. Pair entries include full rotation
  matrices, translations, overlap counts, stereo RMS, and independent
  per-frame stability diagnostics.
- `results/intrinsics_summary.csv` is a compact per-camera table.
- `results/extrinsics_summary.csv` is a compact camera-pair quality table.
- `results/optitrack_rig_provisional.json` contains the 9 strictly accepted
  OptiTrack pair transforms and a camera-19-relative shortest-path pose graph.
- `results/iphone_split_stability.json` records the interleaved iPhone fit
  stability check.
- `results/timing_results.json` records presentation-timestamp interval
  distributions for the iPhone checkerboard and SpyderCHECKR recordings.
- `scripts/` contains the programs that generated the results.

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
   orientation.
4. Reserve every fifth selected view for intrinsic validation.
5. Fit a low-dimensional Brown lens model with tangential distortion and k3
   fixed. This avoids publishing unstable high-order coefficients where the
   target does not cover the image corners.
6. Fit fixed-intrinsic stereo transforms on synchronized detections and compare
   each aggregate transform with independent per-frame PnP estimates.
7. Accept an OptiTrack edge only when all of the following hold:
   - stereo RMS < 0.5 px;
   - median per-frame rotation deviation < 1°;
   - median per-frame translation deviation < 30 mm.
8. Build a minimum-error camera-19-relative pose graph from accepted edges and
   report non-tree loop closure without claiming bundle adjustment.

The two non-tree accepted-edge checks close within 0.497° and 25.65 mm. That is
encouraging, but it is not a substitute for a joint all-camera optimization.

## Reproducing the audit

Python 3 with NumPy and OpenCV is required; FFmpeg/FFprobe is required for the
timestamp audit. Download the public files into
`/tmp/obp-calibration-assets` using these names:

```text
opti_dynamic_checkerboard_cam15.mp4 ... cam22.mp4
edge_dynamic_checkerboard_cam1.mp4 ... cam4.mp4
iphone_dynamic_checkerboard.MOV
iphone_color_palette.MOV
iphone_grayscale_palette.MOV
```

Then run from this directory:

```bash
python scripts/analyze_calibration.py
python scripts/build_pose_graph.py
python scripts/iphone_stability.py
python scripts/timing_audit.py
```

The detector intentionally samples the videos and can take several minutes.
The results in this commit were regenerated twice with matching reported
summary values.

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
  edge-of-frame observations would improve distortion identifiability.

## SpyderCHECKR/color correction scope

The color and grayscale targets are visible, but small. They support coarse
chart rectification, robust patch medians, neutral-ramp white balance,
exposure/tone tests, and inter-camera matching. They do not support a defensible
claim of absolute sensor colorimetry because the recordings are processed 8-bit
video, target patches can be only a few pixels across, and no measured target
Lab/spectral reference file is supplied.

See [`NEXT_AGENT.md`](NEXT_AGENT.md) for the concrete remaining color, cube,
ground-plane, and final bundle-adjustment work.

## License

The scripts are covered by [`../../LICENSE-CODE.md`](../../LICENSE-CODE.md).
Calibration videos, derived values, and documentation are covered by
[`../../LICENSE-DATA.md`](../../LICENSE-DATA.md), including its additional usage
exclusions.
