# Gaussian splatting from the OBP-CV OptiTrack rig

Novel-view reconstruction of a capture-volume trial from the eight genlocked Prime Color cameras, using only
what this repository already publishes: the provisional intrinsics and camera-19-relative pose graph in
[`computer_vision/calibration/results/`](../calibration/results). Nothing here modifies the calibration; it
consumes it.

Alongside the renders, the pipeline produces three things that are useful even if you never train a splat:

- a **calibration check** that draws the epipolar line of a chosen pixel into all seven other cameras;
- a **3D skeleton** per frame, triangulated over the eight cameras (17/17 joints on every frame of the throw,
  median reprojection error ~5 px);
- an **agreement report against Theia3D** for the same trial, which also yields the **lab-frame alignment**
  as a trial-specific candidate. This does not complete the CS-200 lab-frame or bundle-adjustment work in
  [`ROADMAP.md`](../calibration/ROADMAP.md).

## Historical results and evaluation scope

The table below contains **contributor-reported historical results**, not measurements rerun by the
maintainers. Camera 19 was excluded from appearance-loss training, but its images contributed to the
triangulated skeleton and fitted Theia alignment. These are rendering scores **conditioned on supplied
poses**, not a strict independent-camera test. The Theia C3D input-camera provenance is not established by
this pipeline. These results are not evidence of independently validated biomechanics accuracy.

The updated default excludes camera 19 from skeleton triangulation as well. **The table does not describe
that updated default; its GPU scores are pending a new run.** The full Theia-conditioned workflow still
must not be called a strict independent-camera test.

| configuration | athlete model | cam19 full-frame PSNR | cam19 athlete pixels |
| --- | --- | --- | --- |
| vanilla 3DGS, DUSt3R initialisation | none (free Gaussians per frame) | 22.8 dB | – |
| static background + masked athlete | free Gaussians per frame | 23.1 | 20.9 |
| + monocular depth loss, camera-pose refinement | free Gaussians per frame | 23.1 | 20.3 |
| skeleton-anchored athlete, 301 frames | 24 bones | 23.19 | 21.90 |
| **skeleton + Theia foot/toe/head joints** | **29 bones** | **23.21** | **22.07** |
| SMPL-X, one Gaussian per vertex | SMPL-X (joint fit) | 22.99 | 19.76 |
| free Gaussians bound to the SMPL-X mesh | SMPL-X (joint fit) | 22.98 | 19.66 |
| SMPL-X refit against the silhouettes | SMPL-X (joints + masks) | 22.93 | 19.21 |

`run_all.sh` runs the updated skeleton/Theia configuration described below. It does not reproduce the
historical table unchanged.

### Skeleton vs Theia3D

One similarity transform fitted on 3,612 joint observations over 301 frames of `OBP_movement_BaseballThrow_001`:

| | |
| --- | --- |
| similarity scale | **1.0010** (fitted scale relative to the supplied Theia output) |
| residual, median / p90 | **3.5 cm** / 8.3 cm |
| wrists, knees, ankles, elbows | 2.1–3.7 cm |
| hips, shoulders | 5.2–7.9 cm |

The contributor reports wrists and ankles agreeing to about 2 cm. The larger hip/shoulder residuals may
include differences between surface keypoints and joint centres, as well as detection, timing, and alignment
error; this fit does not separate those causes. The same fit gives the
lab up-vector in camera-19 coordinates, `[-0.121, -0.904, -0.410]`.

## What worked, and what did not

**Worked.** A multi-view-consistent initialisation (DUSt3R aligned to the rig by a similarity on camera
centres) instead of random points: 18 → 22.8 dB. Separating a static background, fitted once with the athlete
masked out, from a per-frame athlete: floaters and background flicker disappear. Anchoring the athlete to a
skeleton so one body serves every frame. Per-camera exposure and white balance, learned jointly: the eight
Prime Colors disagree by about 7 %, and averaging that disagreement into the model costs sharpness.

**Did not.** Camera-pose refinement moved the cameras 0.23° and 0.6 cm and changed nothing, which is a
compliment to the published calibration. Monocular depth losses did not help. Nor did more Gaussians on the
athlete; the stage saturates around 9k.

**Did not, and this is the interesting one.** Three SMPL-X variants all scored *below* a plain bone skin.
Binding the free-form point cloud to the fitted mesh still lost (22.98), which suggests a possible contributor: the fitted
surface sits ~6.4 cm from the athlete, so the mesh drags Gaussians off the body regardless of how they are
attached. Adding a silhouette loss cut mask error by a third and pulled the global scale from 1.32 to 1.19, but
6.4 cm of joint error remained. This experiment suggests that a tighter body-model fit may help; it does not establish a general
ranking of parametric models and free-form Gaussians. Meanwhile the cheap version of what a body
model was supposed to provide — foot, toe and head joints, taken from Theia's segment origins — is the best
configuration in the table. The SMPL-X code is kept under `experiments/` so the negative result is reproducible.

**Sparse-view limitation in this trial.** Seven training views fit to ~35 dB while the held-out view sits at 23; the
whole problem is regularising angles no camera sees. Regions outside the ring reconstruct as empty space, which
is honest but limits how far a virtual camera can travel.

## Layout

```
computer_vision/splat/
  run_all.sh            end-to-end reproduction; takes a work directory, writes nothing into the repo
  setup_env.sh          CUDA 12.4 + PyTorch 2.6 + gsplat (Ubuntu 22.04, native or WSL2)
  requirements.txt
  obp_splat/            the pipeline, one step per module (python -m obp_splat <step>)
    dataset.py          videos + published calibration -> COLMAP frames
    verify.py           epipolar overlay across all cameras
    masks.py            athlete masks (YOLO11-seg)
    pose3d.py           RANSAC-DLT triangulation of 17 joints over 8 cameras
    theia.py            agreement with a Theia3D C3D + lab-frame alignment
    init_points.py      DUSt3R point cloud aligned to the rig
    train_background.py static background, athlete masked out
    train_body.py       skeleton-anchored athlete (the main model)
    render_orbit.py     render_quad.py  render_holdout.py
  experiments/          the variants that lost, kept for reproducibility
    train_smplx.py      SMPL-X per-vertex Gaussians (+ silhouette refit)
    train_bind.py       free Gaussians bound to the SMPL-X mesh
    depth_init.py       monocular-depth initialisation
```

## Running it

```bash
bash computer_vision/splat/setup_env.sh                 # once; needs an NVIDIA GPU (10 GB is enough)
source ~/obp-splat-venv/bin/activate
bash computer_vision/splat/run_all.sh ~/splat-work      # downloads the public videos, then runs every step
```

Individual steps, from anywhere:

```bash
export PYTHONPATH=computer_vision/splat
python -m obp_splat --help
python -m obp_splat dataset --videos videos --frame 820 --frames 151 --stride 2 --out data/throw --work ~/splat-work
python -m obp_splat body --frames 'data/throw/t0*' --ref data/throw/t0075 --bg out/background/ckpt.pt \
       --init_points data/throw/t0075/points_dust3r.npz --out out/body --iters 14000 --holdout cam19 --work ~/splat-work
```

Runtime for the default configuration on an RTX 3080: about 15 minutes of preprocessing (masks, skeletons,
DUSt3R) and about 15 minutes of training.

The default script downloads the matching Theia C3D to `<work>/theia.c3d` along with the eight videos.
`fetch.py` records the Drive IDs, file sizes, and **observed** SHA-256 hashes in `download_manifest.json`.
Those hashes document downloaded bytes; they are not a verification against a published trusted manifest.
A manually supplied C3D must be from the same trial and use a verified frame offset.

## Notes and limitations

- The camera-19-relative pose graph is **provisional and not bundle-adjusted**; every result here inherits
  that. A fitted scale of 1.0010 is an observed agreement statistic, not a 0.1% calibration-accuracy bound.
- Everything is trained on one trial. Nothing here has been validated across sessions, subjects or parks.
- Held-out PSNR is a rendering metric, not a biomechanics metric. Here it evaluates appearance at cam19 given supplied pose information;
  it does not say joint angles derived from it would be accurate.
- The Edgertronic and iPhone feeds are not used: the Edgertronic extrinsics are not solved
  (`ROADMAP.md` item 3) and the iPhone is not genlocked.
- SMPL-X model files are licensed and downloaded per user; `experiments/train_smplx.py` takes a local path and
  nothing is redistributed here.
- Videos, checkpoints and datasets stay in the work directory, never in git, per the repository's rule on
  large binaries.

## Reproduction contract

- Use an external work directory such as `~/obp-splat-work`. The setup script installs Ubuntu packages and
  a CUDA 12.4 toolkit if absent; inspect it first. It was developed for Ubuntu 22.04/native or WSL2, with an
  NVIDIA RTX 3080. Windows needs WSL2. Allow disk space for eight videos, extracted frames, masks, models,
  checkpoints, and renders. The contributor reported about 30 minutes on that GPU; this is not a guarantee.
- Core versions are Python 3.10, PyTorch 2.6.0/cu124, and gsplat 1.5.3. Remaining dependencies and downloaded
  model revisions are not fully locked. Preserve `environment.txt`, model identifiers/revisions or weight
  hashes, and `download_manifest.json` alongside any new published results.
- Default selection is video frames 820–1120 inclusive, stride 2 (151 frames); reference frame is the
  middle selected frame. `HOLDOUT=cam19` excludes that camera from appearance training and, by default,
  from skeleton generation. Skeleton files record source/excluded cameras.
- `POSE_HOLDOUT=''` explicitly restores eight-camera skeleton generation for investigation of the
  historical recipe. It does not establish reproducibility of the historical scores by itself.
- `THEIA_FRAME_OFFSET=0` preserves this contributor's original C3D-array/video indexing assumption.
  The 4D reconstruction module uses `+1`. Neither matches the measurement.
  `../4d_reconstruction/offset_study.py` interpolates the C3D at fractional indices and puts the shift
  at about +0.5 frames (1.4 ms), so `0` and `+1` are roundings in opposite directions costing 0.12 cm
  and 0.08 cm of joint residual against the optimum.
  Both sit under the 1.6 cm triangulation floor, so neither invalidates the other's scores. Set
  `THEIA_FRAME_OFFSET` explicitly and re-measure for any other source. The NPZ stores the selected
  offset.
- `out/body/result.json` contains per-frame scores; `out/body/evaluation_protocol.json` records their
  scope, source cameras, frame indices, masks, and image preprocessing. Images used for scoring are
  JPEG-reencoded at quality 95; athlete masks are YOLO segmentation dilated by a 7x7 kernel. These differ
  from the 4D pipeline's PNG/BiRefNet scoring, so the two headline values are not a controlled comparison.
- For an independent-camera claim, exclude the scored camera from all data-derived conditioning inputs,
  including any upstream Theia solve, and reserve a separate split for model selection. Rerun both
  pipelines using identical evaluation frames/masks/images before claiming an improvement.

## Validation

CPU checks cover camera exclusion, triangulation, and 24-/29-bone checkpoint posing, including the renderer's
saved-topology contract. They require no source videos, model downloads, CUDA, or Driveline access:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m unittest discover -s computer_vision/splat/tests -v
ruff check --isolated --select E4,E7,E9,F,I computer_vision/splat
ruff format --isolated --check computer_vision/splat
python3 -m compileall -q computer_vision/splat
bash -n computer_vision/splat/run_all.sh computer_vision/splat/setup_env.sh
```

CI runs these checks separately from the public loader suite. It does not train or score a GPU model.
The updated end-to-end pipeline and its PSNR remain unverified until a new GPU run completes through
`out/holdout.mp4`. The optional experiments are retained for research; their negative results are not
revalidated by the CPU checks. Thanks to [the contributor](https://github.com/lblommesteyn) for this pipeline.
Code retains MIT licensing; datasets and biomechanics documentation retain the repository data license;
third-party models and weights retain their own terms.
