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
  that [`ROADMAP.md`](../calibration/ROADMAP.md) lists as outstanding.

## What the numbers say

Camera 19 is held out of training entirely and used only for scoring, on every frame.

| configuration | athlete model | held-out PSNR | athlete pixels |
| --- | --- | --- | --- |
| vanilla 3DGS, DUSt3R initialisation | none (free Gaussians per frame) | 22.8 dB | – |
| static background + masked athlete | free Gaussians per frame | 23.1 | 20.9 |
| + monocular depth loss, camera-pose refinement | free Gaussians per frame | 23.1 | 20.3 |
| skeleton-anchored athlete, 301 frames | 24 bones | 23.19 | 21.90 |
| **skeleton + Theia foot/toe/head joints** | **29 bones** | **23.21** | **22.07** |
| SMPL-X, one Gaussian per vertex | SMPL-X (joint fit) | 22.99 | 19.76 |
| free Gaussians bound to the SMPL-X mesh | SMPL-X (joint fit) | 22.98 | 19.66 |
| SMPL-X refit against the silhouettes | SMPL-X (joints + masks) | 22.93 | 19.21 |

Reproduce the top row with `run_all.sh` (see below).

### Skeleton vs Theia3D

One similarity transform fitted on 3,612 joint observations over 301 frames of `OBP_movement_BaseballThrow_001`:

| | |
| --- | --- |
| similarity scale | **1.0010** (the published calibration is metric to 0.1 %) |
| residual, median / p90 | **3.5 cm** / 8.3 cm |
| wrists, knees, ankles, elbows | 2.1–3.7 cm |
| hips, shoulders | 5.2–7.9 cm |

Wrists and ankles agree to about 2 cm. Hips and shoulders are larger because COCO keypoints are surface
landmarks and Theia's are joint centres, so that gap is definitional rather than error. The same fit gives the
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
Binding the free-form point cloud to the fitted mesh still lost (22.98), which isolates the cause: the fitted
surface sits ~6.4 cm from the athlete, so the mesh drags Gaussians off the body regardless of how they are
attached. Adding a silhouette loss cut mask error by a third and pulled the global scale from 1.32 to 1.19, but
6.4 cm of joint error remained. **A parametric body model needs a tighter fit than joints plus a coarse
silhouette before it beats free-form Gaussians at eight views.** Meanwhile the cheap version of what a body
model was supposed to provide — foot, toe and head joints, taken from Theia's segment origins — is the best
configuration in the table. The SMPL-X code is kept under `experiments/` so the negative result is reproducible.

**Sparse views are the ceiling.** Seven training views fit to ~35 dB while the held-out view sits at 23; the
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

To include the Theia comparison and the best (29-bone) configuration, place a Theia3D C3D of the same trial at
`<work>/theia.c3d`; `run_all.sh` picks it up automatically. The OBP-CV Drive folder has one for each movement
under `theia3d_output/c3d/`.

## Notes and limitations

- The camera-19-relative pose graph is **provisional and not bundle-adjusted**; every result here inherits
  that. The Theia comparison suggests the scale is right to 0.1 %, which is a useful independent check.
- Everything is trained on one trial. Nothing here has been validated across sessions, subjects or parks.
- Held-out PSNR is a rendering metric, not a biomechanics metric. It says the model predicts an unseen camera;
  it does not say joint angles derived from it would be accurate.
- The Edgertronic and iPhone feeds are not used: the Edgertronic extrinsics are not solved
  (`ROADMAP.md` item 3) and the iPhone is not genlocked.
- SMPL-X model files are licensed and downloaded per user; `experiments/train_smplx.py` takes a local path and
  nothing is redistributed here.
- Videos, checkpoints and datasets stay in the work directory, never in git, per the repository's rule on
  large binaries.
