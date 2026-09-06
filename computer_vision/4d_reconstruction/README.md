# 4D reconstruction of a pitch from the OptiTrack rig and its Theia3D skeleton

A dynamic Gaussian-splat reconstruction of `OBP_movement_BaseballThrow_001` (the 62.7 mph throw) from the eight
genlocked Prime Color cameras. It uses the provisional intrinsics and camera-19-relative pose graph in
[`computer_vision/calibration/results/`](../calibration/results) and the Theia3D C3D of the same trial from the
OBP-CV Drive folder. Nothing here changes the calibration files.

The model is one Gaussian body for the whole pitch. The Theia3D segment transforms pose it in every frame, three
small per-frame corrections learned from the photos take up what the rig cannot express, and a static background
rebuilt once from clean plates sits behind it. Every frame of the pitch at the full 360 fps is used (video frames
800 to 1099: stride to follow-through, release at 950).

[Pull request 61](https://github.com/drivelineresearch/openbiomechanics/pull/61) (`computer_vision/splat/`)
set the held-out protocol and the baseline numbers used below. The two pipelines read the same calibration and
share no files.

## Historical results and evaluation scope

The table contains **contributor-reported results**, not GPU measurements rerun by maintainers. Camera 19
is excluded from background fitting and athlete appearance losses. It still contributed to the supplied
Theia alignment, which was fitted using all eight cameras. The upstream Theia C3D input-camera provenance
is not established by this pipeline. These are appearance-holdout scores **conditioned on supplied poses
and fitted alignment**, not a strict independent-camera evaluation. Athlete-mask and full-frame PSNR
measure different pixel populations.

| model | frames | cam19 athlete pixels (conditioned) | full frame |
| --- | --- | --- | --- |
| pull request 61, skeleton + Theia foot/toe/head joints (29 bones), its best row | 820-1120, every 2nd | 22.07 dB | 23.21 dB |
| **this pipeline: Theia3D-skinned body + per-frame corrections** | 800-1099, every frame | **23.73 dB** | 22.73 dB |

This pipeline scores every 10th frame of its window (30 frames), whereas #61 scores a different window.
It uses extracted PNG frames and BiRefNet masks (threshold >127/255); #61 uses JPEG-reencoded frames and
dilated YOLO masks. The rows are **not a controlled comparison** and do not establish a measured improvement. A second run of the same recipe scored 23.84 dB on the athlete;
the run-to-run spread is about 0.1 dB. The full-frame number is lower here because of the background: with
camera 19's plate excluded, the far wall behind the mound is seen only obliquely by the other seven cameras and
renders dark from camera 19.

### What each part is worth

Measured on the same held-out camera while the recipe was being built. The background plates came from all
eight cameras in these runs, so the numbers sit a little above the appearance-holdout row in the table.

| athlete model | held-out athlete pixels |
| --- | --- |
| rigid body skinned to the 15 Theia3D bones | 21.94 dB |
| + per-bone rigid refinement on 10-frame knots | 22.03 |
| + non-rigid residual network for the trunk and shoulders | 23.10 |
| + low-rank per-Gaussian per-frame offsets (the shipped model) | 24.30 |

Three things from pull request 61 were tried on top and dropped. Per-camera exposure and white-balance
calibration scored 23.91, and it needs a gauge fix so that the held-out camera does not pay for the training
cameras' drift. Learned skin weights scored 24.28, a wash. A penalty on opacity outside the mask scored 22.69;
the residual network blew up to compensate for it.

## How it works

Theia3D writes one 4x4 transform per body segment per frame. Fifteen bones are built from the segment origins:
the pelvis line, pelvis to neck, the head, and thigh, shank, foot, upper arm, forearm and hand on each side.
Each Gaussian follows a softmax blend of the bones weighted by its distance to them in the canonical pose. The
full segment transforms keep the twist about each bone, which a joint-position skeleton loses.

The rig has no spine, clavicle or scapula, so a rigid trunk cannot follow a pitch. Three corrections are learned
with the Gaussians: a rigid refinement per bone on knots every 10 frames, which cannot jitter frame to frame; a
small network from canonical position and time to an offset; and K = 8 direction vectors per Gaussian with 8
weights per frame. Color and opacity stay one shared body. Only positions move.

Color is view-independent (degree 0). Spherical-harmonic lobes do not rotate with a moving body, so higher
degrees paint the shirt green when the athlete turns.

The background plate for each camera is the per-pixel median over every 5th frame of the whole recording where
the dilated athlete mask is empty, which erases the athlete and his shadow. DUSt3R runs on those plates with the
calibrated poses, focals and principal points held fixed, so its geometry agrees with the cameras. The
confident points seed a short Gaussian refinement with no densification.

The loss is 0.8 L1 + 0.2 (1 - SSIM) inside the athlete's mask box plus a 40 px margin, plus 0.2 x L1 between
the rendered alpha and the mask. MCMC densification grows the body to 100,000 Gaussians. Gaussians that drift
more than 40 cm from every bone are switched off so MCMC relocates them.

### Theia3D alignment

`theia_alignment.json` holds the similarity from the Theia3D lab frame (millimeters) into the rig frame (meters,
camera 19 at the origin). It was fitted once, on the joint centers of the 12 limb segments against COCO keypoints
triangulated from the eight cameras over 30 frames around release, with a median residual of 1.7 cm. The C3D
frame index is the video frame index plus one. Its up vector in camera-19 coordinates is
`[-0.126, -0.902, -0.412]`; pull request 61 fitted the same alignment from its own skeleton and got
`[-0.121, -0.904, -0.410]`.

## Directory map

```
computer_vision/4d_reconstruction/
  run_all.sh             end-to-end reproduction; takes a work directory, writes nothing into the repo
  setup_env.sh           PyTorch 2.6 (cu124) + gsplat 1.5.3 + requirements.txt into a venv
  requirements.txt
  theia_alignment.json   Theia3D lab frame -> rig frame
  obp4d/                 the pipeline, one step per module (python -m obp4d <step> --work DIR)
    rig.py               published calibration -> cameras; ring geometry for virtual cameras
    fetch.py             the eight videos and the Theia3D C3D from the public Drive folder
    dataset.py           frame extraction; median background plates
    masks.py             athlete masks (BiRefNet)
    theia.py             C3D -> segment transforms in the rig frame
    background.py        DUSt3R seed with fixed poses + Gaussian refinement
    body.py              the athlete (the main model) and the held-out score
    render.py            holdout | quad | ring videos
    splat.py             rasterizer, loss, checkpoint loading
```

## Running it

```bash
bash computer_vision/4d_reconstruction/setup_env.sh      # once; Linux, NVIDIA GPU, 24 GB used here
source ~/obp4d-venv/bin/activate
bash computer_vision/4d_reconstruction/run_all.sh ~/obp4d-work
```

`run_all.sh` downloads the videos and the C3D, extracts frames, masks them and builds the plates. It then trains
and scores the held-out model with camera 19 excluded from background/body appearance losses, trains again on all eight
cameras, and renders three videos: `holdout_cam19.mp4` (prediction beside the real footage), `quad.mp4` (four
virtual cameras sweeping the ring through the real poses) and `ring.mp4` (one aimed camera circling the pitcher
while descending, then the footage rewound). The score is in `<work>/body_h19/result.json`.

On an RTX 4090 the masks take about 10 minutes, each background about 4 minutes, each body about 6 minutes and
the videos about 15 minutes. Individual steps:

```bash
export PYTHONPATH=computer_vision/4d_reconstruction
python -m obp4d --help
python -m obp4d body --work ~/obp4d-work --bg bg_h19.pt --out body_h19 --holdout 19
python -m obp4d render ring --work ~/obp4d-work --body body --bg bg.pt --out ring.mp4 --elev 25 10
```

gsplat compiles its CUDA kernels on first use. Set `TORCH_CUDA_ARCH_LIST` to your GPU (8.9 for an RTX 4090)
so it does not build every architecture.

## Known limitations

- The pose graph is provisional and not bundle-adjusted. All numbers and renders here depend on it.
- Everything is fitted on one trial. The alignment in `theia_alignment.json` belongs to this rig placement and
  this C3D; another trial needs its own fit.
- The reported PSNR measures appearance prediction given supplied poses/alignment. It is not evidence that
  joint angles read off the renders would be accurate.
- Virtual cameras hold up inside the ring at the real cameras' heights. Below about 15 degrees of elevation
  the background streaks near camera 21, and the stretch between cameras 21, 19 and 18 has the thinnest plate
  coverage.
- The Edgertronic and iPhone recordings are not used. The Edgertronic extrinsics are unsolved and the iPhone is
  not genlocked.
- The work directory holds the videos, masks, checkpoints and renders. None of it belongs in git.

## Reproduction contract and remaining validation

- Use Python 3.10 on Linux with an NVIDIA GPU and a CUDA compiler/toolkit compatible with the pinned
  PyTorch 2.6.0/cu124 and gsplat 1.5.3. `setup_env.sh` creates the Python environment; it does not install
  an NVIDIA driver or CUDA toolkit. The contributor used 24 GB VRAM on an RTX 4090. Reducing `--cap` may
  help memory use but the suggested 16 GB configuration is not independently verified.
- Choose a work directory outside the repository, such as `~/obp4d-work`. Budget space for eight source
  videos, extracted PNGs, plate frames, masks, downloaded models, checkpoints, and renders. The runtime
  estimates above are the contributor's observations, not a guarantee on other hardware.
- `fetch.py` lists the exact public Drive IDs for the throw and Theia C3D. Preserve the source hashes and
  sizes in `download_manifest.json` and installed versions in `environment.txt`. These are observed
  provenance, not checks against a published trusted checksum list. Other Python packages and remote
  model revisions are not fully locked. Record model revisions/weight hashes for any new published run;
  BiRefNet loads its model implementation with `trust_remote_code=True`.
- The default trains on decoded video indices 800–1099 inclusive (300 frames), with canonical frame 950,
  and scores 800, 810, ..., 1090. `result.json` includes the evaluation scope, exact frame lists,
  alignment hash, masks, and image preprocessing. `held_out` refers to appearance losses only.
- The shipped transform maps Theia lab millimeters to camera-19-relative meters. Its fit uses 12 limb
  correspondences over 30 frames around release, but the exact fit-frame indices and fitting script were
  not supplied. Its 1.7 cm residual is an in-sample fit statistic, not independent pose accuracy.
- The contributor supplied `C3D array index = decoded video index + 1`. The separate splat contribution
  uses an offset of zero by default. This discrepancy has not been independently resolved; the CPU tests
  verify implementation of this file's `+1` convention, not its agreement with source timing. Verify
  the source frame/time alignment before comparing residuals or using another trial.
- The trial-specific Theia alignment does not complete the CS-200 lab-origin/axes or bundle-adjustment
  milestones in the [calibration roadmap](../calibration/ROADMAP.md). Keep the published calibration
  provisional and preserve its deterministic rebuild artifacts.
- For a strict independent-camera claim, establish that the scored camera is absent from upstream pose
  estimation and alignment fitting, then rerun the complete pipeline. Compare methods using identical
  frames, masks, RGB preprocessing, and an evaluation split not used to select the recipe.

## Validation

The CPU gate verifies transform direction/units, frame-index bounds, and truthful evaluation metadata:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m unittest discover -s computer_vision/4d_reconstruction/tests -v
ruff check --isolated --select E4,E7,E9,F,I computer_vision/4d_reconstruction
ruff format --isolated --check computer_vision/4d_reconstruction
python3 -m compileall -q computer_vision/4d_reconstruction
bash -n computer_vision/4d_reconstruction/run_all.sh computer_vision/4d_reconstruction/setup_env.sh
```

CI does not download source recordings or train/render GPU models. The original end-to-end run is
contributor-reported; the maintainer changes need a new GPU run before claiming updated reproduction.
Thanks to [Doyoung-Tom Kim](https://github.com/tomdoyo) for this contribution. Code retains MIT licensing;
data and biomechanics documentation retain the repository data license; third-party models retain their
own terms.
