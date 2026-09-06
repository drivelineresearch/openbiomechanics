# OpenBiomechanics Computer Vision

This area combines introductory computer-vision examples with a separately validated, provisional multicamera calibration. Choose a path based on your goal and read the maturity notes before using outputs in research.

## Choose a path

| Goal | Start here | Status / prerequisites |
| --- | --- | --- |
| Learn image, video, and live-camera basics | [`TUTORIAL.md`](TUTORIAL.md) | Educational examples; some require release media |
| Inspect provisional camera calibration | [`calibration/README.md`](calibration/README.md) | Reproducible and tested, but not final lab-frame calibration |
| Reproduce calibration safeguards | [`calibration/tests/`](calibration/tests/) | Uses the exact versions in `calibration/requirements.txt` |
| Reconstruct a trial with a skeleton-driven splat | [`splat/README.md`](splat/README.md) | Experimental, single trial; separate NVIDIA/CUDA environment |
| Continue calibration validation | [`calibration/ROADMAP.md`](calibration/ROADMAP.md) | Requires source recordings and new annotations |
| Review older 2D/checkerboard experiments | `2d_calibration/` and `checkerboard_calibrate/` | Legacy examples; not the validated calibration product |

## Setup

The general examples use a large computer-vision/ML dependency set:

```bash
python3 -m pip install -r computer_vision/requirements.txt
```

Several video examples also require the separately distributed media:

```bash
scripts/download_data.sh --skip-data --with-media
```

The downloader verifies `cv_media.zip` before restoring it. The legacy face-recognition demos use a separate optional environment because `face_recognition` is unmaintained, `dlib`-based, and may require a C++/CMake toolchain:

```bash
python3 -m pip install -r computer_vision/requirements-face-recognition.txt
```

For new work, prefer an actively maintained landmarking library.

The calibration work has a smaller, exact environment because committed numeric artifacts must remain reproducible:

```bash
python3 -m pip install -r computer_vision/calibration/requirements.txt
python3 -m unittest discover -s computer_vision/calibration/tests -v
```

## Directory map

| Path | Purpose |
| --- | --- |
| `hello_world/` | Static-image, video, and live-feed face examples |
| `face_pose/`, `face_tracking/` | Legacy face pose/tracking experiments |
| `2d_calibration/` | Older 2D calibration/frame-processing scripts |
| `checkerboard_calibrate/` | Single-camera checkerboard examples and small outputs |
| `calibration/` | Validated provisional intrinsics, relative extrinsics, timing analysis, tests, and limitations |
| `splat/` | Experimental skeleton-driven reconstruction, epipolar diagnostics, and qualified rendering scores |
| `utils.py` | Shared helpers used by the introductory scripts |

## Calibration status

The committed calibration results include held-out intrinsics for eight OptiTrack, four Edgertronic, and one iPhone feed, plus a connected camera-19-relative OptiTrack pose graph. They are suitable for reproducing the published diagnostics, not for assuming final laboratory coordinates. Read the coordinate conventions, rejected-fit evidence, and limitations in [`calibration/README.md`](calibration/README.md) before using any matrix.

The next stage is documented in [`calibration/ROADMAP.md`](calibration/ROADMAP.md) and [`calibration/ANNOTATION_PLAN.md`](calibration/ANNOTATION_PLAN.md).

## Contributing

Keep educational/legacy examples distinct from validated calibration outputs. Changes under `calibration/` must pass its deterministic rebuild and test gates. See the root [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full checklist and licensing boundaries.
