"""One entry point for the whole pipeline.

    python -m obp_splat <step> [options]

Steps (each is a thin wrapper over the module of the same name, so `--help` works per step):

    fetch        download the 8-camera OptiTrack recordings from the public OBP-CV Drive folder
    dataset      videos + published calibration -> COLMAP-format frames
    verify       draw epipolar lines of one pixel into every other camera (calibration sanity check)
    masks        athlete masks (YOLO11 segmentation)
    pose3d       3D skeleton per frame by RANSAC-DLT triangulation over the 8 cameras
    theia        compare that skeleton with a Theia3D C3D and align to the lab frame
    init         DUSt3R point cloud aligned to the rig (initialisation for the splat)
    background   fit the static background once, athlete masked out
    body         fit the athlete: one Gaussian body skinned to the triangulated skeleton
    orbit        render a novel-view orbit
    quad         render four novel views at once
    holdout      render the held-out camera beside the real footage

Everything writes into a work directory (default ./work, override with --work or OBP_SPLAT_WORK);
nothing is written inside the repository.
"""
import os
import runpy
import sys
from pathlib import Path

STEPS = {"fetch": "fetch", "dataset": "dataset", "verify": "verify", "masks": "masks", "pose3d": "pose3d",
         "theia": "theia", "init": "init_points", "background": "train_background", "body": "train_body",
         "orbit": "render_orbit", "quad": "render_quad", "holdout": "render_holdout"}
# some modules take a positional mode the step name already implies
PREFIX = {"background": ["bg"]}


def work_dir(argv):
    if "--work" in argv:
        i = argv.index("--work"); w = argv[i + 1]; del argv[i:i + 2]
    else:
        w = os.environ.get("OBP_SPLAT_WORK", "work")
    p = Path(w).resolve(); p.mkdir(parents=True, exist_ok=True); os.environ["OBP_SPLAT_WORK"] = str(p)
    return p


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in STEPS:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help") and argv[0] not in STEPS:
            print(f"unknown step: {argv[0]}"); return 2
        return 0
    step = argv[0]; rest = argv[1:]
    w = work_dir(rest)
    os.chdir(w)                                   # every step takes paths relative to the work directory
    expanded = []                                  # expand globs AFTER the chdir, so "data/throw/t0*" works from any shell
    for tok in rest:
        hits = sorted(str(q) for q in Path().glob(tok)) if any(ch in tok for ch in "*?[") else []
        expanded.extend(hits or [tok])
    sys.argv = [f"obp_splat.{STEPS[step]}"] + PREFIX.get(step, []) + expanded
    runpy.run_module(f"obp_splat.{STEPS[step]}", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
