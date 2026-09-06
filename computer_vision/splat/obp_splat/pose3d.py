"""V4 step 1: 3D skeleton per frame from the calibrated rig. YOLO11x-pose gives 17 COCO keypoints per camera; each
joint is triangulated by DLT over all cameras that see it with confidence, with a RANSAC over camera subsets to
reject a wrong 2D detection. Writes skeleton.json in each dataset folder: joints (17,3) in the cam19 world frame
(metres), per-joint reprojection error and number of inlier views.

  python pose3d.py data/throw_f0700 data/throw_seq/t*
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .triangulation import select_views, triangulate

COCO = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]


def proj_matrix(im):
    c = im["cam"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    return K @ im["w2c"][:3, :4]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument(
        "--holdout",
        default="",
        help="comma-separated camera stems to exclude from triangulation",
    )
    args = ap.parse_args()
    import cv2
    from ultralytics import YOLO

    from .core import read_colmap_txt

    model = YOLO("yolo11x-pose.pt")
    excluded = set(args.holdout.split(",")) - {""}
    for d in args.datasets:
        d = Path(d)
        ims = select_views(read_colmap_txt(d), excluded)
        Ps, kps = [], []
        for im in ims:
            img = cv2.imread(str(d / "images" / im["name"]))
            mp = d / "masks" / (Path(im["name"]).stem + ".png")
            mask = cv2.imread(str(mp), 0) if mp.exists() else None
            r = model.predict(img, conf=0.25, verbose=False, imgsz=1280)[0]
            best, bs = None, -1
            if r.keypoints is not None and len(r.keypoints) > 0:
                for kp in r.keypoints.data.cpu().numpy():  # (17,3) x,y,conf
                    # choose the person whose keypoints fall inside the athlete mask
                    pts = kp[kp[:, 2] > 0.3]
                    if len(pts) == 0:
                        continue
                    score = (
                        float(
                            np.mean(
                                [
                                    mask[
                                        int(min(max(y, 0), mask.shape[0] - 1)),
                                        int(min(max(x, 0), mask.shape[1] - 1)),
                                    ]
                                    > 0
                                    for x, y, _ in pts
                                ]
                            )
                        )
                        if mask is not None
                        else float(pts[:, 2].mean())
                    )
                    if score > bs:
                        best, bs = kp, score
            Ps.append(proj_matrix(im))
            kps.append(best)
        joints, errs, nin = [], [], []
        for j in range(17):
            sel = [
                (P, kp[j, :2])
                for P, kp in zip(Ps, kps)
                if kp is not None and kp[j, 2] > 0.5
            ]
            X, e, n = triangulate([s[0] for s in sel], [s[1] for s in sel])
            joints.append(X.tolist() if X is not None else None)
            errs.append(e)
            nin.append(n)
        Path(d / "skeleton.json").write_text(
            json.dumps(
                {
                    "names": COCO,
                    "joints": joints,
                    "reproj_px": errs,
                    "inliers": nin,
                    "source_cameras": [Path(im["name"]).stem for im in ims],
                    "excluded_cameras": sorted(excluded),
                }
            )
            + "\n"
        )
        ok = [e for e in errs if np.isfinite(e)]
        print(
            f"{d.name}: {sum(j is not None for j in joints)}/17 joints, median reproj {np.median(ok) if ok else float('nan'):.1f} px, median inlier views {np.median(nin):.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
