"""V4 step 1: 3D skeleton per frame from the calibrated rig. YOLO11x-pose gives 17 COCO keypoints per camera; each
joint is triangulated by DLT over all cameras that see it with confidence, with a RANSAC over camera subsets to
reject a wrong 2D detection. Writes skeleton.json in each dataset folder: joints (17,3) in the cam19 world frame
(metres), per-joint reprojection error and number of inlier views.

  python pose3d.py data/throw_f0700 data/throw_seq/t*
"""
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .core import read_colmap_txt

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
model = YOLO("yolo11x-pose.pt")


def proj_matrix(im):
    c = im["cam"]; K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]); return K @ im["w2c"][:3, :4]


def dlt(Ps, uvs):
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; return X[:3] / X[3]


def reproj(P, X):
    x = P @ np.append(X, 1); return x[:2] / x[2]


def triangulate(Ps, uvs, thr_px=12.0):
    n = len(Ps)
    if n < 2: return None, np.inf, 0
    best = (None, np.inf, 0)
    for k in (2, 3):
        if n < k: break
        for sub in itertools.combinations(range(n), k):
            X = dlt([Ps[i] for i in sub], [uvs[i] for i in sub])
            err = np.array([np.linalg.norm(reproj(Ps[i], X) - uvs[i]) for i in range(n)])
            inl = err < thr_px
            if inl.sum() >= 2 and (inl.sum() > best[2] or (inl.sum() == best[2] and err[inl].mean() < best[1])):
                Xr = dlt([Ps[i] for i in np.flatnonzero(inl)], [uvs[i] for i in np.flatnonzero(inl)])
                err2 = np.array([np.linalg.norm(reproj(Ps[i], Xr) - uvs[i]) for i in range(n)]); inl2 = err2 < thr_px
                best = (Xr, float(err2[inl2].mean()) if inl2.any() else np.inf, int(inl2.sum()))
    return best


for d in sys.argv[1:]:
    d = Path(d); ims = read_colmap_txt(d); masks = {}
    Ps, kps = [], []
    for im in ims:
        img = cv2.imread(str(d / "images" / im["name"])); mp = d / "masks" / (Path(im["name"]).stem + ".png")
        mask = cv2.imread(str(mp), 0) if mp.exists() else None
        r = model.predict(img, conf=0.25, verbose=False, imgsz=1280)[0]
        best, bs = None, -1
        if r.keypoints is not None and len(r.keypoints) > 0:
            for kp in r.keypoints.data.cpu().numpy():   # (17,3) x,y,conf
                # choose the person whose keypoints fall inside the athlete mask
                pts = kp[kp[:, 2] > 0.3]
                if len(pts) == 0: continue
                score = float(np.mean([mask[int(min(max(y, 0), mask.shape[0] - 1)), int(min(max(x, 0), mask.shape[1] - 1))] > 0 for x, y, _ in pts])) if mask is not None else float(pts[:, 2].mean())
                if score > bs: best, bs = kp, score
        Ps.append(proj_matrix(im)); kps.append(best)
    joints, errs, nin = [], [], []
    for j in range(17):
        sel = [(P, kp[j, :2]) for P, kp in zip(Ps, kps) if kp is not None and kp[j, 2] > 0.5]
        X, e, n = triangulate([s[0] for s in sel], [s[1] for s in sel])
        joints.append(X.tolist() if X is not None else None); errs.append(e); nin.append(n)
    json.dump(dict(names=COCO, joints=joints, reproj_px=errs, inliers=nin), open(d / "skeleton.json", "w"))
    ok = [e for e in errs if np.isfinite(e)]
    print(f"{d.name}: {sum(j is not None for j in joints)}/17 joints, median reproj {np.median(ok) if ok else float('nan'):.1f} px, median inlier views {np.median(nin):.0f}", flush=True)
