#!/usr/bin/env python3
"""Interleaved split stability check for iPhone rolling-shutter calibration."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import analyze_calibration as ac


def fit(dets, info):
    ids = sorted(dets)
    obj = [ac.OBJ.copy() for _ in ids]
    img = [dets[i].reshape(-1, 1, 2).astype(np.float32) for i in ids]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, (info["width"], info["height"]), None, None, flags=flags)
    return {"rms_px": float(rms), "K": K.tolist(), "D": D.reshape(-1).tolist()}


info, dets = ac.detect_video(ac.ROOT / "iphone_dynamic_checkerboard.MOV", 6)
ids = sorted(dets)
a = {i: dets[i] for j, i in enumerate(ids) if j % 2 == 0}
b = {i: dets[i] for j, i in enumerate(ids) if j % 2 == 1}
fa, fb = fit(a, info), fit(b, info)
Ka, Kb = np.asarray(fa["K"]), np.asarray(fb["K"])
payload = {
    "split_a": fa,
    "split_b": fb,
    "relative_fx_difference_pct": float(abs(Ka[0, 0] - Kb[0, 0]) / np.mean([Ka[0, 0], Kb[0, 0]]) * 100),
    "relative_fy_difference_pct": float(abs(Ka[1, 1] - Kb[1, 1]) / np.mean([Ka[1, 1], Kb[1, 1]]) * 100),
    "principal_point_difference_px": float(np.linalg.norm(Ka[:2, 2] - Kb[:2, 2])),
    "detections_per_split": [len(a), len(b)],
}
(ac.OUT / "iphone_split_stability.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
