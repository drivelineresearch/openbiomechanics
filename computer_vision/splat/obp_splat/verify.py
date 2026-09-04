"""Sanity-check the calibration convention: draw the epipolar lines of one pixel in a reference camera in every other camera."""
import sys

import cv2
import numpy as np

from .dataset import load_calibration

intr, poses = load_calibration()
ref, px = sys.argv[1], np.array([float(sys.argv[2]), float(sys.argv[3])]); d = sys.argv[4]
def K(c): k = intr[c]; return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
R0, t0 = poses[ref]; tiles = []
for c in sorted(poses, key=int):
    img = cv2.imread(f"{d}/images/cam{c}.png")
    if c == ref:
        cv2.circle(img, tuple(px.astype(int)), 12, (0, 0, 255), 3)
    else:
        R1, t1 = poses[c]; R = R1 @ R0.T; t = t1 - R @ t0   # ref cam -> cam c
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        Fm = np.linalg.inv(K(c)).T @ tx @ R @ np.linalg.inv(K(ref))
        l = Fm @ np.array([px[0], px[1], 1.0]); a, b, cc = l
        pts = []
        for x in (0, 1279):
            if abs(b) > 1e-9: pts.append((x, int(-(a * x + cc) / b)))
        cv2.line(img, pts[0], pts[1], (0, 0, 255), 2)
    cv2.putText(img, f"cam{c}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3); tiles.append(cv2.resize(img, (640, 360)))
grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]); cv2.imwrite(f"{d}/epipolar_{ref}.jpg", grid); print("wrote")
