"""Athlete masks for every image of one or more COLMAP-style datasets (YOLO11 instance segmentation).
The athlete is the person instance whose mask centroid is closest to the projection of the rig's look-at
target (the mound); other people (staff at the desk) are ignored. Writes masks/<cam>.png (255 = athlete).

  python masks.py data/throw_f0700 data/throw_seq/t*
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .core import read_colmap_txt

model = YOLO("yolo11x-seg.pt")


def target_of(ims):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for im in ims:
        R = im["w2c"][:3, :3]
        C = -R.T @ im["w2c"][:3, 3]
        d = R.T @ np.array([0, 0, 1.0])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ C
    return np.linalg.solve(A, b)


for d in sys.argv[1:]:
    d = Path(d)
    ims = read_colmap_txt(d)
    (d / "masks").mkdir(exist_ok=True)
    tgt = target_of(ims)
    for im in ims:
        img = cv2.imread(str(d / "images" / im["name"]))
        h, w = img.shape[:2]
        R, t = im["w2c"][:3, :3], im["w2c"][:3, 3]
        p = R @ tgt + t
        c = im["cam"]
        u, v = c["fx"] * p[0] / p[2] + c["cx"], c["fy"] * p[1] / p[2] + c["cy"]
        r = model.predict(
            img, classes=[0], conf=0.25, verbose=False, imgsz=1280, retina_masks=True
        )[0]
        mask = np.zeros((h, w), np.uint8)
        if r.masks is not None and len(r.masks) > 0:
            ms = r.masks.data.cpu().numpy()
            best, bd = None, 1e9
            for m in ms:
                ys, xs = np.nonzero(m)
                if len(xs) < 500:
                    continue
                dist = np.hypot(xs.mean() - u, ys.mean() - v)
                if dist < bd:
                    best, bd = m, dist
            if best is not None:
                mask = (
                    cv2.resize(
                        best.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    * 255
                )
                mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        cv2.imwrite(str(d / "masks" / (Path(im["name"]).stem + ".png")), mask)
        print(d.name, im["name"], "mask px", int((mask > 0).sum()), flush=True)
