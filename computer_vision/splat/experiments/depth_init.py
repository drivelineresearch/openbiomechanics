"""Monocular-depth initialization for sparse-view splatting: run Depth Anything V2 (metric, indoor) on every
training image, unproject a subsample of pixels into the calibrated world frame, and write points.npz that
train_gs.py can use with --init_points. Depth from a metric model is only approximately in metres, so each
view's depth is rescaled so that the median unprojected point lands near the calibrated scene centre distance.

  python depth_init.py --data data/throw_f0700 --out data/throw_f0700/points.npz --per_view 20000
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from obp_splat.core import read_colmap_txt

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--per_view", type=int, default=20000)
    ap.add_argument("--holdout", default="")
    a = ap.parse_args(); d = Path(a.data); hold = set(a.holdout.split(",")) - {""}
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model=MODEL, device=0 if torch.cuda.is_available() else -1)
    meta = json.load(open(d / "meta.json")); cen = np.array(meta["scene_center_m"])
    pts, cols, per_view = [], [], {}
    for im in read_colmap_txt(d):
        if Path(im["name"]).stem in hold: continue
        img = imageio.imread(d / "images" / im["name"])[..., :3]
        from PIL import Image
        depth = np.array(pipe(Image.fromarray(img))["predicted_depth"], dtype=np.float32)
        if depth.shape != img.shape[:2]:
            depth = np.array(Image.fromarray(depth).resize((img.shape[1], img.shape[0]), Image.BILINEAR))
        c = im["cam"]; w2c = im["w2c"]; R, t = w2c[:3, :3], w2c[:3, 3]; C = -R.T @ t
        # rescale so the median depth of the frame matches the calibrated camera-to-centre distance
        target = np.linalg.norm(cen - C); s = target / np.median(depth); depth = depth * s
        ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        sel = np.random.default_rng(0).choice(img.size // 3, a.per_view, replace=False)
        u, v, z = xs.ravel()[sel], ys.ravel()[sel], depth.ravel()[sel]
        x = (u - c["cx"]) / c["fx"] * z; y = (v - c["cy"]) / c["fy"] * z
        P = np.stack([x, y, z], 1) @ R + C  # cam -> world: X_w = R^T (X_c - t) = R^T X_c + C
        pts.append(P); cols.append(img.reshape(-1, 3)[sel] / 255.0); per_view[im["name"]] = float(s)
        print(im["name"], "scale", round(float(s), 3), "median depth m", round(float(np.median(depth)), 2), flush=True)
    pts = np.concatenate(pts); cols = np.concatenate(cols)
    np.savez(a.out, xyz=pts.astype(np.float32), rgb=cols.astype(np.float32), scales=json.dumps(per_view))
    print("wrote", a.out, len(pts), "points; extent", np.round(pts.min(0), 1), np.round(pts.max(0), 1))


if __name__ == "__main__":
    main()
