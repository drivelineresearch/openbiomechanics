"""Render a skeleton-anchored checkpoint from FOUR novel viewpoints at once (2x2 grid), all inside the camera ring
where the model has evidence: a slow orbit, a low third-base side, a high home-plate view and a fixed catcher-side view.

  python render_quad.py --ckpt out/v9/ckpt.pt --data data/throw_full --out out/v9/quad.mp4 --repeat 2
"""
import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from obp_splat import train_body as train_skel

from .core import load_views, look_at, render, rig_geometry
from .train_body import bone_frames, pose_gaussians


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=2); ap.add_argument("--orbit_deg", type=float, default=200.0)
    ap.add_argument("--bg_opa", type=float, default=0.05); ap.add_argument("--fg_opa", type=float, default=0.1)
    ap.add_argument("--bg_radius", type=float, default=3.5); ap.add_argument("--gray", type=float, default=0.10)
    ap.add_argument("--smooth", type=float, default=1.0); ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args(); dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev); bg = torch.load(ck["bg"], map_location=dev)["splats"]
    if "bones" in ck: train_skel.BONES = ck["bones"]   # the checkpoint knows its own skeleton (24 or 29 bones)
    frames = sorted(Path(a.data).glob("t0*"))
    if len(ck["J"]) != len(frames):                       # checkpoint may have been trained on a subset
        frames = frames[:: max(1, round(len(frames) / len(ck["J"])))][: len(ck["J"])]
    views0 = load_views(frames[0], set(), dev)
    target, up, radius, height = rig_geometry(views0)
    keep = (torch.sigmoid(bg["opacities"]) >= a.bg_opa) & ((bg["means"] - target).norm(dim=1) <= a.bg_radius)
    bg = {k: v[keep] for k, v in bg.items()}
    fk = torch.sigmoid(ck["can"]["opacities"]) >= a.fg_opa
    can = {k: v[fk] for k, v in ck["can"].items()}; W = ck["W"][fk]
    print(f"background {int(keep.sum())} gaussians, athlete {int(fk.sum())}", flush=True)
    J = (ck["J"] + ck["dJ"]).clone()
    if a.smooth > 0:
        r = int(3 * a.smooth); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev) / a.smooth) ** 2); k = k / k.sum()
        Jp = torch.cat([J[:1].repeat(r, 1, 1), J, J[-1:].repeat(r, 1, 1)])
        J = torch.stack([(Jp[i:i + 2 * r + 1] * k[:, None, None]).sum(0) for i in range(len(J))])
    e1 = F.normalize(torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=dev)), dim=0); e2 = torch.linalg.cross(up, e1)
    bgc = torch.full((3,), a.gray, device=dev); ref = views0[0]
    T = len(J); n_out = (T - 1) * a.repeat + 1; out = []
    def cam(ang, hfrac, rfrac):
        eye = target + rfrac * radius * (math.cos(ang) * e1 + math.sin(ang) * e2) + hfrac * height * up
        v = dict(ref); v["w2c"] = look_at(eye, target, up); return v
    with torch.no_grad():
        for k_out in range(n_out):
            tt = k_out / a.repeat; t0 = min(int(math.floor(tt)), T - 2); w = tt - t0
            Jt = (1 - w) * J[t0] + w * J[t0 + 1]
            Rf, tf, _ = bone_frames(Jt); sp = pose_gaussians(can, W, ck["Rc"], ck["tc"], Rf, tf)
            full = {kk: torch.cat([bg[kk], sp[kk]], 0) for kk in sp}
            ang = math.radians(-a.orbit_deg / 2 + a.orbit_deg * tt / max(T - 1, 1))
            vs = [cam(ang, 0.85, 0.85), cam(1.4, 0.55, 0.8), cam(-0.6, 1.05, 0.75), cam(2.6, 0.7, 0.85)]
            ims = [cv2.resize((render(full, v, 3, bgc)[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), (a.size, int(a.size * 9 / 16))) for v in vs]
            grid = np.vstack([np.hstack(ims[:2]), np.hstack(ims[2:])])
            out.append(np.ascontiguousarray(grid))
    imageio.mimwrite(a.out, out, fps=30, quality=8); print("wrote", a.out, len(out), "frames", out[0].shape)


if __name__ == "__main__":
    main()
