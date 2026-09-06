"""Re-render a skeleton-anchored showcase from its checkpoint with render-time cleanup: background Gaussians that
fewer than `min_views` real cameras see are dropped (they are unconstrained floaters), and both virtual cameras stay
at real camera height inside the ring, where the model has evidence.

  python render_show.py --ckpt out/show/ckpt.pt --data data/throw_dense --out out/show/clean.mp4 --repeat 4
"""

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from obp_splat import train_body as train_skel

from .core import load_views, look_at, render, rig_geometry
from .skeleton import checkpoint_bones
from .train_body import bone_frames, pose_gaussians


def seen_by(means, views, min_views):
    cnt = torch.zeros(len(means), device=means.device)
    for v in views:
        R, t = v["w2c"][:3, :3], v["w2c"][:3, 3]
        p = means @ R.T + t
        z = p[:, 2].clamp(min=1e-6)
        u = v["K"][0, 0] * p[:, 0] / z + v["K"][0, 2]
        w = v["K"][1, 1] * p[:, 1] / z + v["K"][1, 2]
        cnt += (
            (p[:, 2] > 0.5) & (u >= 0) & (u < v["w"]) & (w >= 0) & (w < v["h"])
        ).float()
    return cnt >= min_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--orbit_deg", type=float, default=160.0)
    ap.add_argument("--min_views", type=int, default=3)
    ap.add_argument("--bg_opa", type=float, default=0.15)
    ap.add_argument("--fg_opa", type=float, default=0.2)
    ap.add_argument("--bg_max_scale", type=float, default=0.5)
    ap.add_argument(
        "--bg_radius",
        type=float,
        default=0.0,
        help="keep only background within this distance (m) of the mound target; 0 = all",
    )
    ap.add_argument("--stage_gray", type=float, default=0.0)
    ap.add_argument(
        "--smooth",
        type=float,
        default=1.0,
        help="temporal Gaussian sigma (frames) on joint trajectories; 0 = off",
    )
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev)
    bg = torch.load(ck["bg"], map_location=dev)["splats"]
    train_skel.BONES = checkpoint_bones(ck)
    frames = sorted(Path(a.data).glob("t0*"))
    views0 = load_views(frames[0], set(), dev)
    target0, _, _, _ = rig_geometry(views0)
    keep = (
        seen_by(bg["means"], views0, a.min_views)
        & (torch.sigmoid(bg["opacities"]) >= a.bg_opa)
        & (torch.exp(bg["scales"]).max(1).values <= a.bg_max_scale)
    )
    if a.bg_radius > 0:
        keep &= (bg["means"] - target0).norm(dim=1) <= a.bg_radius
    bg = {k: v[keep] for k, v in bg.items()}
    print(
        f"background: kept {int(keep.sum())} of {len(keep)} gaussians (>= {a.min_views} cameras, opacity >= {a.bg_opa}, scale <= {a.bg_max_scale} m)",
        flush=True,
    )
    fk = torch.sigmoid(ck["can"]["opacities"]) >= a.fg_opa
    ck["can"] = {k: v[fk] for k, v in ck["can"].items()}
    ck["W"] = ck["W"][fk]
    print(
        f"athlete: kept {int(fk.sum())} of {len(fk)} gaussians with opacity >= {a.fg_opa}",
        flush=True,
    )
    target, up, radius, height = rig_geometry(views0)
    e1 = F.normalize(
        torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=dev)), dim=0
    )
    e2 = torch.linalg.cross(up, e1)
    bgc = torch.full((3,), a.stage_gray, device=dev)
    out = []
    J = (ck["J"] + ck["dJ"]).clone()  # (T,17,3) joints per frame
    if a.smooth > 0:  # temporal Gaussian smoothing of every joint trajectory
        r = int(3 * a.smooth)
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev) / a.smooth) ** 2)
        k = k / k.sum()
        Jp = torch.cat([J[:1].repeat(r, 1, 1), J, J[-1:].repeat(r, 1, 1)])
        J = torch.stack(
            [(Jp[i : i + 2 * r + 1] * k[:, None, None]).sum(0) for i in range(len(J))]
        )
    T = len(frames)
    n_out = (T - 1) * a.repeat + 1
    with torch.no_grad():
        for k_out in range(n_out):
            tt = k_out / a.repeat
            t0 = min(math.floor(tt), T - 2)
            w = tt - t0
            Jt = (1 - w) * J[t0] + w * J[
                t0 + 1
            ]  # pose interpolation between splat frames
            Rf, tf, _ = bone_frames(Jt)
            sp = pose_gaussians(ck["can"], ck["W"], ck["Rc"], ck["tc"], Rf, tf)
            full = {k: torch.cat([bg[k], sp[k]], 0) for k in sp}
            ang = math.radians(-a.orbit_deg / 2 + a.orbit_deg * tt / max(T - 1, 1))
            ref = views0[0]
            eye = (
                target
                + 0.85 * radius * (math.cos(ang) * e1 + math.sin(ang) * e2)
                + 0.85 * height * up
            )
            v1 = dict(ref)
            v1["w2c"] = look_at(eye, target, up)
            eye2 = (
                target
                + 0.8 * radius * (math.cos(1.4) * e1 + math.sin(1.4) * e2)
                + 0.75 * height * up
            )
            v2 = dict(ref)
            v2["w2c"] = look_at(eye2, target, up)
            im1 = render(full, v1, 3, bgc)[0].clamp(0, 1)
            im2 = render(full, v2, 3, bgc)[0].clamp(0, 1)
            out.append((torch.cat([im1, im2], 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimwrite(a.out, out, fps=30, quality=8)
    print("wrote", a.out, len(out), "frames")


if __name__ == "__main__":
    main()
