"""Render a sequence of per-frame Gaussian splat checkpoints from a novel, slowly orbiting camera (4D playback),
plus a fixed novel view, into one mp4. Frames are rendered at the same resolution as the OptiTrack feeds.

  python render_seq.py --seq out/seq --data data/throw_seq --out out/throw_4d.mp4
"""

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from obp_splat.core import load_views, look_at, render, rig_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--orbit_deg",
        type=float,
        default=120.0,
        help="total orbit sweep across the sequence",
    )
    ap.add_argument(
        "--bg",
        default="",
        help="frozen background checkpoint to composite under each frame",
    )
    ap.add_argument(
        "--repeat", type=int, default=2, help="output frames per splat frame"
    )
    a = ap.parse_args()
    dev = "cuda"
    ckpts = sorted(Path(a.seq).glob("t*/ckpt.pt"))
    assert ckpts, "no checkpoints"
    views = load_views(Path(a.data) / ckpts[0].parent.name, set(), dev)
    ref = views[0]
    target, up, radius, height = rig_geometry(views)
    dev_t = target.device
    e1 = F.normalize(
        torch.linalg.cross(up, torch.tensor([1.0, 0.0, 0.0], device=dev_t)), dim=0
    )
    e2 = torch.linalg.cross(up, e1)
    bg = torch.zeros(3, device=dev)
    bgsp = torch.load(a.bg, map_location=dev)["splats"] if a.bg else None
    frames = []
    with torch.no_grad():
        for i, ck in enumerate(ckpts):
            sp = torch.load(ck, map_location=dev)["splats"]
            if bgsp is not None:
                sp = {k: torch.cat([bgsp[k], sp[k]], 0) for k in sp}
            ang = math.radians(
                -a.orbit_deg / 2 + a.orbit_deg * i / max(len(ckpts) - 1, 1)
            )
            eye = (
                target
                + 0.9 * radius * (math.cos(ang) * e1 + math.sin(ang) * e2)
                + 0.8 * height * up
            )
            v1 = dict(ref)
            v1["w2c"] = look_at(eye, target, up)
            eye2 = (
                target
                + 0.9 * radius * (math.cos(1.2) * e1 + math.sin(1.2) * e2)
                + 0.5 * height * up
            )  # fixed novel view, lower
            v2 = dict(ref)
            v2["w2c"] = look_at(eye2, target, up)
            im1 = render(sp, v1, 3, bg)[0].clamp(0, 1)
            im2 = render(sp, v2, 3, bg)[0].clamp(0, 1)
            fr = (torch.cat([im1, im2], 1).cpu().numpy() * 255).astype(np.uint8)
            frames.extend([fr] * a.repeat)
            print(f"{ck.parent.name}: {len(sp['means'])} gaussians", flush=True)
    imageio.mimwrite(a.out, frames, fps=30, quality=8)
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
