"""Item 2: the honesty video. Render the HELD-OUT camera from the model for every frame and put it beside the real
footage of that camera, with the per-camera colour calibration applied and the PSNR printed on the frame.

  python render_holdout.py --ckpt out/v6/ckpt.pt --data data/throw_full --cam cam19 --out out/v6/holdout_cam19.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from obp_splat import train_body as train_skel

from .core import load_views, render
from .skeleton import checkpoint_bones
from .train_body import bone_frames, pose_gaussians


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cam", default="cam19")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev)
    bg = torch.load(ck["bg"], map_location=dev)["splats"]
    train_skel.BONES = checkpoint_bones(ck)
    col = ck.get("color", {}).get(a.cam)
    gain = torch.exp(torch.tensor(col[0], device=dev)) if col else 1.0
    off = torch.tensor(col[1], device=dev) if col else 0.0
    frames = sorted(Path(a.data).glob("t0*"))
    bgc = torch.zeros(3, device=dev)
    out = []
    ps = []
    with torch.no_grad():
        for t, f in enumerate(frames):
            if t % a.every:
                continue
            v = next(
                x for x in load_views(f, set(), dev) if Path(x["name"]).stem == a.cam
            )
            Rf, tf, _ = bone_frames(ck["J"][t] + ck["dJ"][t])
            sp = pose_gaussians(ck["can"], ck["W"], ck["Rc"], ck["tc"], Rf, tf)
            full = {k: torch.cat([bg[k], sp[k]], 0) for k in sp}
            img = (render(full, v, 3, bgc)[0] * gain + off).clamp(0, 1)
            p = (-10 * torch.log10(((img - v["img"]) ** 2).mean())).item()
            ps.append(p)
            fr = (torch.cat([v["img"], img], 1).cpu().numpy() * 255).astype(np.uint8)
            fr = np.ascontiguousarray(fr)
            vf = json.loads(Path(f / "meta.json").read_text())["frame"]
            cv2.putText(
                fr,
                f"real {a.cam} (excluded from appearance training)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                fr,
                f"rendered with supplied skeleton   {p:.1f} dB   frame {vf}",
                (1300, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            out.append(fr)
    imageio.mimwrite(a.out, out, fps=a.fps, quality=8)
    print(
        f"wrote {a.out}: {len(out)} frames, held-out PSNR mean {np.mean(ps):.2f} dB, min {np.min(ps):.2f}, max {np.max(ps):.2f}"
    )


if __name__ == "__main__":
    main()
