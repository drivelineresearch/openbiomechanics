"""Videos from a trained body over its background, 30 fps, the pitch played at a tenth of real speed:

holdout   the held-out camera's prediction beside the real footage (2560x720)
quad      four panels, each a virtual camera sweeping one quarter of the ring through the real poses (1280x720)
ring      one aimed camera circling the pitcher once while descending, then the footage rewound (1280x720)

python -m obp4d render holdout --work W --body body_h19 --bg bg_h19.pt --cam 19 --out holdout.mp4
python -m obp4d render quad    --work W --body body --bg bg.pt --out quad.mp4
python -m obp4d render ring    --work W --body body --bg bg.pt --out ring.mp4 [--elev 25 10] [--dist 4.5]
"""

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from . import theia
from .body import Body
from .rig import centers, load_rig, look_at, ring, ring_pose, to_torch
from .splat import load_background, rasterize


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["holdout", "quad", "ring"])
    ap.add_argument("--body", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cam", type=int, default=19, help="holdout: the camera to compare against"
    )
    ap.add_argument(
        "--elev",
        type=float,
        nargs=2,
        default=[25, 10],
        help="ring: elevation in degrees at the start and the end",
    )
    ap.add_argument(
        "--dist",
        type=float,
        default=4.5,
        help="ring: camera distance from the pitcher in meters",
    )
    a = ap.parse_args(argv)
    rig = load_rig()
    vm, K = to_torch(rig)
    bg = load_background(a.bg)
    body = Body.load(f"{a.body}/body.pt", "theia.c3d")
    frames = list(range(*body.frames))
    n = len(frames)

    @torch.no_grad()
    def view(frame, w2c, Kc):
        if not torch.is_tensor(w2c):
            w2c = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
        bgc = rasterize(*bg, w2c, Kc)[0]
        col, alpha, _ = body.render(frame, w2c, Kc)
        return ((col + (1 - alpha) * bgc)[0].clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8
        )

    with imageio.get_writer(
        a.out, fps=30, codec="libx264", quality=8, pixelformat="yuv420p"
    ) as wr:
        if a.mode == "holdout":
            for f in frames:
                photo = np.array(
                    Image.open(f"images/cam{a.cam}_{f:04d}.png").convert("RGB")
                )
                wr.append_data(np.hstack([view(f, vm[a.cam], K[a.cam]), photo]))
        elif a.mode == "quad":
            c2w, order = ring(rig)
            for i, f in enumerate(frames):
                panels = [
                    np.array(
                        Image.fromarray(
                            view(
                                f, ring_pose(c2w, order, 2 * q + 2 * i / (n - 1)), K[19]
                            )
                        ).resize((640, 360), Image.LANCZOS)
                    )
                    for q in range(4)
                ]
                wr.append_data(
                    np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
                )
        else:
            up = theia.UP / np.linalg.norm(theia.UP)
            target = body.skel.pelvis + 0.2 * up
            C = centers(rig)

            def horiz(v):
                v = v - (v @ up) * up
                return v / np.linalg.norm(v)

            _, order = ring(rig)
            path = order[order.index(15) :] + order[: order.index(15)] + [15]
            e1 = horiz(C[15] - target)
            e2 = np.cross(up, e1)
            az = np.unwrap(
                [
                    np.arctan2(horiz(C[c] - target) @ e2, horiz(C[c] - target) @ e1)
                    for c in path
                ]
            )
            az[-1] = az[0] + np.sign(az[1] - az[0]) * 2 * np.pi
            forward = []
            for i, f in enumerate(frames):
                u = i / (n - 1)
                ang = np.interp(u * (len(path) - 1), np.arange(len(path)), az)
                elev = np.radians(a.elev[0] + (a.elev[1] - a.elev[0]) * u)
                eye = target + a.dist * (
                    np.cos(elev) * (np.cos(ang) * e1 + np.sin(ang) * e2)
                    + np.sin(elev) * up
                )
                forward.append(view(f, look_at(eye, target, up), K[19]))
            for img in forward + forward[::-1]:
                wr.append_data(img)
    print("RENDER_DONE", a.out, flush=True)
