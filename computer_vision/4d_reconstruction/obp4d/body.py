"""The athlete: one Gaussian body for the whole pitch, skinned to the Theia3D segments over the frozen background.

In every frame each Gaussian's position is a linear blend of 15 bone motions from the canonical pose (the ball
release frame), with skin weights from its distance to the bones. Three per-frame corrections are learned with
the Gaussians: a rigid refinement per bone on 10-frame knots, a small non-rigid residual network (canonical
position and time to an offset) for the trunk and shoulders the rig has no segments for, and a low-rank offset
per Gaussian (K direction vectors per Gaussian times K weights per frame). Color is view-independent so it can
follow the body. The loss is photometric inside the athlete's mask box plus a silhouette term against the mask.
MCMC densification grows the body to a fixed budget; Gaussians that drift away from every bone are switched
off so it relocates them.

  python -m obp4d body --work W --bg bg.pt --out body [--holdout 19] [--frames 800 1100] [--canon 950]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy import MCMCStrategy
from PIL import Image
from scipy.spatial.transform import Rotation

from . import theia
from .rig import CAMS, H, W, load_rig, to_torch
from .splat import load_background, photometric, rasterize

dev = "cuda"
SIGMA = 0.03  # skin weight falloff (meters): softmax(-0.5 (distance / SIGMA)^2) over the bones
PRUNE_DIST = 0.40  # Gaussians farther than this from every bone are switched off for MCMC to relocate
MARGIN = 40  # pixels around the athlete's mask box covered by the photometric loss
SIL_W = 0.2  # silhouette term: L1 between rendered alpha and the mask
KNOT = 10  # the rigid per-bone refinement lives on knots every KNOT frames, linearly interpolated
DELTA_LR, SMOOTH_W, PRIOR_W = 2e-3, 100.0, 1.0
RESID_LR, RESID_W, RESID_START = (
    5e-4,
    1.0,
    2000,
)  # residual network: rate, L2 on its offsets, start step
K_LR, LRW_LR, DIRS_LR, LRSMOOTH_W, DIRS_L2, LR_START = (
    8,
    1e-3,
    1.6e-4,
    100.0,
    1e-3,
    2000,
)  # low-rank offsets
LRS = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "colors": 2.5e-3,
    "dirs": DIRS_LR,
}
INIT_N, INIT_SPREAD, INIT_SCALE = (
    20000,
    0.06,
    0.02,
)  # starting Gaussians scattered along the bones

# (segment whose motion the bone follows, endpoint a, endpoint b, lo, hi): the bone is a + u (b - a), u in [lo, hi]
BONES = [
    ("pelvis", "l_thigh", "r_thigh", -0.2, 1.2),
    ("torso", "pelvis", "torso", 0.0, 1.0),
    ("head", "torso", "head", 0.0, 2.2),
    ("l_thigh", "l_thigh", "l_shank", 0, 1),
    ("l_shank", "l_shank", "l_foot", 0, 1),
    ("l_foot", "l_foot", "l_toes", 0, 1.6),
    ("r_thigh", "r_thigh", "r_shank", 0, 1),
    ("r_shank", "r_shank", "r_foot", 0, 1),
    ("r_foot", "r_foot", "r_toes", 0, 1.6),
    ("l_uarm", "l_uarm", "l_larm", 0, 1),
    ("l_larm", "l_larm", "l_hand", 0, 1),
    ("l_hand", "l_larm", "l_hand", 1.0, 1.6),
    ("r_uarm", "r_uarm", "r_larm", 0, 1),
    ("r_larm", "r_larm", "r_hand", 0, 1),
    ("r_hand", "r_larm", "r_hand", 1.0, 1.6),
]


def quat_mul(q, r):
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def rodrigues(v):
    """(K, 3) axis-angle -> (K, 3, 3)"""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a = v / th
    Kx = torch.zeros(len(v), 3, 3, device=v.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0], Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = (
        -a[:, 2],
        a[:, 1],
        a[:, 2],
        -a[:, 0],
        -a[:, 1],
        a[:, 0],
    )
    return (
        torch.eye(3, device=v.device)
        + torch.sin(th)[..., None] * Kx
        + (1 - torch.cos(th))[..., None] * (Kx @ Kx)
    )


class Skeleton:
    """the 15 bones of the canonical pose and their rigid motion into every video frame"""

    def __init__(self, c3d, canon):
        T_all, names = theia.load_segments(c3d)
        self.canon = canon
        seg = [names.index(b[0]) for b in BONES]
        k = theia.video_to_c3d_index(canon, len(T_all))
        M = T_all[:, seg] @ np.linalg.inv(T_all[k, seg])  # canonical -> frame, per bone
        self.M = torch.tensor(M, dtype=torch.float32, device=dev)
        q = Rotation.from_matrix(M[:, :, :3, :3].reshape(-1, 3, 3)).as_quat()[
            :, [3, 0, 1, 2]
        ]
        self.Q = torch.tensor(
            q.reshape(len(M), len(BONES), 4), dtype=torch.float32, device=dev
        )
        self.origin = torch.tensor(
            T_all[:, seg][:, :, :3, 3], dtype=torch.float32, device=dev
        )
        P = {n: T_all[k, i, :3, 3] for i, n in enumerate(names)}
        self.A = torch.tensor(
            np.stack([P[a] + lo * (P[b] - P[a]) for _, a, b, lo, _ in BONES]),
            dtype=torch.float32,
            device=dev,
        )
        self.B = torch.tensor(
            np.stack([P[a] + hi * (P[b] - P[a]) for _, a, b, _, hi in BONES]),
            dtype=torch.float32,
            device=dev,
        )
        self.pelvis = T_all[k, names.index("pelvis"), :3, 3]

    def bone_dist(self, p):
        """(N, 15) distance from points (N, 3) to each canonical bone segment"""
        ab = self.B - self.A
        u = ((p[:, None] - self.A[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None]
        return (p[:, None] - (self.A[None] + u.clamp(0, 1)[..., None] * ab[None])).norm(
            dim=-1
        )

    def motion(self, frame, delta, f0):
        """per-bone (R, t, quaternion) canonical -> video frame, with the knot refinement delta applied about
        the bone origin"""
        i = theia.video_to_c3d_index(frame, len(self.M))
        Mf, Qf = self.M[i], self.Q[i]
        if delta is None:
            return Mf[:, :3, :3], Mf[:, :3, 3], Qf
        u = (frame - f0) / KNOT
        k = min(int(u), len(delta) - 2)
        d = (1 - (u - k)) * delta[k] + (u - k) * delta[k + 1]
        Rd, o = rodrigues(d[:, :3]), self.origin[i]
        R = Rd @ Mf[:, :3, :3]
        t = (Rd @ (Mf[:, :3, 3] - o)[..., None])[..., 0] + o + d[:, 3:]
        th = d[:, :3].norm(dim=-1, keepdim=True)
        qd = torch.cat(
            [torch.cos(th / 2), d[:, :3] / th.clamp_min(1e-8) * torch.sin(th / 2)], -1
        )
        return R, t, quat_mul(qd, Qf)


class Deform(torch.nn.Module):
    """non-rigid residual: canonical position and time -> offset in meters, starts as a no-op"""

    def __init__(self, nx=6, nt=4):
        super().__init__()
        self.fx = 2.0 ** torch.arange(nx, device=dev) * np.pi
        self.ft = 2.0 ** torch.arange(nt, device=dev) * np.pi
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 + 6 * nx + 1 + 2 * nt, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        px = (x[:, :, None] * self.fx).flatten(1)
        pt = t * self.ft
        return self.net(
            torch.cat(
                [
                    x,
                    torch.sin(px),
                    torch.cos(px),
                    t.expand(len(x), 1),
                    torch.sin(pt).expand(len(x), -1),
                    torch.cos(pt).expand(len(x), -1),
                ],
                1,
            )
        )


class Body:
    """canonical Gaussians plus the three corrections; the residual network and the low-rank weights are
    attached part-way through training"""

    def __init__(self, skel, params, f0, n, delta, deform=None, lrw=None):
        self.skel, self.params, self.f0, self.n, self.delta, self.deform, self.lrw = (
            skel,
            params,
            f0,
            n,
            delta,
            deform,
            lrw,
        )

    @property
    def frames(self):
        return self.f0, self.f0 + self.n

    def time(self, frame):
        return torch.tensor([[(frame - self.f0) / self.n]], device=dev)

    def posed(self, frame):
        p = self.params
        w = torch.softmax(
            -0.5 * (self.skel.bone_dist(p["means"].detach()) / SIGMA) ** 2, dim=1
        )
        R, t, q = self.skel.motion(frame, self.delta, self.f0)
        means = (
            w[..., None] * (torch.einsum("kij,nj->nki", R, p["means"]) + t[None])
        ).sum(1)
        if self.deform is not None:
            means = means + self.deform(p["means"].detach(), self.time(frame))
        if self.lrw is not None:
            means = means + (
                p["dirs"].view(-1, K_LR, 3) * self.lrw[frame - self.f0][None, :, None]
            ).sum(1)
        return means, quat_mul(q[w.argmax(1)], p["quats"])

    def render(self, frame, vm, K, absgrad=False):
        means, quats = self.posed(frame)
        p = self.params
        return rasterize(
            means,
            quats,
            torch.exp(p["scales"]),
            torch.sigmoid(p["opacities"]),
            p["colors"],
            vm,
            K,
            absgrad,
        )

    def save(self, path):
        torch.save(
            {
                "params": {k: v.detach().cpu() for k, v in self.params.items()},
                "delta": self.delta.detach().cpu(),
                "lrw": None if self.lrw is None else self.lrw.detach().cpu(),
                "deform": None
                if self.deform is None
                else {k: v.cpu() for k, v in self.deform.state_dict().items()},
                "canon": self.skel.canon,
                "frames": list(self.frames),
                "alignment_sha256": theia.ALIGNMENT_SHA256,
            },
            path,
        )

    @staticmethod
    def load(path, c3d):
        ck = torch.load(path, weights_only=True)
        if ck.get("alignment_sha256", theia.ALIGNMENT_SHA256) != theia.ALIGNMENT_SHA256:
            raise ValueError("Checkpoint was trained with a different Theia alignment")
        skel = Skeleton(c3d, ck["canon"])
        f0, f1 = ck["frames"]
        params = {k: v.to(dev) for k, v in ck["params"].items()}
        deform = None
        if ck["deform"] is not None:
            deform = Deform().to(dev)
            deform.load_state_dict({k: v.to(dev) for k, v in ck["deform"].items()})
        lrw = None if ck["lrw"] is None else ck["lrw"].to(dev)
        return Body(skel, params, f0, f1 - f0, ck["delta"].to(dev), deform, lrw)


def init_params(skel):
    """Gaussians scattered along the canonical bones, gray, half opaque"""
    rng = np.random.default_rng(0)
    A, B = skel.A.cpu().numpy(), skel.B.cpu().numpy()
    L = np.linalg.norm(B - A, axis=1)
    k = rng.choice(len(A), INIT_N, p=L / L.sum())
    u = rng.random((INIT_N, 1))
    means = A[k] + u * (B[k] - A[k]) + rng.normal(0, INIT_SPREAD, (INIT_N, 3))
    P = torch.nn.Parameter
    return torch.nn.ParameterDict(
        {
            "means": P(torch.tensor(means, dtype=torch.float32, device=dev)),
            "scales": P(torch.full((INIT_N, 3), float(np.log(INIT_SCALE)), device=dev)),
            "quats": P(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(INIT_N, 1)),
            "opacities": P(torch.zeros(INIT_N, device=dev)),
            "colors": P(torch.zeros(INIT_N, 1, 3, device=dev)),
            "dirs": P(torch.zeros(INIT_N, K_LR * 3, device=dev)),
        }
    )


def composite(body, frame, cam, bg_img, vm, K, absgrad=False):
    col, alpha, info = body.render(frame, vm[cam], K[cam], absgrad)
    return col + (1 - alpha) * bg_img[cam], alpha, info


@torch.no_grad()
def evaluate(body, cam, gt, mk, bg_img, vm, K, frames):
    """PSNR at one camera over every 10th frame: on the athlete's mask pixels and on the full frame"""
    ci, athlete, full = CAMS.index(cam), [], []
    for fi in range(0, len(frames), 10):
        pred = composite(body, frames[fi], cam, bg_img, vm, K)[0][0]
        err = (pred - gt[ci, fi].float() / 255) ** 2
        athlete.append(-10 * torch.log10(err[mk[ci, fi] > 127].mean()).item())
        full.append(-10 * torch.log10(err.mean()).item())
    return float(np.mean(athlete)), float(np.mean(full)), len(athlete)


@torch.no_grad()
def progress(body, path, gt, mk, bg_img, vm, K, frames):
    """camera 19 at three frames, prediction beside the photo"""
    rows = []
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        pred = composite(body, f, 19, bg_img, vm, K)[0][0]
        rows.append(
            torch.cat([pred, gt[CAMS.index(19), frames.index(f)].float() / 255], 1)
        )
    Image.fromarray(
        (torch.cat(rows, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    ).resize((1280, 1080)).save(path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--holdout",
        type=int,
        choices=[0, *CAMS],
        default=0,
        help="camera left out of training, 0 = none",
    )
    ap.add_argument(
        "--frames",
        type=int,
        nargs=2,
        default=[800, 1100],
        help="video frame range, [first, last)",
    )
    ap.add_argument(
        "--canon",
        type=int,
        default=950,
        help="canonical pose: the video frame of ball release",
    )
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--cap", type=int, default=100000, help="MCMC Gaussian budget")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rig = load_rig()
    vm, K = to_torch(rig)
    f0, f1 = a.frames
    frames = list(range(f0, f1))
    n = len(frames)
    train_cams = [c for c in CAMS if c != a.holdout]
    skel = Skeleton("theia.c3d", a.canon)
    params = init_params(skel)
    delta = torch.nn.Parameter(torch.zeros(n // KNOT + 1, len(BONES), 6, device=dev))
    deform = Deform().to(dev)
    lrw = torch.nn.Parameter(0.1 * torch.randn(n, K_LR, device=dev))
    body = Body(skel, params, f0, n, delta)
    opts = {k: torch.optim.Adam([params[k]], lr=LRS[k], eps=1e-15) for k in params}
    extra = [
        torch.optim.Adam([delta], lr=DELTA_LR),
        torch.optim.Adam(deform.parameters(), lr=RESID_LR),
        torch.optim.Adam([lrw], lr=LRW_LR),
    ]
    strategy = MCMCStrategy(
        cap_max=a.cap,
        refine_start_iter=500,
        refine_stop_iter=int(0.7 * a.iters),
        refine_every=100,
        verbose=False,
    )
    state = strategy.initialize_state()
    bg = load_background(a.bg)
    bg_img = {
        c: rasterize(*bg, vm[c], K[c])[0] for c in CAMS
    }  # static, rendered once per camera

    print("loading", len(CAMS) * n, "images and masks", flush=True)
    gt = torch.stack(
        [
            torch.stack(
                [
                    torch.from_numpy(
                        np.array(
                            Image.open(f"images/cam{c}_{f:04d}.png").convert("RGB")
                        )
                    )
                    for f in frames
                ]
            )
            for c in CAMS
        ]
    ).to(dev)
    mk = torch.stack(
        [
            torch.stack(
                [
                    torch.from_numpy(
                        np.array(Image.open(f"masks/cam{c}_{f:04d}.png").convert("L"))
                    )
                    for f in frames
                ]
            )
            for c in CAMS
        ]
    ).to(dev)
    assert gt.shape == (len(CAMS), n, H, W, 3) and mk.shape == (len(CAMS), n, H, W), (
        gt.shape,
        mk.shape,
    )

    rng = np.random.default_rng(0)
    for it in range(a.iters):
        if it == RESID_START:
            body.deform = deform
        if it == LR_START:
            body.lrw = lrw
        ci, fi = CAMS.index(train_cams[rng.integers(len(train_cams))]), rng.integers(n)
        c, f = CAMS[ci], frames[fi]
        m = mk[ci, fi].float() / 255
        ys, xs = torch.where(m > 0.5)
        if len(ys) == 0:
            continue
        y0, y1 = max(int(ys.min()) - MARGIN, 0), min(int(ys.max()) + MARGIN, H)
        x0, x1 = max(int(xs.min()) - MARGIN, 0), min(int(xs.max()) + MARGIN, W)
        pred, alpha, info = composite(body, f, c, bg_img, vm, K, absgrad=True)
        photo, s = photometric(
            pred[:, y0:y1, x0:x1], (gt[ci, fi][None].float() / 255)[:, y0:y1, x0:x1]
        )
        sil = (alpha[0, y0:y1, x0:x1, 0] - m[y0:y1, x0:x1]).abs().mean()
        loss = (
            photo
            + SIL_W * sil
            + 0.01 * torch.sigmoid(params["opacities"]).mean()
            + 0.01 * torch.exp(params["scales"]).mean()
        )
        loss = (
            loss
            + SMOOTH_W * ((delta[1:] - delta[:-1]) ** 2).mean()
            + PRIOR_W * (delta**2).mean()
        )
        if body.deform is not None:
            loss = (
                loss
                + RESID_W
                * (deform(params["means"].detach(), body.time(f)) ** 2).sum(1).mean()
            )
        if body.lrw is not None:
            loss = (
                loss
                + LRSMOOTH_W * ((lrw[1:] - lrw[:-1]) ** 2).mean()
                + DIRS_L2 * (params["dirs"] ** 2).mean()
            )
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, info, lr=LRS["means"])
        for o in list(opts.values()) + extra:
            o.step()
            o.zero_grad(set_to_none=True)
        if it % 100 == 0 and it > 0:
            far = skel.bone_dist(params["means"].detach()).min(1).values > PRUNE_DIST
            params["opacities"].data[far] = -7.0
        if it % 500 == 0:
            print(
                f"it {it} loss {loss.item():.4f} ssim {s.item():.3f} sil {sil.item():.4f} gaussians {len(params['means'])}",
                flush=True,
            )
        if it % 2500 == 0 or it == a.iters - 1:
            progress(
                body, out / f"progress_{it:05d}.png", gt, mk, bg_img, vm, K, frames
            )
            body.save(out / "body.pt")

    with torch.no_grad():
        cam = a.holdout or 19
        athlete, full, n_eval = evaluate(body, cam, gt, mk, bg_img, vm, K, frames)
        r = torch.stack(
            [deform(params["means"], body.time(f)).norm(dim=1) for f in frames[::10]]
        )
        o = torch.stack(
            [
                (params["dirs"].view(-1, K_LR, 3) * lrw[fi][None, :, None])
                .sum(1)
                .norm(dim=1)
                for fi in range(0, n, 10)
            ]
        )
    result = {
        "camera": cam,
        "held_out": bool(a.holdout),
        "holdout_scope": "appearance losses only; supplied poses and alignment are conditioning inputs",
        "evaluation_protocol": theia.evaluation_protocol(a.holdout, frames),
        "frames_scored": n_eval,
        "athlete_psnr_db": round(athlete, 2),
        "full_frame_psnr_db": round(full, 2),
        "gaussians": len(params["means"]),
        "residual_cm": {
            "median": round(r.median().item() * 100, 2),
            "p95": round(r.quantile(0.95).item() * 100, 2),
            "max": round(r.max().item() * 100, 2),
        },
        "lowrank_offset_cm": {
            "median": round(o.median().item() * 100, 2),
            "p95": round(o.quantile(0.95).item() * 100, 2),
            "max": round(o.max().item() * 100, 2),
        },
        "bone_refinement": {
            "rotation_deg_median": round(
                float(torch.rad2deg(delta[:, :, :3].norm(dim=-1).median())), 2
            ),
            "translation_cm_median": round(
                float(delta[:, :, 3:].norm(dim=-1).median()) * 100, 2
            ),
        },
    }
    Path(out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print("BODY_DONE", json.dumps(result), flush=True)
