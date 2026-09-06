"""Minimal 3D Gaussian Splatting trainer on gsplat for a COLMAP-text dataset with known poses and no SfM points.

Designed for the OpenBiomechanics 8-camera OptiTrack rig: few views, known calibration, random init inside the
capture volume. Supports warm-starting from a previous frame's checkpoint so a genlocked video can be fit
frame by frame (sequential 4D reconstruction).

  python train_gs.py --data data/colmap/throw_f0700 --out out/throw_f0700 --iters 4000
  python train_gs.py --data data/colmap/throw_f0706 --out out/throw_f0706 --init out/throw_f0700/ckpt.pt --iters 600

Renders held-out camera(s) (--holdout cam19) for a PSNR check, writes ckpt.pt, and a novel-view orbit mp4.
"""

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy


def read_colmap_txt(d):
    cams = {}
    for line in (d / "sparse" / "0" / "cameras.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        cams[int(p[0])] = {
            "w": int(p[2]),
            "h": int(p[3]),
            "fx": float(p[4]),
            "fy": float(p[5]),
            "cx": float(p[6]),
            "cy": float(p[7]),
        }
    imgs = []
    for line in (d / "sparse" / "0" / "images.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        qw, qx, qy, qz = q
        R = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ]
        )
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        imgs.append({"name": p[9], "cam": cams[int(p[8])], "w2c": w2c})
    return imgs


def load_views(d, holdout, device):
    views = []
    for im in read_colmap_txt(d):
        img = (
            torch.from_numpy(imageio.imread(d / "images" / im["name"])[..., :3])
            .float()
            .div(255)
        )
        c = im["cam"]
        K = torch.tensor(
            [[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]],
            dtype=torch.float32,
        )
        views.append(
            {
                "name": im["name"],
                "img": img.to(device),
                "K": K.to(device),
                "w2c": torch.tensor(im["w2c"], dtype=torch.float32).to(device),
                "w": c["w"],
                "h": c["h"],
                "train": (Path(im["name"]).stem not in holdout),
            }
        )
    return views


def init_gaussians(meta, n, device, seed=0, box=(4.0, 3.5, 4.0), points=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cen = torch.tensor(meta["scene_center_m"])
    r = meta["camera_radius_m"]
    if points:  # monocular-depth unprojection from depth_init.py: positions + colours
        z = np.load(points)
        means = torch.from_numpy(z["xyz"])
        n = len(means)
        sh0 = ((torch.from_numpy(z["rgb"]) - 0.5) / 0.28209479177387814)[:, None, :]
        # per-point scale from the mean distance to the 3 nearest neighbours (chunked)
        d = torch.cat(
            [
                torch.cdist(
                    means[i : i + 4096], means[torch.randperm(n, generator=g)[:20000]]
                )
                .topk(4, largest=False)
                .values[:, 1:]
                .mean(1)
                for i in range(0, n, 4096)
            ]
        )
        scales = torch.log(d.clamp(min=0.01, max=0.3))[:, None].repeat(1, 3)
    else:
        # capture volume: a box around the mound, plus a thin shell of far points for walls/floor
        near = cen + (torch.rand(int(n * 0.8), 3, generator=g) - 0.5) * torch.tensor(
            box
        )
        far = cen + (torch.rand(n - int(n * 0.8), 3, generator=g) - 0.5) * torch.tensor(
            [2.0 * r, 1.2 * r, 2.0 * r]
        )
        means = torch.cat([near, far])
        scales = torch.log(torch.full((n, 3), 0.05))
        sh0 = torch.rand(n, 1, 3) - 0.5
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1
    opac = torch.logit(torch.full((n,), 0.1))
    shN = torch.zeros(n, 15, 3)
    p = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opac,
        "sh0": sh0,
        "shN": shN,
    }
    return torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device)) for k, v in p.items()}
    )


def render(splats, view, sh_degree, bg):
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    out, _alpha, info = rasterization(
        splats["means"],
        F.normalize(splats["quats"], dim=-1),
        torch.exp(splats["scales"]),
        torch.sigmoid(splats["opacities"]),
        colors,
        view["w2c"][None],
        view["K"][None],
        view["w"],
        view["h"],
        sh_degree=sh_degree,
        packed=False,
        absgrad=True,
        backgrounds=bg[None],
        render_mode="RGB",
    )
    return out[0], info


@torch.no_grad()
def visibility_prune(splats, opts, state, views, min_views=3, min_dist=1.5):
    """Drop Gaussians seen by fewer than min_views training cameras or closer than min_dist m to any camera.
    Sparse-view floaters live along single camera rays; anything real in an 8-camera rig is seen from >= 3."""
    m = splats["means"]
    cnt = torch.zeros(len(m), device=m.device)
    near = torch.zeros(len(m), dtype=torch.bool, device=m.device)
    for v in views:
        R, t = v["w2c"][:3, :3], v["w2c"][:3, 3]
        p = m @ R.T + t
        z = p[:, 2]
        u = v["K"][0, 0] * p[:, 0] / z.clamp(min=1e-6) + v["K"][0, 2]
        w = v["K"][1, 1] * p[:, 1] / z.clamp(min=1e-6) + v["K"][1, 2]
        cnt += ((z > 0) & (u >= 0) & (u < v["w"]) & (w >= 0) & (w < v["h"])).float()
        near |= p.norm(dim=1) < min_dist
    keep = (cnt >= min_views) & ~near
    return keep_subset(splats, opts, state, keep)


@torch.no_grad()
def keep_subset(splats, opts, state, keep):
    """Slice every parameter, its Adam state and the strategy state to a boolean keep mask."""
    if keep.all():
        return 0
    for k in splats:
        splats[k] = torch.nn.Parameter(splats[k][keep])
        o = opts[k]
        g = o.param_groups[0]
        old = g["params"][0]
        st = o.state.pop(old, None)
        g["params"] = [splats[k]]
        if st:
            for kk in ("exp_avg", "exp_avg_sq"):
                if kk in st:
                    st[kk] = st[kk][keep]
            o.state[splats[k]] = st
    for kk in list(state.keys()):
        if torch.is_tensor(state[kk]) and state[kk].shape[0] == len(keep):
            state[kk] = state[kk][keep]
    return int((~keep).sum())


def psnr(a, b):
    return -10 * torch.log10(F.mse_loss(a, b))


def rig_geometry(views):
    """Target = least-squares intersection of the cameras' optical axes; up = -(mean camera y-axis in world);
    radius/height = the real cameras' mean distance and elevation, so novel views stay where real ones are valid."""
    dev = views[0]["w2c"].device
    Cs, ds, ys = [], [], []
    for v in views:
        R = v["w2c"][:3, :3]
        C = -R.T @ v["w2c"][:3, 3]
        Cs.append(C)
        ds.append(R.T @ torch.tensor([0.0, 0.0, 1.0], device=dev))
        ys.append(R.T @ torch.tensor([0.0, 1.0, 0.0], device=dev))
    Cs, ds = torch.stack(Cs), torch.stack(ds)
    A = torch.zeros(3, 3, device=dev)
    b = torch.zeros(3, device=dev)
    for C, d in zip(Cs, ds):
        P = torch.eye(3, device=dev) - d[:, None] * d[None, :]
        A += P
        b += P @ C
    target = torch.linalg.solve(A, b)
    up = F.normalize(-torch.stack(ys).mean(0), dim=0)
    rel = Cs - target
    height = (rel @ up).mean()
    radius = (rel - (rel @ up)[:, None] * up).norm(dim=1).mean()
    return target, up, float(radius), float(height)


def look_at(eye, target, up):
    f = F.normalize(target - eye, dim=0)
    r = F.normalize(torch.linalg.cross(f, up), dim=0)
    u = torch.linalg.cross(r, f)
    R = torch.stack([r, -u, f])
    w2c = torch.eye(4, device=eye.device)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def orbit_video(
    splats, views, meta, out, n=90, sh_degree=3, height_scale=0.8, radius_scale=0.9
):
    target, up, radius, height = rig_geometry(views)
    ref = views[0]
    dev = target.device
    e1 = F.normalize(
        torch.linalg.cross(up, torch.tensor([1.0, 0.0, 0.0], device=dev)), dim=0
    )
    e2 = torch.linalg.cross(up, e1)
    frames = []
    with torch.no_grad():
        for i in range(n):
            a = 2 * math.pi * i / n
            eye = (
                target
                + radius_scale * radius * (math.cos(a) * e1 + math.sin(a) * e2)
                + height_scale * height * up
            )
            v = dict(ref)
            v["w2c"] = look_at(eye, target, up)
            img, _ = render(splats, v, sh_degree, torch.zeros(3, device=dev))
            frames.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimwrite(out, frames, fps=30, quality=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--init", default="")
    ap.add_argument("--n_init", type=int, default=100_000)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--orbit", type=int, default=0)
    ap.add_argument(
        "--vis_prune",
        type=int,
        default=3,
        help="prune Gaussians seen by fewer than this many training cameras (0 = off)",
    )
    ap.add_argument("--min_dist", type=float, default=1.5)
    ap.add_argument(
        "--init_box",
        type=float,
        nargs=3,
        default=[4.0, 3.5, 4.0],
        help="init volume (m) around the scene center",
    )
    ap.add_argument("--init_points", default="", help="points.npz from depth_init.py")
    a = ap.parse_args()
    dev = "cuda"
    d = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(d / "meta.json").read_text())
    holdout = set(a.holdout.split(",")) - {""}
    views = load_views(d, holdout, dev)
    train = [v for v in views if v["train"]]
    test = [v for v in views if not v["train"]]
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in ck["splats"].items()}
        )
    else:
        splats = init_gaussians(
            meta, a.n_init, dev, box=tuple(a.init_box), points=a.init_points or None
        )
    lr = {
        "means": 1.6e-4 * meta["camera_radius_m"],
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    opts = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr[k], "name": k}], eps=1e-15)
        for k in splats
    }
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=500,
        refine_stop_iter=int(a.iters * 0.7),
        reset_every=3000,
        refine_every=100,
        absgrad=True,
        grow_grad2d=0.0006,
        prune_opa=0.005,
    )
    if a.init:
        strategy = DefaultStrategy(
            verbose=False,
            refine_start_iter=50,
            refine_stop_iter=int(a.iters * 0.6),
            reset_every=10**9,
            refine_every=50,
            absgrad=True,
            grow_grad2d=0.0008,
            prune_opa=0.005,
        )
    state = strategy.initialize_state(scene_scale=meta["camera_radius_m"])
    bg = torch.zeros(3, device=dev)
    t0 = time.time()
    n_pruned = 0
    for it in range(a.iters):
        v = train[np.random.randint(len(train))]
        sh = min(it // 500, a.sh_degree) if not a.init else a.sh_degree
        img, info = render(splats, v, sh, bg)
        strategy.step_pre_backward(splats, opts, state, it, info)
        l1 = (img - v["img"]).abs().mean()
        loss = l1
        loss.backward()
        strategy.step_post_backward(splats, opts, state, it, info, packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if a.vis_prune and it % 100 == 0 and it > 0:
            n_pruned = visibility_prune(
                splats, opts, state, train, min_views=a.vis_prune, min_dist=a.min_dist
            )
        if it % 500 == 0 or it == a.iters - 1:
            with torch.no_grad():
                ps = [
                    psnr(
                        render(splats, tv, a.sh_degree, bg)[0].clamp(0, 1), tv["img"]
                    ).item()
                    for tv in test
                ]
            print(
                f"it {it:5d} loss {loss.item():.4f} pruned {n_pruned if a.vis_prune else 0} n_gauss {len(splats['means']):7d} holdout_psnr {np.mean(ps) if ps else float('nan'):.2f} {time.time() - t0:.0f}s",
                flush=True,
            )
    with torch.no_grad():
        for v in views:
            img, _ = render(splats, v, a.sh_degree, bg)
            side = torch.cat([v["img"], img.clamp(0, 1)], 1)
            imageio.imwrite(
                out
                / f"{'test' if not v['train'] else 'train'}_{Path(v['name']).stem}.jpg",
                (side.cpu().numpy() * 255).astype(np.uint8),
                quality=90,
            )
        torch.save(
            {"splats": {k: v.detach() for k, v in splats.items()}, "meta": meta},
            out / "ckpt.pt",
        )
        res = {
            "iters": a.iters,
            "n_gaussians": len(splats["means"]),
            "holdout": sorted(holdout),
            "holdout_psnr": {
                Path(tv["name"]).stem: psnr(
                    render(splats, tv, a.sh_degree, bg)[0].clamp(0, 1), tv["img"]
                ).item()
                for tv in test
            },
            "train_psnr": {
                Path(tv["name"]).stem: psnr(
                    render(splats, tv, a.sh_degree, bg)[0].clamp(0, 1), tv["img"]
                ).item()
                for tv in train
            },
            "seconds": time.time() - t0,
        }
        Path(out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        if a.orbit:
            orbit_video(
                splats, views, meta, out / "orbit.mp4", n=a.orbit, sh_degree=a.sh_degree
            )


if __name__ == "__main__":
    main()
