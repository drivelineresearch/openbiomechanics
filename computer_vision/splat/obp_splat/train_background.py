"""V2: static background fitted once (athlete masked out, several frames), then athlete-only dynamic Gaussians
per frame composited over the frozen background. Foreground Gaussians are penalised for any opacity outside
the athlete mask, so they cannot drift into the background.

  # background from several frames of the sequence (masks from masks.py)
  python train_fgbg.py bg --frames data/throw_seq/t0000 data/throw_seq/t0010 data/throw_seq/t0020 data/throw_seq/t0030 \
        --init_points data/throw_seq/t0000/points_dust3r.npz --out out/v2_bg --iters 6000 --holdout cam19
  # athlete for one frame
  python train_fgbg.py fg --data data/throw_f0700 --bg out/v2_bg/ckpt.pt --init_points data/throw_f0700/points_dust3r.npz \
        --out out/v2_f0700 --iters 3000 --holdout cam19 --orbit 90
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
from gsplat.strategy import DefaultStrategy

from .core import (
    keep_subset,
    load_views,
    orbit_video,
    psnr,
    render,
    rig_geometry,
    visibility_prune,
)


def load_masks(d, views):
    for v in views:
        p = Path(d) / "masks" / (Path(v["name"]).stem + ".png")
        m = (
            torch.from_numpy(imageio.imread(p)).float().div(255)
            if p.exists()
            else torch.zeros(v["h"], v["w"])
        )
        v["mask"] = (m > 0.5).to(v["img"].device)
    return views


def points_split(npz, views, inside):
    """Points from DUSt3R that project inside (or outside) the athlete mask; inside needs >= 2 views, outside needs none."""
    z = np.load(npz)
    xyz = torch.from_numpy(z["xyz"]).cuda()
    rgb = torch.from_numpy(z["rgb"]).cuda()
    cnt = torch.zeros(len(xyz), device="cuda")
    seen = torch.zeros(len(xyz), device="cuda")
    for v in views:
        R, t = v["w2c"][:3, :3], v["w2c"][:3, 3]
        p = xyz @ R.T + t
        zc = p[:, 2].clamp(min=1e-6)
        u = (v["K"][0, 0] * p[:, 0] / zc + v["K"][0, 2]).long()
        w = (v["K"][1, 1] * p[:, 1] / zc + v["K"][1, 2]).long()
        ok = (p[:, 2] > 0) & (u >= 0) & (u < v["w"]) & (w >= 0) & (w < v["h"])
        m = torch.zeros(len(xyz), dtype=torch.bool, device="cuda")
        m[ok] = v["mask"][w[ok], u[ok]]
        cnt += m.float()
        seen += ok.float()
    keep = (cnt >= 2) if inside else ((cnt == 0) & (seen >= 1))
    return xyz[keep], rgb[keep]


def make_splats(xyz, rgb, scale=None):
    n = len(xyz)
    g = torch.Generator(device="cpu").manual_seed(0)
    if scale is None:
        d = torch.cat(
            [
                torch.cdist(
                    xyz[i : i + 4096],
                    xyz[torch.randperm(n, generator=g)[:20000].cuda()],
                )
                .topk(4, largest=False)
                .values[:, 1:]
                .mean(1)
                for i in range(0, n, 4096)
            ]
        )
        scale = d.clamp(min=0.005, max=0.3)
    p = {
        "means": xyz.clone(),
        "scales": torch.log(scale)[:, None].repeat(1, 3),
        "quats": torch.tensor([1.0, 0, 0, 0], device="cuda").repeat(n, 1),
        "opacities": torch.logit(torch.full((n,), 0.1, device="cuda")),
        "sh0": ((rgb - 0.5) / 0.28209479177387814)[:, None, :],
        "shN": torch.zeros(n, 15, 3, device="cuda"),
    }
    return torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.contiguous()) for k, v in p.items()}
    )


def combined(fg, bg):
    if bg is None:
        return fg
    return {k: torch.cat([bg[k].detach(), fg[k]], 0) for k in fg}


def optimizers(splats, scene_scale):
    lr = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    return {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr[k], "name": k}], eps=1e-15)
        for k in splats
    }


def evaluate(fg, bg, views, out, tag):
    res = {}
    with torch.no_grad():
        for v in views:
            img, _ = render(combined(fg, bg), v, 3, torch.zeros(3, device="cuda"))
            img = img.clamp(0, 1)
            m = v["mask"]
            full = psnr(img, v["img"]).item()
            ath = psnr(img[m], v["img"][m]).item() if m.any() else float("nan")
            res[Path(v["name"]).stem] = {
                "psnr": full,
                "athlete_psnr": ath,
                "train": v["train"],
            }
            side = torch.cat([v["img"], img], 1)
            imageio.imwrite(
                out
                / f"{tag}_{'train' if v['train'] else 'test'}_{Path(v['name']).stem}.jpg",
                (side.cpu().numpy() * 255).astype(np.uint8),
                quality=90,
            )
    return res


def se3_exp(d):
    """small-angle SE(3) exponential for a (6,) tensor [rot, trans] -> (4,4)."""
    w, u = d[:3], d[3:]
    th = w.norm() + 1e-9
    k = w / th
    z = torch.zeros_like(th)
    K = torch.stack(
        [
            torch.stack([z, -k[2], k[1]]),
            torch.stack([k[2], z, -k[0]]),
            torch.stack([-k[1], k[0], z]),
        ]
    )
    R = torch.eye(3, device=d.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K
    T = torch.eye(4, device=d.device)
    T[:3, :3] = R
    T[:3, 3] = u
    return T


def view_w2c(v, pose):
    return (se3_exp(pose[v["idx"]]) @ v["w2c"]) if pose is not None else v["w2c"]


def render_fg(fg, v, sh, bgc, pose=None, depth=False):
    from gsplat import rasterization

    out, alpha, info = rasterization(
        fg["means"],
        F.normalize(fg["quats"], dim=-1),
        torch.exp(fg["scales"]),
        torch.sigmoid(fg["opacities"]),
        torch.cat([fg["sh0"], fg["shN"]], 1),
        view_w2c(v, pose)[None],
        v["K"][None],
        v["w"],
        v["h"],
        sh_degree=sh,
        packed=False,
        absgrad=True,
        backgrounds=bgc[None],
        render_mode="RGB+ED" if depth else "RGB",
    )
    return out[0, ..., :3], alpha[0, ..., 0], info, (out[0, ..., 3] if depth else None)


def depth_loss(rend, ref, mask):
    """scale-and-shift-invariant (Pearson) depth loss on masked pixels with a reference depth."""
    m = mask & (ref > 0) & (rend > 0)
    if m.sum() < 500:
        return torch.zeros((), device=rend.device)
    a, b = rend[m], ref[m]
    a = a - a.mean()
    b = b - b.mean()
    return 1 - (a * b).sum() / (a.norm() * b.norm() + 1e-6)


def load_depths(npz, views):
    z = np.load(npz)
    for v in views:
        k = f"depth_{Path(v['name']).stem}"
        if k in z:
            d = torch.from_numpy(z[k]).cuda()[None, None]
            v["depth"] = F.interpolate(
                d, size=(v["h"], v["w"]), mode="bilinear", align_corners=False
            )[0, 0]
    return views


FG_GROW, FG_MIN_VIEWS = None, 2


def train(
    fg,
    bg,
    views,
    iters,
    out,
    scene_scale,
    warm,
    mask_lambda=10.0,
    log_every=100,
    depth_lambda=0.0,
    refine_pose=False,
):
    """bg None: fit the background with athlete pixels ignored. bg given: fit the athlete composited OVER the frozen
    background (bg rendered once per training view and cached; the athlete occludes everything in every camera)."""
    train_v = [v for v in views if v["train"]]
    opts = optimizers(fg, scene_scale)
    bgc = torch.zeros(3, device="cuda")
    for i, v in enumerate(views):
        v["idx"] = i
    cam_names = sorted({Path(v["name"]).stem for v in views})
    cam_idx = {c: i for i, c in enumerate(cam_names)}
    color = torch.nn.Parameter(torch.zeros(len(cam_names), 2, 3, device="cuda"))
    color_opt = torch.optim.Adam([color], lr=1e-3)

    def calib(img, v):
        c = color[cam_idx[Path(v["name"]).stem]]
        return img * torch.exp(c[0]) + c[1]

    pose = (
        torch.nn.Parameter(torch.zeros(len(views), 6, device="cuda"))
        if refine_pose
        else None
    )
    pose_opt = torch.optim.Adam([pose], lr=1e-4) if refine_pose else None
    if bg is not None:
        with torch.no_grad():
            for v in train_v:
                v["bg_rgb"] = render(bg, v, 3, bgc)[0].clamp(0, 1)
    fg_stage = bg is not None
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=50 if warm else 300,
        refine_stop_iter=int(iters * 0.6),
        reset_every=10**9 if warm else 3000,
        refine_every=200 if fg_stage else 100,
        absgrad=True,
        grow_grad2d=0.002 if fg_stage else 0.0006,
        prune_opa=0.005,
    )
    cap = 250_000 if fg_stage else 600_000
    if fg_stage and FG_GROW is not None:
        strategy.grow_grad2d = FG_GROW
    state = strategy.initialize_state(scene_scale=scene_scale)
    t0 = time.time()
    for it in range(iters):
        v = train_v[np.random.randint(len(train_v))]
        sh = min(it // 300, 3)
        use_depth = depth_lambda > 0 and "depth" in v
        rgb, alpha, info, dep = render_fg(fg, v, sh, bgc, pose, depth=use_depth)
        strategy.step_pre_backward(fg, opts, state, it, info)
        if bg is None:
            loss = (
                (calib(rgb, v) - v["img"]).abs() * (~v["mask"]).float()[..., None]
            ).mean()
            if use_depth:
                loss = loss + depth_lambda * depth_loss(dep, v["depth"], ~v["mask"])
        else:
            img = rgb + (1 - alpha)[..., None] * v["bg_rgb"]
            loss = (img - v["img"]).abs().mean() + mask_lambda * (
                alpha * (~v["mask"]).float()
            ).mean()
            if use_depth:
                loss = loss + depth_lambda * depth_loss(
                    dep, v["depth"], v["mask"] & (alpha > 0.5)
                )
        loss.backward()
        strategy.step_post_backward(fg, opts, state, it, info, packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if bg is None and it > 300:
            color_opt.step()
        color_opt.zero_grad(set_to_none=True)
        if refine_pose:
            if it > 200:
                pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
        if it % 100 == 0 and it > 0:
            visibility_prune(
                fg,
                opts,
                state,
                train_v,
                min_views=3 if bg is None else FG_MIN_VIEWS,
                min_dist=1.5,
            )
            if len(fg["means"]) > cap:  # hard cap: keep the most opaque
                thr = torch.sigmoid(fg["opacities"]).topk(cap).values[-1]
                keep_subset(fg, opts, state, torch.sigmoid(fg["opacities"]) >= thr)
        if it % log_every == 0 or it == iters - 1:
            extra = (
                f" pose |rot| {pose[:, :3].norm(dim=1).mean().item() * 57.3:.3f} deg |t| {pose[:, 3:].norm(dim=1).mean().item() * 100:.1f} cm"
                if refine_pose
                else ""
            )
            print(
                f"it {it:5d} loss {loss.item():.4f} n {len(fg['means']):7d} {time.time() - t0:.0f}s{extra}",
                flush=True,
            )
    if refine_pose:
        with torch.no_grad():
            for v in views:
                v["w2c"] = view_w2c(v, pose).detach()
    if bg is None:
        train.color = {c: color[i].detach().cpu().tolist() for c, i in cam_idx.items()}
    return fg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bg", "fg"])
    ap.add_argument("--frames", nargs="*", default=[])
    ap.add_argument("--data", default="")
    ap.add_argument("--bg", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--init_points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--orbit", type=int, default=0)
    ap.add_argument("--depth_lambda", type=float, default=0.0)
    ap.add_argument("--refine_pose", action="store_true")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--fg_grow", type=float, default=None)
    ap.add_argument("--fg_min_views", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hold = set(a.holdout.split(",")) - {""}
    global FG_GROW, FG_MIN_VIEWS
    FG_GROW, FG_MIN_VIEWS = a.fg_grow, a.fg_min_views
    if a.stage == "bg":
        views = []
        for d in a.frames:
            views += load_masks(d, load_views(Path(d), hold, "cuda"))
        meta = json.loads(Path(Path(a.frames[0]) / "meta.json").read_text())
        xyz, rgb = points_split(
            a.init_points, [v for v in views if v["train"]], inside=False
        )
        print("background init points", len(xyz), flush=True)
        if a.depth_lambda > 0:
            for d, chunk in zip(
                a.frames, [views[i : i + 8] for i in range(0, len(views), 8)]
            ):
                p = Path(d) / "points_dust3r.npz"
                if p.exists():
                    load_depths(p, chunk)
        bg = train(
            make_splats(xyz, rgb),
            None,
            views,
            a.iters,
            out,
            meta["camera_radius_m"],
            warm=False,
            depth_lambda=a.depth_lambda,
            refine_pose=a.refine_pose,
        )
        if a.refine_pose:
            Path(out / "refined_poses.json").write_text(
                json.dumps(
                    {Path(v["name"]).stem: v["w2c"].cpu().tolist() for v in views[:8]}
                )
                + "\n"
            )
        torch.save(
            {
                "splats": {k: v.detach() for k, v in bg.items()},
                "meta": meta,
                "color": getattr(train, "color", None),
            },
            out / "ckpt.pt",
        )
        print(
            "per-camera exposure gains:",
            {c: [round(math.exp(g), 3) for g in v[0]] for c, v in train.color.items()},
            flush=True,
        )
        res = evaluate(bg, None, views[:8], out, "bg")
        Path(out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: round(v["psnr"], 2) for k, v in res.items()}))
    else:
        views = load_masks(a.data, load_views(Path(a.data), hold, "cuda"))
        meta = json.loads(Path(Path(a.data) / "meta.json").read_text())
        bg = torch.load(a.bg, map_location="cuda")["splats"]
        if a.init:
            ck = torch.load(a.init, map_location="cuda")
            fg = torch.nn.ParameterDict(
                {k: torch.nn.Parameter(v) for k, v in ck["splats"].items()}
            )
        else:
            xyz, rgb = points_split(
                a.init_points, [v for v in views if v["train"]], inside=True
            )
            print("athlete init points", len(xyz), flush=True)
            if (
                len(xyz) < 2000
            ):  # fall back to a random cloud in the athlete's column above the mound
                tgt, up, _, _ = rig_geometry(views)
                xyz = tgt + (torch.rand(20000, 3, device="cuda") - 0.5) * 1.5 + up * 0.9
                rgb = torch.rand(len(xyz), 3, device="cuda")
            fg = make_splats(
                xyz, rgb, scale=torch.full((len(xyz),), 0.02, device="cuda")
            )
        if a.depth_lambda > 0:
            load_depths(a.init_points, views)
        rp_path = Path(a.bg).parent / "refined_poses.json"
        if rp_path.exists():  # reuse the background stage's refined cameras
            rp = json.loads(Path(rp_path).read_text())
            for v in views:
                if Path(v["name"]).stem in rp:
                    v["w2c"] = torch.tensor(rp[Path(v["name"]).stem], device="cuda")
        fg = train(
            fg,
            bg,
            views,
            a.iters,
            out,
            meta["camera_radius_m"],
            warm=bool(a.init),
            depth_lambda=a.depth_lambda,
            refine_pose=False,
        )
        torch.save(
            {
                "splats": {k: v.detach() for k, v in fg.items()},
                "meta": meta,
                "bg": a.bg,
            },
            out / "ckpt.pt",
        )
        res = evaluate(fg, bg, views, out, a.tag)
        Path(out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        print(
            json.dumps(
                {
                    k: (round(v["psnr"], 2), round(v["athlete_psnr"], 2))
                    for k, v in res.items()
                }
            )
        )
        if a.orbit:
            orbit_video(combined(fg, bg), views, meta, out / "orbit.mp4", n=a.orbit)


if __name__ == "__main__":
    main()
