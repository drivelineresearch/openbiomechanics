"""V8: SMPL-X geometry + free Gaussians (the combination of V6 and V7).

The athlete's Gaussians are NOT the mesh vertices. They start where the images say the athlete is (the DUSt3R
point cloud of the reference frame, silhouette-carved, exactly like V6) and each one is bound to the nearest
SMPL-X triangle, storing its position in that triangle's local frame. Every frame, the triangle's frame comes
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from the fitted SMPL-X mesh, so the Gaussians inherit the body model's articulation, including hands, feet and
head, while keeping this athlete's real silhouette.

  python train_bind.py --frames data/throw_full/t0* --ref data/throw_full/t0150 --bg out/v6_bg/ckpt.pt \
      --smplx out/v7/ckpt.pt --model data/smplx/models/smplx/SMPLX_NEUTRAL.npz \
      --init_points data/throw_full/t0150/points_dust3r.npz --out out/v8 --iters 14000 --holdout cam19
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from obp_splat.core import load_views, look_at, psnr, render, rig_geometry
from obp_splat.train_background import load_masks, points_split
from train_smplx import mat_to_quat


def tri_frames(V, faces):
    """rigid frame + centroid per triangle from vertex positions. V: (...,Nv,3) -> R (...,F,3,3), c (...,F,3)"""
    a, b, c = V[..., faces[:, 0], :], V[..., faces[:, 1], :], V[..., faces[:, 2], :]
    e0 = b - a; e1 = c - a
    n = F.normalize(torch.linalg.cross(e0, e1, dim=-1), dim=-1)
    t = F.normalize(e0, dim=-1)
    u = torch.linalg.cross(n, t, dim=-1)
    return torch.stack([t, u, n], -1), (a + b + c) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--bg", required=True)
    ap.add_argument("--smplx", required=True); ap.add_argument("--model", required=True); ap.add_argument("--init_points", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--iters", type=int, default=14000); ap.add_argument("--holdout", default="")
    ap.add_argument("--n_gauss", type=int, default=120000); ap.add_argument("--orbit_deg", type=float, default=160.0); ap.add_argument("--tag", default="v8")
    a = ap.parse_args(); dev = "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); hold = set(a.holdout.split(",")) - {""}
    frames = [Path(f) for f in a.frames]; T = len(frames); ref_i = [i for i, f in enumerate(frames) if f.resolve() == Path(a.ref).resolve()][0]
    ck = torch.load(a.smplx, map_location="cpu"); Vp = ck["Vp"].to(dev)                    # (T,Nv,3) fitted SMPL-X vertices
    import smplx
    faces = torch.from_numpy(smplx.create(str(Path(a.model).parents[1]), model_type="smplx", gender="neutral", use_pca=False, batch_size=1).faces.astype(np.int64)).to(dev)

    # views (JPEG in RAM, one frame at a time)
    views = []
    for f in frames:
        vs = load_masks(f, load_views(f, hold, "cpu")); views.append(vs)
        for v in vs:
            v["jpg"] = cv2.imencode(".jpg", cv2.cvtColor((v["img"] * 255).round().to(torch.uint8).numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]; del v["img"]
            v["mask_np"] = np.packbits(v["mask"].numpy()); v["mask_shape"] = tuple(v["mask"].shape); del v["mask"]
            v["K"] = v["K"].to(dev); v["w2c"] = v["w2c"].to(dev)
    dec = lambda b: torch.from_numpy(cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    get_img = lambda v: dec(v["jpg"]).to(dev).float() / 255
    get_mask = lambda v: torch.from_numpy(np.unpackbits(v["mask_np"])[: v["mask_shape"][0] * v["mask_shape"][1]].reshape(v["mask_shape"]).astype(bool)).to(dev)
    bg = torch.load(a.bg, map_location=dev)["splats"]; bgc = torch.zeros(3, device=dev)
    with torch.no_grad():
        for vs in views:
            for v in vs:
                if v["train"]: v["bg_jpg"] = cv2.imencode(".jpg", cv2.cvtColor((render(bg, v, 3, bgc)[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]

    # free Gaussians from the images, carved by the reference frame's silhouettes (V6 behaviour)
    ref_views = [dict(v, mask=get_mask(v)) for v in views[ref_i] if v["train"]]
    xyz, rgb = points_split(a.init_points, ref_views, inside=True)
    if len(xyz) > a.n_gauss:
        sel = torch.randperm(len(xyz), device=dev)[:a.n_gauss]; xyz, rgb = xyz[sel], rgb[sel]
    Rref, cref = tri_frames(Vp[ref_i], faces)                                              # (F,3,3), (F,3)
    d2 = torch.cat([torch.cdist(xyz[i:i + 4096], cref).min(1).indices for i in range(0, len(xyz), 4096)])
    keep = (xyz - cref[d2]).norm(dim=1) < 0.25                                             # drop points far from the body
    xyz, rgb, tri = xyz[keep], rgb[keep], d2[keep]
    local = torch.einsum("nij,nj->ni", Rref[tri].transpose(1, 2), xyz - cref[tri])         # position in the triangle's frame
    qref = mat_to_quat(Rref)[tri]
    n = len(xyz); print(f"bound {n} gaussians to {len(faces)} SMPL-X triangles (median distance {(xyz - cref[tri]).norm(dim=1).median()*100:.1f} cm)", flush=True)

    g = torch.nn.ParameterDict(dict(local=torch.nn.Parameter(local.clone()), scales=torch.nn.Parameter(torch.log(torch.full((n, 3), 0.015, device=dev))),
                                    quats=torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1)),
                                    opacities=torch.nn.Parameter(torch.logit(torch.full((n,), 0.3, device=dev))),
                                    sh0=torch.nn.Parameter(((rgb - 0.5) / 0.28209479177387814)[:, None, :]), shN=torch.nn.Parameter(torch.zeros(n, 15, 3, device=dev))))
    cams = sorted({Path(v["name"]).stem for vs in views for v in vs}); cidx = {c: i for i, c in enumerate(cams)}
    color = torch.nn.Parameter(torch.zeros(len(cams), 2, 3, device=dev))
    calib = lambda img, v: img * torch.exp(color[cidx[Path(v["name"]).stem]][0]) + color[cidx[Path(v["name"]).stem]][1]
    lr = dict(local=2e-4, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=1.25e-4)
    opt = torch.optim.Adam([{"params": g[k], "lr": lr[k]} for k in g] + [{"params": color, "lr": 1e-3}], eps=1e-15)
    from pytorch_msssim import ssim as ssim_fn

    # cache the per-frame triangle frames for the bound triangles only (memory: T x n x 3 x 3 would be huge, so recompute)
    def posed(t):
        R, c = tri_frames(Vp[t], faces)
        means = c[tri] + torch.einsum("nij,nj->ni", R[tri], g["local"])
        qf = mat_to_quat(R[tri])
        # q_total = (qf * qref^-1) * q_local
        w1, x1, y1, z1 = qf.unbind(-1); w2, x2, y2, z2 = (qref * torch.tensor([1.0, -1, -1, -1], device=dev)).unbind(-1)
        dq = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
        q = F.normalize(g["quats"], dim=-1)
        w1, x1, y1, z1 = dq.unbind(-1); w2, x2, y2, z2 = q.unbind(-1)
        quats = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
        return means, quats

    train_pairs = [(t, v) for t, vs in enumerate(views) for v in vs if v["train"]]; t0 = time.time()
    for it in range(a.iters):
        t, v = train_pairs[np.random.randint(len(train_pairs))]; sh = min(it // 1500, 3)
        means, quats = posed(t)
        rgbr, alpha, _ = rasterization(means, F.normalize(quats, dim=-1), torch.exp(g["scales"]), torch.sigmoid(g["opacities"]), torch.cat([g["sh0"], g["shN"]], 1),
                                       v["w2c"][None], v["K"][None], v["w"], v["h"], sh_degree=sh, packed=False, backgrounds=bgc[None], render_mode="RGB")
        rgbr, alpha = rgbr[0], alpha[0, ..., 0]
        gt = get_img(v); mk = get_mask(v)
        img = calib(rgbr + (1 - alpha)[..., None] * (dec(v["bg_jpg"]).to(dev).float() / 255), v)
        l1 = (img - gt).abs().mean(); ss = 1 - ssim_fn(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0)
        loss = 0.8 * l1 + 0.2 * ss + 10.0 * (alpha * (~mk).float()).mean() + 0.5 * ((1 - alpha) * mk.float()).mean() + 1e-2 * ((g["local"] - local) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): g["scales"].clamp_(max=math.log(0.06))
        if it % 1000 == 0 or it == a.iters - 1:
            print(f"it {it:5d} loss {loss.item():.4f} |drift| {(g['local'] - local).norm(dim=1).mean().item()*100:.2f}cm {time.time()-t0:.0f}s", flush=True)

    res = {}; hold_frames = []; stage_frames = []
    with torch.no_grad():
        target_pt, up, radius, height = rig_geometry(views[0]); e1 = F.normalize(torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=dev)), dim=0); e2 = torch.linalg.cross(up, e1)
        keepbg = (torch.sigmoid(bg["opacities"]) >= 0.05) & ((bg["means"] - target_pt).norm(dim=1) <= 3.5); bgk = {k: v[keepbg] for k, v in bg.items()}
        for t, vs in enumerate(views):
            means, quats = posed(t)
            sp = dict(means=means, quats=quats, scales=g["scales"], opacities=g["opacities"], sh0=g["sh0"], shN=g["shN"])
            full = {k: torch.cat([bg[k], sp[k]], 0) for k in sp}
            for v in vs:
                if v["train"]: continue
                img = calib(render(full, v, 3, bgc)[0], v).clamp(0, 1); gt = get_img(v); m = get_mask(v)
                res[frames[t].name] = dict(psnr=psnr(img, gt).item(), athlete_psnr=psnr(img[m], gt[m]).item() if m.any() else float("nan"))
                fr = np.ascontiguousarray((torch.cat([gt, img], 1).cpu().numpy() * 255).astype(np.uint8))
                cv2.putText(fr, "real cam19  (held out)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(fr, f"{a.tag}: gaussians on the SMPL-X mesh   {res[frames[t].name]['psnr']:.1f} dB", (1220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                hold_frames.append(fr)
            ang = math.radians(-a.orbit_deg / 2 + a.orbit_deg * t / max(T - 1, 1)); ref = vs[0]
            eye = target_pt + 0.85 * radius * (math.cos(ang) * e1 + math.sin(ang) * e2) + 0.85 * height * up; v1 = dict(ref); v1["w2c"] = look_at(eye, target_pt, up)
            eye2 = target_pt + 0.8 * radius * (math.cos(1.4) * e1 + math.sin(1.4) * e2) + 0.75 * height * up; v2 = dict(ref); v2["w2c"] = look_at(eye2, target_pt, up)
            fullk = {k: torch.cat([bgk[k], sp[k]], 0) for k in sp}
            im1 = render(fullk, v1, 3, torch.full((3,), 0.10, device=dev))[0].clamp(0, 1); im2 = render(fullk, v2, 3, torch.full((3,), 0.10, device=dev))[0].clamp(0, 1)
            stage_frames.append((torch.cat([im1, im2], 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimwrite(out / "holdout_cam19.mp4", hold_frames, fps=30, quality=8)
    imageio.mimwrite(out / "stage.mp4", stage_frames, fps=30, quality=8)
    torch.save(dict(g={k: v.detach() for k, v in g.items()}, tri=tri.cpu(), qref=qref.cpu(), smplx=a.smplx, bg=a.bg,
                    color={c: color[i].detach().cpu().tolist() for c, i in cidx.items()}), out / "ckpt.pt")
    json.dump(res, open(out / "result.json", "w"), indent=1)
    vals = [r["psnr"] for r in res.values()]; ath = [r["athlete_psnr"] for r in res.values()]
    print(json.dumps(dict(holdout_psnr_mean=float(np.mean(vals)), athlete_psnr_mean=float(np.nanmean(ath)), n_frames=len(vals), n_gauss=n)))


if __name__ == "__main__":
    main()
