"""V4: skeleton-anchored athlete. One canonical set of Gaussians lives on the athlete's body in a reference frame
and is posed to every other frame by linear blend skinning driven by the triangulated 3D skeleton
(pose3d.py). All frames are trained jointly, composited over the frozen V2/V3 background. This is the
poor-man's version of the SMPL-X-anchored avatars (GauHuman etc.) that needs no licensed body model:
bones are the 14 COCO limb segments plus a torso, and skinning weights come from distance to the bones.

  python train_skel.py --frames data/throw_seq/t* --ref data/throw_seq/t0015 --bg out/v2_bg/ckpt.pt \
        --init_points data/throw_seq/t0015/points_dust3r.npz --out out/v4 --iters 8000 --holdout cam19
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

from .core import load_views, look_at, psnr, render, rig_geometry
from .train_background import load_masks, points_split

# 17 COCO joints + 8 virtual mid-points (indices 17..24) so long limbs get two bones each: smoother skin at elbows/knees
MID = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
BONES = [(5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6), (1, 2), (3, 4),
         (5, 17), (17, 7), (7, 18), (18, 9), (6, 19), (19, 8), (8, 20), (20, 10),
         (11, 21), (21, 13), (13, 22), (22, 15), (12, 23), (23, 14), (14, 24), (24, 16)]


EXTRA_BONES = [(15, 18), (16, 19), (0, 17), (9, 9), (10, 10)]   # ankle->toes, nose->head (indices into the extended joint set)


def with_mids(J):
    """append virtual mid-joints (indices grow after any extra joints)"""
    mids = torch.stack([(J[..., a, :] + J[..., b, :]) / 2 for a, b in MID], -2)
    return torch.cat([J, mids], -2)


def load_skel(d):
    s = json.load(open(Path(d) / "skeleton.json")); J = np.array([j if j is not None else [np.nan] * 3 for j in s["joints"]], dtype=np.float32)
    return torch.from_numpy(J)


def fill_missing(J_all):
    """fill missing joints per frame by linear interpolation over time, then hold."""
    T, N, _ = J_all.shape
    for j in range(N):
        ok = ~torch.isnan(J_all[:, j, 0]); idx = torch.arange(T)
        if ok.sum() == 0: J_all[:, j] = 0
        elif ok.sum() < T:
            for c in range(3): J_all[:, j, c] = torch.from_numpy(np.interp(idx.numpy(), idx[ok].numpy(), J_all[ok, j, c].numpy()))
    return J_all


def bone_frames(J):
    """rigid frame per bone from joints (N,3): origin = parent joint, x = bone direction, y/z from a stable reference."""
    Rs, ts, lens = [], [], []
    ref = torch.tensor([0.0, -1.0, 0.0], device=J.device)
    for a, b in BONES:
        d = J[b] - J[a]; L = d.norm() + 1e-6; x = d / L
        y = ref - (ref @ x) * x
        if y.norm() < 1e-3: y = torch.tensor([1.0, 0, 0], device=J.device) - x[0] * x
        y = y / y.norm(); z = torch.linalg.cross(x, y)
        Rs.append(torch.stack([x, y, z], 1)); ts.append(J[a]); lens.append(L)
    return torch.stack(Rs), torch.stack(ts), torch.stack(lens)


def skin_weights(P, J, k=3, sigma=0.12):
    """soft weights of points P (n,3) to bones by distance to the segment."""
    d = []
    for a, b in BONES:
        A, B = J[a], J[b]; ab = B - A; t = ((P - A) @ ab / (ab @ ab + 1e-6)).clamp(0, 1); d.append((P - (A + t[:, None] * ab)).norm(dim=1))
    d = torch.stack(d, 1)  # n, nb
    w = torch.exp(-(d / sigma) ** 2); top = w.topk(k, dim=1); mask = torch.zeros_like(w).scatter_(1, top.indices, 1.0); w = w * mask
    weak = w.sum(1) < 1e-6
    w[weak] = 0; w[weak, d[weak].argmin(1)] = 1.0   # never let a row vanish (it would collapse the point onto the world origin)
    return w / (w.sum(1, keepdim=True) + 1e-9)


def bone_distance(P, J):
    d = []
    for a, b in BONES:
        A, B = J[a], J[b]; ab = B - A; t = ((P - A) @ ab / (ab @ ab + 1e-6)).clamp(0, 1); d.append((P - (A + t[:, None] * ab)).norm(dim=1))
    return torch.stack(d, 1).min(1).values


def pose_gaussians(can, W, Rc, tc, Rf, tf):
    """canonical means/quats -> frame via LBS. Rc,tc: canonical bone frames; Rf,tf: frame bone frames."""
    # per-bone transform: x_f = Rf Rc^T (x - tc) + tf
    M_R = torch.einsum("bij,bkj->bik", Rf, Rc)                      # (nb,3,3) = Rf @ Rc^T
    M_t = tf - torch.einsum("bij,bj->bi", M_R, tc)                    # (nb,3)
    Rw = torch.einsum("nb,bij->nij", W, M_R); tw = W @ M_t            # blended (not exactly rigid, fine for splats)
    means = torch.einsum("nij,nj->ni", Rw, can["means"]) + tw
    # rotate quats by the dominant bone's rotation
    dom = W.argmax(1); Rd = M_R[dom]
    q = F.normalize(can["quats"], dim=-1); w, x, y, z = q.unbind(-1)
    Rq = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
    R = Rd @ Rq
    # matrix -> quat (w,x,y,z), batched, numerically simple version
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]; qw = torch.sqrt((1 + tr).clamp(min=1e-6)) / 2
    qx = (R[:, 2, 1] - R[:, 1, 2]) / (4 * qw); qy = (R[:, 0, 2] - R[:, 2, 0]) / (4 * qw); qz = (R[:, 1, 0] - R[:, 0, 1]) / (4 * qw)
    quats = torch.stack([qw, qx, qy, qz], -1)
    return dict(means=means, quats=quats, scales=can["scales"], opacities=can["opacities"], sh0=can["sh0"], shN=can["shN"])


def rast(sp, v, sh, bgc):
    out, alpha, info = rasterization(sp["means"], F.normalize(sp["quats"], dim=-1), torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]), torch.cat([sp["sh0"], sp["shN"]], 1),
                                     v["w2c"][None], v["K"][None], v["w"], v["h"], sh_degree=sh, packed=False, backgrounds=bgc[None], render_mode="RGB")
    return out[0], alpha[0, ..., 0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", nargs="+", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--bg", required=True)
    ap.add_argument("--init_points", required=True); ap.add_argument("--out", required=True); ap.add_argument("--iters", type=int, default=8000); ap.add_argument("--holdout", default=""); ap.add_argument("--n_gauss", type=int, default=60000); ap.add_argument("--repeat", type=int, default=2); ap.add_argument("--orbit_deg", type=float, default=120.0); ap.add_argument("--extra_joints", default="")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); hold = set(a.holdout.split(",")) - {""}; dev = "cuda"
    frames = [Path(f) for f in a.frames]; ref_i = [i for i, f in enumerate(frames) if f.resolve() == Path(a.ref).resolve()][0]
    import cv2
    def enc(u8): return cv2.imencode(".jpg", cv2.cvtColor(u8.numpy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    def dec(buf): return torch.from_numpy(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    views = []
    for f in frames:   # load ONE frame at a time and compress immediately: 2400 float views would be ~26 GB
        vs = load_masks(f, load_views(f, hold, "cpu")); views.append(vs)
        for v in vs:
            v["jpg"] = enc((v["img"] * 255).round().to(torch.uint8)); del v["img"]
            v["mask_np"] = np.packbits(v["mask"].numpy()); v["mask_shape"] = tuple(v["mask"].shape); del v["mask"]
            v["K"] = v["K"].to(dev); v["w2c"] = v["w2c"].to(dev)
    def get_img(v): return dec(v["jpg"]).to(dev).float() / 255
    def get_mask(v): return torch.from_numpy(np.unpackbits(v["mask_np"])[: v["mask_shape"][0] * v["mask_shape"][1]].reshape(v["mask_shape"]).astype(bool)).to(dev)
    cam_names = sorted({Path(v["name"]).stem for vs in views for v in vs}); cam_idx = {c: i for i, c in enumerate(cam_names)}
    color = torch.nn.Parameter(torch.zeros(len(cam_names), 2, 3, device=dev))   # per-camera log-gain and offset (exposure / white balance)
    def calib(img, v): c = color[cam_idx[Path(v["name"]).stem]]; return img * torch.exp(c[0]) + c[1]
    global BONES
    Jraw = fill_missing(torch.stack([load_skel(f) for f in frames])).to(dev)
    J_all = with_mids(Jraw)                                  # 17 COCO + 8 mids = 25
    if a.extra_joints:                                       # Theia head/toes/pelvis/torso appended at 25..29
        z = np.load(a.extra_joints, allow_pickle=True)
        row = {int(fv): i for i, fv in enumerate(z["frame_video"])}                      # align by VIDEO frame, not position
        sel = [row[int(json.load(open(f / "meta.json"))["frame"])] for f in frames]
        Je = torch.from_numpy(z["joints"][sel]).to(dev)[:, 17:]
        ok = torch.isfinite(Je[..., 0]).all(0); Je = torch.nan_to_num(Je)
        base = J_all.shape[1]; J_all = torch.cat([J_all, Je], 1)
        head, ltoe, rtoe, pelvis, torso = (base + i for i in range(5))
        add = [(15, ltoe), (16, rtoe), (0, head), (pelvis, torso), (torso, 0)]
        good = lambda i: i < base or bool(ok[i - base])
        BONES = BONES + [(x, y) for x, y in add if good(x) and good(y)]
        print(f"extra joints {int(ok.sum())}/5, bones now {len(BONES)}", flush=True)
    bones = [bone_frames(J_all[t]) for t in range(len(frames))]; Rc, tc, Lc = bones[ref_i]
    bg = torch.load(a.bg, map_location=dev)["splats"]; bgc = torch.zeros(3, device=dev)
    with torch.no_grad():
        for vs in views:
            for v in vs:
                if v["train"]: v["bg_jpg"] = enc((render(bg, v, 3, bgc)[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu())
    # canonical Gaussians: DUSt3R athlete points of the reference frame, plus samples along the bones
    ref_views = [dict(v, mask=get_mask(v)) for v in views[ref_i] if v["train"]]
    xyz, rgb = points_split(a.init_points, ref_views, inside=True)
    if len(xyz) > a.n_gauss: sel = torch.randperm(len(xyz), device=dev)[:a.n_gauss]; xyz, rgb = xyz[sel], rgb[sel]
    extra = []
    for (p, q), L in zip(BONES, Lc):
        t = torch.rand(1500, 1, device=dev); r = (torch.randn(1500, 3, device=dev) * 0.05); extra.append(J_all[ref_i][p] + t * (J_all[ref_i][q] - J_all[ref_i][p]) + r)
    xyz = torch.cat([xyz, torch.cat(extra)]); rgb = torch.cat([rgb, torch.full((len(torch.cat(extra)), 3), 0.5, device=dev)])
    keep = bone_distance(xyz, J_all[ref_i]) < 0.35; print("dropping off-body init points:", int((~keep).sum()), flush=True); xyz, rgb = xyz[keep], rgb[keep]
    # silhouette carve: keep canonical points inside the athlete mask in every training view of the reference frame
    inside = torch.ones(len(xyz), dtype=torch.bool, device=dev)
    for v in views[ref_i]:
        if not v["train"]: continue
        R, tt = v["w2c"][:3, :3], v["w2c"][:3, 3]; p = xyz @ R.T + tt; zc = p[:, 2].clamp(min=1e-6)
        u = (v["K"][0, 0] * p[:, 0] / zc + v["K"][0, 2]).long().clamp(0, v["w"] - 1); w_ = (v["K"][1, 1] * p[:, 1] / zc + v["K"][1, 2]).long().clamp(0, v["h"] - 1)
        inside &= get_mask(v)[w_, u] | (bone_distance(xyz, J_all[ref_i]) < 0.06)
    print("silhouette carve keeps", int(inside.sum()), "of", len(xyz), flush=True); xyz, rgb = xyz[inside], rgb[inside]
    W0 = skin_weights(xyz, J_all[ref_i]); W_logit = torch.nn.Parameter(torch.log(W0 + 1e-4))
    n = len(xyz); print("canonical gaussians", n, flush=True)
    can = torch.nn.ParameterDict(dict(means=torch.nn.Parameter(xyz.clone()), scales=torch.nn.Parameter(torch.log(torch.full((n, 3), 0.02, device=dev))), quats=torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1)),
                                      opacities=torch.nn.Parameter(torch.logit(torch.full((n,), 0.2, device=dev))), sh0=torch.nn.Parameter(((rgb - 0.5) / 0.28209479177387814)[:, None, :]), shN=torch.nn.Parameter(torch.zeros(n, 15, 3, device=dev))))
    # per-frame skeleton correction (small joint offsets) so triangulation noise can be absorbed
    dJ = torch.nn.Parameter(torch.zeros(len(frames), J_all.shape[1], 3, device=dev))
    lr = dict(means=2e-4, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=1.25e-4)
    opt = torch.optim.Adam([{"params": can[k], "lr": lr[k]} for k in can] + [{"params": dJ, "lr": 1e-4}, {"params": W_logit, "lr": 5e-3}, {"params": color, "lr": 1e-3}], eps=1e-15)
    from pytorch_msssim import ssim as ssim_fn
    t0 = time.time(); train_pairs = [(t, v) for t, vs in enumerate(views) for v in vs if v["train"]]
    for it in range(a.iters):
        t, v = train_pairs[np.random.randint(len(train_pairs))]; sh = min(it // 1000, 3)
        Rf, tf, _ = bone_frames(J_all[t] + dJ[t])
        W = torch.softmax(W_logit, dim=1)
        sp = pose_gaussians(can, W, Rc, tc, Rf, tf)
        rgb_r, alpha = rast(sp, v, sh, bgc)
        gt = get_img(v); mk = get_mask(v)
        img = calib(rgb_r + (1 - alpha)[..., None] * (dec(v["bg_jpg"]).to(dev).float() / 255), v)
        l1 = (img - gt).abs().mean(); ss = 1 - ssim_fn(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0)
        loss = 0.8 * l1 + 0.2 * ss + 10.0 * (alpha * (~mk).float()).mean() + 0.5 * ((1 - alpha) * mk.float()).mean() + 1e-3 * dJ[t].abs().mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            can["scales"].clamp_(max=math.log(0.08))
            if it % 500 == 0 and it > 0:  # kill canonical Gaussians that wandered off the body or sit on a camera
                dbone = torch.stack([((can["means"] - J_all[ref_i][a]) - (((can["means"] - J_all[ref_i][a]) @ (J_all[ref_i][b] - J_all[ref_i][a])) / ((J_all[ref_i][b] - J_all[ref_i][a]) @ (J_all[ref_i][b] - J_all[ref_i][a]) + 1e-6)).clamp(0, 1)[:, None] * (J_all[ref_i][b] - J_all[ref_i][a])).norm(dim=1) for a, b in BONES], 1).min(1).values
                bad = dbone > 0.35
                for vs in views:
                    for v in vs:
                        C = -v["w2c"][:3, :3].T @ v["w2c"][:3, 3]; bad |= (can["means"] - C).norm(dim=1) < 1.5
                if bad.any():
                    can["opacities"].data[bad] = -8.0  # effectively transparent; keeps tensor shapes and optimizer state
                    print(f"  silenced {int(bad.sum())} off-body gaussians", flush=True)
        if it % 500 == 0 or it == a.iters - 1:
            print(f"it {it:5d} loss {loss.item():.4f} |dJ| {dJ.abs().mean().item()*100:.2f}cm gain-spread {torch.exp(color[:, 0]).std().item():.3f} {time.time()-t0:.0f}s", flush=True)
    # evaluation on every frame's held-out camera + rendering of a novel-view playback
    res = {}
    with torch.no_grad():
        W = torch.softmax(W_logit, dim=1)
        target, up, radius, height = rig_geometry(views[0]); e1 = F.normalize(torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=dev)), dim=0); e2 = torch.linalg.cross(up, e1)
        frames_out = []
        for t, vs in enumerate(views):
            Rf, tf, _ = bone_frames(J_all[t] + dJ[t]); sp = pose_gaussians(can, W, Rc, tc, Rf, tf); full = {k: torch.cat([bg[k], sp[k].detach()], 0) for k in sp}
            for v in vs:
                if v["train"]: continue
                img, _ = render(full, v, 3, bgc); img = calib(img, v).clamp(0, 1); m = get_mask(v); gt = get_img(v)
                res[frames[t].name] = dict(psnr=psnr(img, gt).item(), athlete_psnr=psnr(img[m], gt[m]).item() if m.any() else float("nan"))
                if t in (0, len(views) // 2, len(views) - 1):
                    imageio.imwrite(out / f"v4_test_{frames[t].name}_{Path(v['name']).stem}.jpg", (torch.cat([gt, img], 1).cpu().numpy() * 255).astype(np.uint8), quality=90)
            ang = math.radians(-a.orbit_deg / 2 + a.orbit_deg * t / max(len(views) - 1, 1)); ref = vs[0]
            eye = target + 0.9 * radius * (math.cos(ang) * e1 + math.sin(ang) * e2) + 0.8 * height * up; v1 = dict(ref); v1["w2c"] = look_at(eye, target, up)
            eye2 = target + 0.9 * radius * (math.cos(1.2) * e1 + math.sin(1.2) * e2) + 0.5 * height * up; v2 = dict(ref); v2["w2c"] = look_at(eye2, target, up)
            im1 = render(full, v1, 3, bgc)[0].clamp(0, 1); im2 = render(full, v2, 3, bgc)[0].clamp(0, 1)
            frames_out.extend([(torch.cat([im1, im2], 1).cpu().numpy() * 255).astype(np.uint8)] * a.repeat)
        imageio.mimwrite(out / "v4_4d.mp4", frames_out, fps=30, quality=8)
    torch.save(dict(can={k: v.detach() for k, v in can.items()}, W=W.detach(), Rc=Rc, tc=tc, dJ=dJ.detach(), J=J_all, bg=a.bg, color={c: color[i].detach().cpu().tolist() for c, i in cam_idx.items()}, bones=BONES), out / "ckpt.pt")
    json.dump(res, open(out / "result.json", "w"), indent=1)
    vals = [r["psnr"] for r in res.values()]; ath = [r["athlete_psnr"] for r in res.values()]
    print(json.dumps(dict(holdout_psnr_mean=float(np.mean(vals)), athlete_psnr_mean=float(np.nanmean(ath)), n_frames=len(vals))))


if __name__ == "__main__":
    main()
