"""V7: SMPL-X body model with per-vertex Gaussians (hands, feet and head come from the model, not from bones).

Stage A: fit SMPL-X (shared betas + scale, per-frame global orientation, body pose and translation) to the 3D
joints, using Driveline's Theia3D joint centres when available (theia_joints.npz) and our triangulated COCO
joints otherwise. Temporal smoothness and a pose prior keep it stable.
Stage B: one Gaussian per SMPL-X vertex, carried by the mesh. Each frame's vertices come from the fitted model;
per-vertex rotation comes from the change of the local mesh frame (normal + tangent), so Gaussians follow the
surface. Appearance (offset, scale, rotation, opacity, colour) is trained on the images over the frozen
background, exactly like V6, with the held-out camera never used.

  python train_smplx.py --frames data/throw_full/t0* --bg out/v6_bg/ckpt.pt --model data/smplx/models/smplx/SMPLX_NEUTRAL.npz \
      --theia data/throw_full/theia_joints.npz --out out/v7 --fit_iters 1500 --iters 12000 --holdout cam19
"""

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
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
from obp_splat.train_background import load_masks
from obp_splat.train_body import fill_missing, load_skel

# COCO index -> SMPL-X body joint index (SMPL-X: 0 pelvis, 1/2 hips, 4/5 knees, 7/8 ankles, 16/17 shoulders, 18/19 elbows, 20/21 wrists)
COCO2X = {
    11: 1,
    12: 2,
    13: 4,
    14: 5,
    15: 7,
    16: 8,
    5: 16,
    6: 17,
    7: 18,
    8: 19,
    9: 20,
    10: 21,
}


def vertex_frames(V, nb0, nb1):
    """orthonormal frame per vertex from the mesh itself: n = normalised (e0 x e1), t = normalised e0."""
    e0 = V[:, nb0] - V
    e1 = V[:, nb1] - V
    n = F.normalize(torch.linalg.cross(e0, e1, dim=-1), dim=-1)
    t = F.normalize(e0 - (e0 * n).sum(-1, keepdim=True) * n, dim=-1)
    b = torch.linalg.cross(n, t, dim=-1)
    return torch.stack([t, b, n], -1)  # (..., V, 3, 3)


def mat_to_quat(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = torch.sqrt((1 + tr).clamp(min=1e-6)) / 2
    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * qw)
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * qw)
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * qw)
    return torch.stack([qw, qx, qy, qz], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--theia", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument(
        "--sil_iters",
        type=int,
        default=0,
        help="silhouette refinement steps after the joint fit",
    )
    ap.add_argument("--fit_iters", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--orbit_deg", type=float, default=160.0)
    a = ap.parse_args()
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hold = set(a.holdout.split(",")) - {""}
    frames = [Path(f) for f in a.frames]
    T = len(frames)

    # ---------------- Stage A: fit SMPL-X to 3D joints ----------------
    import smplx

    model = smplx.create(
        str(Path(a.model).parents[1]),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=T,
    ).to(dev)
    Jours = fill_missing(torch.stack([load_skel(f) for f in frames])).to(
        dev
    )  # (T,17,3) ours
    target = torch.full(
        (T, 22, 3), float("nan"), device=dev
    )  # SMPL-X body joints 0..21
    for c, x in COCO2X.items():
        target[:, x] = Jours[:, c]
    if a.theia:
        z = np.load(a.theia, allow_pickle=True)
        row = {
            int(fv): i for i, fv in enumerate(z["frame_video"])
        }  # align by VIDEO frame, not position
        sel = [
            row[int(json.loads(Path(f / "meta.json").read_text())["frame"])]
            for f in frames
        ]
        Jt = torch.from_numpy(z["joints"][sel][:, :17]).to(
            dev
        )  # (T,17,3), NaN where absent
        for c, x in COCO2X.items():
            m = torch.isfinite(Jt[:, c, 0])
            target[m, x] = Jt[m, c]  # prefer Theia joint centres
        print(
            "using Theia joints for",
            int(torch.isfinite(Jt[:, 11, 0]).sum()),
            "frames",
            flush=True,
        )
    valid = torch.isfinite(target[..., 0])
    betas = torch.nn.Parameter(torch.zeros(1, 10, device=dev))
    scale = torch.nn.Parameter(torch.zeros(1, device=dev))
    orient = torch.nn.Parameter(torch.zeros(T, 3, device=dev))
    pose = torch.nn.Parameter(torch.zeros(T, 63, device=dev))
    transl = torch.nn.Parameter(target[:, 0].nan_to_num().clone())
    opt = torch.optim.Adam(
        [
            {"params": [orient, transl], "lr": 3e-2},
            {"params": [pose], "lr": 3e-2},
            {"params": [betas, scale], "lr": 1e-2},
        ]
    )
    t0 = time.time()
    for it in range(a.fit_iters):
        o = model(
            betas=betas.expand(T, -1),
            global_orient=orient,
            body_pose=pose,
            transl=torch.zeros_like(transl),
            return_verts=False,
        )
        J = o.joints[:, :22] * torch.exp(scale) + transl[:, None]
        jl = ((J - target.nan_to_num()) ** 2).sum(-1)[valid].mean()
        smooth = (
            ((pose[1:] - pose[:-1]) ** 2).mean()
            + ((orient[1:] - orient[:-1]) ** 2).mean()
            + ((transl[1:] - transl[:-1]) ** 2).mean()
        )
        loss = jl + 0.5 * smooth + 1e-3 * (pose**2).mean() + 1e-2 * (betas**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == a.fit_iters - 1:
            print(
                f"  fit {it:5d} joint rmse {jl.sqrt().item() * 100:5.2f} cm  scale {torch.exp(scale).item():.3f}  {time.time() - t0:.0f}s",
                flush=True,
            )
    # ---- optional silhouette refinement: the joints fix the skeleton, the masks fix the SHAPE ----
    if a.sil_iters:
        views_s = []
        for f in frames:
            vs = load_masks(f, load_views(f, hold, "cpu"))
            views_s.append(vs)
            for v in vs:
                v["mask_np"] = np.packbits(v["mask"].numpy())
                v["mask_shape"] = tuple(v["mask"].shape)
                del v["mask"]
                del v["img"]
                v["K"] = v["K"].to(dev)
                v["w2c"] = v["w2c"].to(dev)

        def gm(v):
            return torch.from_numpy(
                np.unpackbits(v["mask_np"])[: v["mask_shape"][0] * v["mask_shape"][1]]
                .reshape(v["mask_shape"])
                .astype(bool)
            ).to(dev)

        dV = torch.nn.Parameter(
            torch.zeros(1, 10475, 3, device=dev)
        )  # shared per-vertex shape correction
        opt2 = torch.optim.Adam(
            [
                {"params": [betas, scale], "lr": 3e-3},
                {"params": [pose, orient, transl], "lr": 3e-3},
                {"params": [dV], "lr": 1e-3},
            ]
        )
        pairs = [(t, v) for t, vs in enumerate(views_s) for v in vs if v["train"]]
        ones = None
        t1 = time.time()
        for it in range(a.sil_iters):
            t, v = pairs[np.random.randint(len(pairs))]
            o = model(
                betas=betas.expand(T, -1),
                global_orient=orient,
                body_pose=pose,
                transl=torch.zeros_like(transl),
                return_verts=True,
            )
            Vt = (o.vertices[t] + dV[0]) * torch.exp(scale) + transl[t]
            if ones is None or ones.shape[0] != Vt.shape[0]:
                ones = torch.ones(Vt.shape[0], device=dev)
            _, alpha, _ = rasterization(
                Vt,
                torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(Vt), 1),
                torch.full((len(Vt), 3), 0.012, device=dev),
                ones * 0.99,
                torch.zeros(len(Vt), 1, 3, device=dev),
                v["w2c"][None],
                v["K"][None],
                v["w"],
                v["h"],
                sh_degree=0,
                packed=False,
                backgrounds=torch.zeros(3, device=dev)[None],
                render_mode="RGB",
            )
            m = gm(v).float()
            sil = (alpha[0, ..., 0] - m).abs().mean()
            J = o.joints[:, :22] * torch.exp(scale) + transl[:, None]
            jl = ((J - target.nan_to_num()) ** 2).sum(-1)[valid].mean()
            loss = sil + 2.0 * jl + 1e-2 * (dV**2).mean() + 1e-3 * (pose**2).mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step()
            if it % 200 == 0 or it == a.sil_iters - 1:
                print(
                    f"  sil {it:5d} silhouette {sil.item():.4f} joint rmse {jl.sqrt().item() * 100:5.2f} cm |dV| {dV.norm(dim=-1).mean().item() * 100:.2f}cm scale {torch.exp(scale).item():.3f} {time.time() - t1:.0f}s",
                    flush=True,
                )
        del views_s
    with torch.no_grad():
        o = model(
            betas=betas.expand(T, -1),
            global_orient=orient,
            body_pose=pose,
            transl=torch.zeros_like(transl),
            return_verts=True,
        )
        verts = o.vertices + (dV[0] if a.sil_iters else 0)
        Vp = (verts * torch.exp(scale) + transl[:, None]).detach()
        J_fit = (o.joints[:, :22] * torch.exp(scale) + transl[:, None]).detach()
        rmse = ((J_fit - target.nan_to_num()) ** 2).sum(-1)[valid].mean().sqrt().item()
    print(
        f"SMPL-X fit: {Vp.shape[1]} vertices, joint rmse {rmse * 100:.2f} cm",
        flush=True,
    )

    # mesh neighbours for the per-vertex frame
    faces = torch.from_numpy(model.faces.astype(np.int64)).to(dev)
    nb0 = torch.zeros(Vp.shape[1], dtype=torch.long, device=dev)
    nb1 = torch.zeros_like(nb0)
    nb0[faces[:, 0]] = faces[:, 1]
    nb1[faces[:, 0]] = faces[:, 2]
    nb0[faces[:, 1]] = faces[:, 2]
    nb1[faces[:, 1]] = faces[:, 0]
    nb0[faces[:, 2]] = faces[:, 0]
    nb1[faces[:, 2]] = faces[:, 1]
    Fr = vertex_frames(Vp, nb0, nb1)  # (T,V,3,3)

    # ---------------- Stage B: per-vertex Gaussians ----------------
    views = []
    for f in frames:
        vs = load_masks(f, load_views(f, hold, "cpu"))
        views.append(vs)
        for v in vs:
            v["jpg"] = cv2.imencode(
                ".jpg",
                cv2.cvtColor(
                    (v["img"] * 255).round().to(torch.uint8).numpy(), cv2.COLOR_RGB2BGR
                ),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )[1]
            del v["img"]
            v["mask_np"] = np.packbits(v["mask"].numpy())
            v["mask_shape"] = tuple(v["mask"].shape)
            del v["mask"]
            v["K"] = v["K"].to(dev)
            v["w2c"] = v["w2c"].to(dev)

    def dec(b):
        return torch.from_numpy(
            cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        )

    def get_img(v):
        return dec(v["jpg"]).to(dev).float() / 255

    def get_mask(v):
        return torch.from_numpy(
            np.unpackbits(v["mask_np"])[: v["mask_shape"][0] * v["mask_shape"][1]]
            .reshape(v["mask_shape"])
            .astype(bool)
        ).to(dev)

    bg = torch.load(a.bg, map_location=dev)["splats"]
    bgc = torch.zeros(3, device=dev)
    with torch.no_grad():
        for vs in views:
            for v in vs:
                if v["train"]:
                    v["bg_jpg"] = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(
                            (render(bg, v, 3, bgc)[0].clamp(0, 1) * 255)
                            .round()
                            .to(torch.uint8)
                            .cpu()
                            .numpy(),
                            cv2.COLOR_RGB2BGR,
                        ),
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )[1]
    cams = sorted({Path(v["name"]).stem for vs in views for v in vs})
    cidx = {c: i for i, c in enumerate(cams)}
    color = torch.nn.Parameter(torch.zeros(len(cams), 2, 3, device=dev))

    def calib(img, v):
        return (
            img * torch.exp(color[cidx[Path(v["name"]).stem]][0])
            + color[cidx[Path(v["name"]).stem]][1]
        )

    V = Vp.shape[1]
    g = torch.nn.ParameterDict(
        {
            "offset": torch.nn.Parameter(
                torch.zeros(V, 3, device=dev)
            ),  # local-frame offset from the vertex
            "scales": torch.nn.Parameter(
                torch.log(torch.full((V, 3), 0.012, device=dev))
            ),
            "quats": torch.nn.Parameter(
                torch.tensor([1.0, 0, 0, 0], device=dev).repeat(V, 1)
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((V,), 0.5, device=dev))
            ),
            "sh0": torch.nn.Parameter(torch.zeros(V, 1, 3, device=dev)),
            "shN": torch.nn.Parameter(torch.zeros(V, 15, 3, device=dev)),
        }
    )
    lr = {
        "offset": 5e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }
    opt = torch.optim.Adam(
        [{"params": g[k], "lr": lr[k]} for k in g] + [{"params": color, "lr": 1e-3}],
        eps=1e-15,
    )
    from pytorch_msssim import ssim as ssim_fn

    train_pairs = [(t, v) for t, vs in enumerate(views) for v in vs if v["train"]]
    qb = mat_to_quat(Fr)  # per-frame per-vertex frame as quaternion
    t0 = time.time()

    def posed(t):
        R = Fr[t]
        means = Vp[t] + torch.einsum("vij,vj->vi", R, g["offset"])
        q = F.normalize(g["quats"], dim=-1)
        qf = qb[t]
        w1, x1, y1, z1 = qf.unbind(-1)
        w2, x2, y2, z2 = q.unbind(-1)
        quats = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            -1,
        )
        return means, quats

    for it in range(a.iters):
        t, v = train_pairs[np.random.randint(len(train_pairs))]
        sh = min(it // 1500, 3)
        means, quats = posed(t)
        rgb, alpha, _ = rasterization(
            means,
            F.normalize(quats, dim=-1),
            torch.exp(g["scales"]),
            torch.sigmoid(g["opacities"]),
            torch.cat([g["sh0"], g["shN"]], 1),
            v["w2c"][None],
            v["K"][None],
            v["w"],
            v["h"],
            sh_degree=sh,
            packed=False,
            backgrounds=bgc[None],
            render_mode="RGB",
        )
        rgb, alpha = rgb[0], alpha[0, ..., 0]
        gt = get_img(v)
        mk = get_mask(v)
        img = calib(
            rgb + (1 - alpha)[..., None] * (dec(v["bg_jpg"]).to(dev).float() / 255), v
        )
        l1 = (img - gt).abs().mean()
        ss = 1 - ssim_fn(
            img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0
        )
        loss = (
            0.8 * l1
            + 0.2 * ss
            + 10.0 * (alpha * (~mk).float()).mean()
            + 0.5 * ((1 - alpha) * mk.float()).mean()
            + 1e-2 * (g["offset"] ** 2).mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            g["scales"].clamp_(max=math.log(0.05))
        if it % 1000 == 0 or it == a.iters - 1:
            print(
                f"it {it:5d} loss {loss.item():.4f} |offset| {g['offset'].norm(dim=1).mean().item() * 100:.2f}cm {time.time() - t0:.0f}s",
                flush=True,
            )

    # ---------------- evaluation + videos ----------------
    res = {}
    hold_frames = []
    stage_frames = []
    with torch.no_grad():
        target_pt, up, radius, height = rig_geometry(
            [dict(v, img=torch.zeros(1)) for v in views[0]]
        )
        e1 = F.normalize(
            torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device=dev)), dim=0
        )
        e2 = torch.linalg.cross(up, e1)
        for t, vs in enumerate(views):
            means, quats = posed(t)
            sp = {
                "means": means,
                "quats": quats,
                "scales": g["scales"],
                "opacities": g["opacities"],
                "sh0": g["sh0"],
                "shN": g["shN"],
            }
            full = {k: torch.cat([bg[k], sp[k]], 0) for k in sp}
            for v in vs:
                if v["train"]:
                    continue
                img = calib(render(full, v, 3, bgc)[0], v).clamp(0, 1)
                gt = get_img(v)
                m = get_mask(v)
                res[frames[t].name] = {
                    "psnr": psnr(img, gt).item(),
                    "athlete_psnr": psnr(img[m], gt[m]).item()
                    if m.any()
                    else float("nan"),
                }
                fr = np.ascontiguousarray(
                    (torch.cat([gt, img], 1).cpu().numpy() * 255).astype(np.uint8)
                )
                cv2.putText(
                    fr,
                    "real cam19  (held out)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    fr,
                    f"SMPL-X gaussians from the other 7   {res[frames[t].name]['psnr']:.1f} dB",
                    (1300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                hold_frames.append(fr)
            ang = math.radians(-a.orbit_deg / 2 + a.orbit_deg * t / max(T - 1, 1))
            ref = vs[0]
            eye = (
                target_pt
                + 0.85 * radius * (math.cos(ang) * e1 + math.sin(ang) * e2)
                + 0.85 * height * up
            )
            v1 = dict(ref)
            v1["w2c"] = look_at(eye, target_pt, up)
            eye2 = (
                target_pt
                + 0.8 * radius * (math.cos(1.4) * e1 + math.sin(1.4) * e2)
                + 0.75 * height * up
            )
            v2 = dict(ref)
            v2["w2c"] = look_at(eye2, target_pt, up)
            keep = (torch.sigmoid(bg["opacities"]) >= 0.05) & (
                (bg["means"] - target_pt).norm(dim=1) <= 3.5
            )
            bgk = {k: v[keep] for k, v in bg.items()}
            fullk = {k: torch.cat([bgk[k], sp[k]], 0) for k in sp}
            im1 = render(fullk, v1, 3, torch.full((3,), 0.10, device=dev))[0].clamp(
                0, 1
            )
            im2 = render(fullk, v2, 3, torch.full((3,), 0.10, device=dev))[0].clamp(
                0, 1
            )
            stage_frames.append(
                (torch.cat([im1, im2], 1).cpu().numpy() * 255).astype(np.uint8)
            )
    imageio.mimwrite(out / "holdout_cam19.mp4", hold_frames, fps=30, quality=8)
    imageio.mimwrite(out / "stage.mp4", stage_frames, fps=30, quality=8)
    torch.save(
        {
            "gauss": {k: v.detach() for k, v in g.items()},
            "Vp": Vp.cpu(),
            "betas": betas.detach().cpu(),
            "pose": pose.detach().cpu(),
            "orient": orient.detach().cpu(),
            "transl": transl.detach().cpu(),
            "scale": torch.exp(scale).detach().cpu(),
            "bg": a.bg,
            "color": {c: color[i].detach().cpu().tolist() for c, i in cidx.items()},
        },
        out / "ckpt.pt",
    )
    Path(out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    vals = [r["psnr"] for r in res.values()]
    ath = [r["athlete_psnr"] for r in res.values()]
    print(
        json.dumps(
            {
                "smplx_joint_rmse_cm": rmse * 100,
                "holdout_psnr_mean": float(np.mean(vals)),
                "athlete_psnr_mean": float(np.nanmean(ath)),
                "n_frames": len(vals),
            }
        )
    )


if __name__ == "__main__":
    main()
