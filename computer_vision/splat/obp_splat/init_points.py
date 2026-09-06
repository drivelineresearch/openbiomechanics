"""Multi-view-consistent initialization for sparse-view splatting: run DUSt3R (mini-dust3r) on the training
images, then align its self-consistent reconstruction to the OpenBiomechanics calibration with a similarity
transform fitted on the camera centres (Umeyama), and write points.npz for train_gs.py --init_points.

  python dust3r_init.py --data data/throw_f0700 --out data/throw_f0700/points_dust3r.npz --holdout cam19
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .core import read_colmap_txt


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ d @ Vt
    s = np.trace(np.diag(sig) @ d) / (S**2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--max_points", type=int, default=400000)
    ap.add_argument("--conf", type=int, default=3)
    ap.add_argument("--niter", type=int, default=300)
    a = ap.parse_args()
    d = Path(a.data)
    hold = set(a.holdout.split(",")) - {""}
    from mini_dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from mini_dust3r.image_pairs import make_pairs
    from mini_dust3r.inference import inference
    from mini_dust3r.model import AsymmetricCroCo3DStereo
    from mini_dust3r.utils.image import load_images

    ims = [im for im in read_colmap_txt(d) if Path(im["name"]).stem not in hold]
    paths = [str(d / "images" / im["name"]) for im in ims]
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to("cuda")
    imgs = load_images(folder_or_list=paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    out = inference(pairs, model, "cuda", batch_size=1)
    scene = global_aligner(
        dust3r_output=out, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer
    )
    loss = scene.compute_global_alignment(
        init="mst", niter=a.niter, schedule="linear", lr=0.01
    )
    print("global alignment loss", float(loss), flush=True)
    cam2world = scene.get_im_poses().detach().cpu().numpy()
    C_d = cam2world[:, :3, 3]
    C_c = np.array([(-im["w2c"][:3, :3].T @ im["w2c"][:3, 3]) for im in ims])
    s, R, t = umeyama(C_d, C_c)
    resid = np.linalg.norm((s * (R @ C_d.T).T + t) - C_c, axis=1)
    print(
        "similarity scale",
        round(float(s), 4),
        "camera-centre residuals (m):",
        np.round(resid, 3),
        flush=True,
    )
    xyz, rgb = [], []
    for pts, conf, img in zip(scene.get_pts3d(), scene.im_conf, scene.imgs):
        pts = pts.detach().cpu().numpy().reshape(-1, 3)
        conf = conf.detach().cpu().numpy().reshape(-1)
        img = np.asarray(img).reshape(-1, 3)
        m = conf > a.conf
        xyz.append(pts[m])
        rgb.append(img[m])
    xyz = np.concatenate(xyz)
    rgb = np.concatenate(rgb)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    xyz = s * (R @ xyz.T).T + t
    if len(xyz) > a.max_points:
        sel = np.random.default_rng(0).choice(len(xyz), a.max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    extra = {}
    for im, dep, conf in zip(ims, scene.get_depthmaps(), scene.im_conf):
        stem = Path(im["name"]).stem
        extra[f"depth_{stem}"] = (dep.detach().cpu().numpy() * s).astype(np.float32)
        extra[f"conf_{stem}"] = conf.detach().cpu().numpy().astype(np.float32)
    np.savez(
        a.out,
        xyz=xyz.astype(np.float32),
        rgb=rgb.astype(np.float32),
        align=json.dumps({"scale": float(s), "resid_m": resid.tolist()}),
        **extra,
    )
    print(
        "wrote",
        a.out,
        len(xyz),
        "points; extent",
        np.round(xyz.min(0), 1),
        np.round(xyz.max(0), 1),
    )


if __name__ == "__main__":
    main()
