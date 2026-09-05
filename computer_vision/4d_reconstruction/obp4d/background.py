"""Static background: DUSt3R depth for the eight median plates with the calibrated poses and intrinsics held
fixed, so the geometry lands where the cameras say, then a short Gaussian refinement on the plates (no
densification, view-independent color). Writes a checkpoint the body trainer and the renderer read.

  python -m obp4d background --work W --out bg.pt [--holdout 19]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from .rig import CAMS, W, load_rig, to_torch
from .splat import photometric, rasterize

CONF = 3.0  # DUSt3R confidence threshold
ITERS = 5000
LRS = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
SH_C0 = 0.28209479177387814


def dust3r_seed(rig, cams):
    """confident DUSt3R points (xyz meters, rgb in [0, 1]) for the plates of the given cameras"""
    from mini_dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from mini_dust3r.image_pairs import make_pairs
    from mini_dust3r.inference import inference
    from mini_dust3r.model import AsymmetricCroCo3DStereo
    from mini_dust3r.utils.image import load_images

    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").cuda()
    imgs = load_images([f"plates/cam{c}.png" for c in cams], size=512, verbose=False)
    s = imgs[0]["img"].shape[-1] / W
    out = inference(make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True), model, "cuda", batch_size=4)
    scene = global_aligner(out, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer, min_conf_thr=CONF, optimize_pp=True)
    c2w = []
    for c in cams:
        R, t, _ = rig[c]
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = t
        c2w.append(torch.tensor(np.linalg.inv(m), dtype=torch.float32))
    scene.preset_pose(c2w)
    scene.preset_focal([rig[c][2][0, 0] * s for c in cams])
    scene.preset_principal_point([torch.tensor(rig[c][2][:2, 2] * s) for c in cams])
    loss = scene.compute_global_alignment(init="known_poses", niter=300, schedule="cosine", lr=0.01)
    pts = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    ok = [m.cpu().numpy() for m in scene.get_masks()]
    xyz = np.concatenate([p[m] for p, m in zip(pts, ok)])
    rgb = np.concatenate([np.asarray(im)[m] for im, m in zip(scene.imgs, ok)])
    if rgb.max() > 1.5:
        rgb = rgb / 255
    print(f"DUSt3R alignment loss {float(loss):.3f}, {len(xyz)} confident points", flush=True)
    return xyz.astype(np.float32), rgb.astype(np.float32)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=0, help="camera left out of training, 0 = none")
    a = ap.parse_args(argv)
    rig = load_rig()
    vm, K = to_torch(rig)
    cams = [c for c in CAMS if c != a.holdout]
    xyz, rgb = dust3r_seed(rig, cams)
    n = len(xyz)
    d = cKDTree(xyz).query(xyz, k=4)[0][:, 1:].mean(1)  # each point's scale from its neighbors
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.tensor(xyz, device="cuda")),
            "scales": torch.nn.Parameter(torch.tensor(np.log(np.clip(d, 0.005, 0.1)), dtype=torch.float32, device="cuda")[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device="cuda").repeat(n, 1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device="cuda"))),
            "colors": torch.nn.Parameter(torch.tensor((rgb - 0.5) / SH_C0, device="cuda")[:, None, :]),
        }
    )
    opts = {k: torch.optim.Adam([params[k]], lr=LRS[k], eps=1e-15) for k in params}
    plates = {c: torch.from_numpy(np.array(Image.open(f"plates/cam{c}.png").convert("RGB"))).cuda().float()[None] / 255 for c in cams}
    rng = np.random.default_rng(0)
    for it in range(ITERS):
        c = cams[rng.integers(len(cams))]
        col, _, _ = rasterize(params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["colors"], vm[c], K[c])
        loss, s = photometric(col, plates[c])
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if it % 500 == 0 or it == ITERS - 1:
            print(f"it {it} loss {loss.item():.4f} ssim {s.item():.3f}", flush=True)
    with torch.no_grad():
        keep = torch.sigmoid(params["opacities"]) > 0.05
        torch.save(
            {
                "means": params["means"][keep].cpu(),
                "quats": F.normalize(params["quats"][keep], dim=-1).cpu(),
                "scales": torch.exp(params["scales"][keep]).cpu(),
                "opacities": torch.sigmoid(params["opacities"][keep]).cpu(),
                "colors": params["colors"][keep].cpu(),
            },
            a.out,
        )
    print(f"BACKGROUND_DONE {int(keep.sum())} of {n} Gaussians kept -> {a.out}", flush=True)
