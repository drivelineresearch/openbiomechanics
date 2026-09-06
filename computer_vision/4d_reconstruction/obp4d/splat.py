"""Rasterization, the photometric loss and the background checkpoint, shared by the two trainers and the renderer."""

import gsplat
import torch
from torchmetrics.functional import structural_similarity_index_measure

from .rig import H, W


def rasterize(means, quats, scales, opacities, colors, viewmat, K, absgrad=False):
    """color (1, H, W, 3), alpha (1, H, W, 1) and gsplat's info; colors are view-independent (degree 0)"""
    return gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat,
        K,
        W,
        H,
        sh_degree=0,
        packed=False,
        absgrad=absgrad,
        backgrounds=torch.zeros(1, 3, device=means.device),
    )


def photometric(pred, target):
    """0.8 L1 + 0.2 (1 - SSIM) between (1, h, w, 3) images in [0, 1]; also returns the SSIM"""
    s = structural_similarity_index_measure(
        pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), data_range=1.0
    )
    return 0.8 * (pred - target).abs().mean() + 0.2 * (1 - s), s


def load_background(path, dev="cuda"):
    """means, quats, scales, opacities, colors, ready for rasterize()"""
    d = torch.load(path, weights_only=True)
    return [d[k].to(dev) for k in ("means", "quats", "scales", "opacities", "colors")]
