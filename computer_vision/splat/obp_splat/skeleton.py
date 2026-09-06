"""CPU-testable skeleton geometry and checkpoint topology for the splat pipeline."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 17 COCO joints + 8 virtual mid-points (indices 17..24) so long limbs get two bones each: smoother skin at elbows/knees
MID = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
BONES = [
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (0, 5),
    (0, 6),
    (1, 2),
    (3, 4),
    (5, 17),
    (17, 7),
    (7, 18),
    (18, 9),
    (6, 19),
    (19, 8),
    (8, 20),
    (20, 10),
    (11, 21),
    (21, 13),
    (13, 22),
    (22, 15),
    (12, 23),
    (23, 14),
    (14, 24),
    (24, 16),
]


EXTRA_BONES = [
    (15, 18),
    (16, 19),
    (0, 17),
    (9, 9),
    (10, 10),
]  # ankle->toes, nose->head (indices into the extended joint set)


def with_mids(J):
    """append virtual mid-joints (indices grow after any extra joints)"""
    mids = torch.stack([(J[..., a, :] + J[..., b, :]) / 2 for a, b in MID], -2)
    return torch.cat([J, mids], -2)


def load_skel(d):
    s = json.loads(Path(Path(d) / "skeleton.json").read_text())
    J = np.array(
        [j if j is not None else [np.nan] * 3 for j in s["joints"]], dtype=np.float32
    )
    return torch.from_numpy(J)


def fill_missing(J_all):
    """fill missing joints per frame by linear interpolation over time, then hold."""
    T, N, _ = J_all.shape
    for j in range(N):
        ok = ~torch.isnan(J_all[:, j, 0])
        idx = torch.arange(T)
        if ok.sum() == 0:
            J_all[:, j] = 0
        elif ok.sum() < T:
            for c in range(3):
                J_all[:, j, c] = torch.from_numpy(
                    np.interp(idx.numpy(), idx[ok].numpy(), J_all[ok, j, c].numpy())
                )
    return J_all


def bone_frames(J, bones=None):
    """rigid frame per bone from joints (N,3): origin = parent joint, x = bone direction, y/z from a stable reference."""
    bones = BONES if bones is None else bones
    Rs, ts, lens = [], [], []
    ref = torch.tensor([0.0, -1.0, 0.0], device=J.device)
    for a, b in bones:
        d = J[b] - J[a]
        L = d.norm() + 1e-6
        x = d / L
        y = ref - (ref @ x) * x
        if y.norm() < 1e-3:
            y = torch.tensor([1.0, 0, 0], device=J.device) - x[0] * x
        y = y / y.norm()
        z = torch.linalg.cross(x, y)
        Rs.append(torch.stack([x, y, z], 1))
        ts.append(J[a])
        lens.append(L)
    return torch.stack(Rs), torch.stack(ts), torch.stack(lens)


def skin_weights(P, J, k=3, sigma=0.12, bones=None):
    """soft weights of points P (n,3) to bones by distance to the segment."""
    bones = BONES if bones is None else bones
    d = []
    for a, b in bones:
        A, B = J[a], J[b]
        ab = B - A
        t = ((P - A) @ ab / (ab @ ab + 1e-6)).clamp(0, 1)
        d.append((P - (A + t[:, None] * ab)).norm(dim=1))
    d = torch.stack(d, 1)  # n, nb
    w = torch.exp(-((d / sigma) ** 2))
    top = w.topk(k, dim=1)
    mask = torch.zeros_like(w).scatter_(1, top.indices, 1.0)
    w = w * mask
    weak = w.sum(1) < 1e-6
    w[weak] = 0
    w[weak, d[weak].argmin(1)] = (
        1.0  # never let a row vanish (it would collapse the point onto the world origin)
    )
    return w / (w.sum(1, keepdim=True) + 1e-9)


def bone_distance(P, J, bones=None):
    bones = BONES if bones is None else bones
    d = []
    for a, b in bones:
        A, B = J[a], J[b]
        ab = B - A
        t = ((P - A) @ ab / (ab @ ab + 1e-6)).clamp(0, 1)
        d.append((P - (A + t[:, None] * ab)).norm(dim=1))
    return torch.stack(d, 1).min(1).values


def pose_gaussians(can, W, Rc, tc, Rf, tf):
    """canonical means/quats -> frame via LBS. Rc,tc: canonical bone frames; Rf,tf: frame bone frames."""
    # per-bone transform: x_f = Rf Rc^T (x - tc) + tf
    M_R = torch.einsum("bij,bkj->bik", Rf, Rc)  # (nb,3,3) = Rf @ Rc^T
    M_t = tf - torch.einsum("bij,bj->bi", M_R, tc)  # (nb,3)
    Rw = torch.einsum("nb,bij->nij", W, M_R)
    tw = W @ M_t  # blended (not exactly rigid, fine for splats)
    means = torch.einsum("nij,nj->ni", Rw, can["means"]) + tw
    # rotate quats by the dominant bone's rotation
    dom = W.argmax(1)
    Rd = M_R[dom]
    q = F.normalize(can["quats"], dim=-1)
    w, x, y, z = q.unbind(-1)
    Rq = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).view(-1, 3, 3)
    R = Rd @ Rq
    # matrix -> quat (w,x,y,z), batched, numerically simple version
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    qw = torch.sqrt((1 + tr).clamp(min=1e-6)) / 2
    qx = (R[:, 2, 1] - R[:, 1, 2]) / (4 * qw)
    qy = (R[:, 0, 2] - R[:, 2, 0]) / (4 * qw)
    qz = (R[:, 1, 0] - R[:, 0, 1]) / (4 * qw)
    quats = torch.stack([qw, qx, qy, qz], -1)
    return {
        "means": means,
        "quats": quats,
        "scales": can["scales"],
        "opacities": can["opacities"],
        "sh0": can["sh0"],
        "shN": can["shN"],
    }


def checkpoint_bones(checkpoint):
    """Recover and validate the topology before posing a saved body."""
    bones = [tuple(b) for b in checkpoint.get("bones", BONES)]
    n = len(bones)
    if (
        checkpoint["W"].shape[1] != n
        or checkpoint["Rc"].shape != (n, 3, 3)
        or checkpoint["tc"].shape != (n, 3)
    ):
        raise ValueError("Checkpoint bone topology does not match W/Rc/tc dimensions")
    joints = checkpoint["J"].shape[-2]
    if not bones or any(
        len(b) != 2 or any(not isinstance(i, int) or i < 0 or i >= joints for i in b)
        for b in bones
    ):
        raise ValueError("Checkpoint bone topology contains invalid joint indices")
    return bones
