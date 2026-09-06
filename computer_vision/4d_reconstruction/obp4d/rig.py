"""The eight-camera OptiTrack rig from the published calibration (meters, OpenCV camera convention, world
frame = camera 19), and the ring geometry used to place virtual cameras between the real ones."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

CALIB = Path(__file__).resolve().parents[2] / "calibration" / "results"
CAMS = list(range(15, 23))
W, H = 1280, 720


def load_rig():
    """{camera: (R, t, K)} with X_camera = R X_world + t, in meters"""
    intr = {
        r["camera"]: r
        for r in csv.DictReader(
            (CALIB / "intrinsics_summary.csv").read_text().splitlines()
        )
    }
    poses = json.loads(Path(CALIB / "optitrack_rig_provisional.json").read_text())[
        "camera_poses"
    ]
    rig = {}
    for c in CAMS:
        p, i = poses[str(c)], intr[f"optitrack_{c}"]
        assert json.loads(i["distortion"]) == [0.0] * 5, (
            f"camera {c} was calibrated with distortion"
        )
        K = np.array(
            [
                [float(i["fx_px"]), 0, float(i["cx_px"])],
                [0, float(i["fy_px"]), float(i["cy_px"])],
                [0, 0, 1],
            ]
        )
        rig[c] = (
            np.array(p["reference_to_camera_rotation_matrix"]),
            np.array(p["reference_to_camera_translation_mm"]) / 1000,
            K,
        )
    return rig


def to_torch(rig, dev="cuda"):
    """world-to-camera (1, 4, 4) and intrinsics (1, 3, 3) per camera, as gsplat takes them"""
    vm, K = {}, {}
    for c, (R, t, Kc) in rig.items():
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = t
        vm[c] = torch.tensor(m, dtype=torch.float32, device=dev)[None]
        K[c] = torch.tensor(Kc, dtype=torch.float32, device=dev)[None]
    return vm, K


def centers(rig):
    return {c: -R.T @ t for c, (R, t, _) in rig.items()}


def ring(rig):
    """camera-to-world per camera, and the cameras sorted by angle in their best-fit plane"""
    c2w = {}
    for c, (R, t, _) in rig.items():
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = t
        c2w[c] = np.linalg.inv(m)
    C = np.stack([c2w[c][:3, 3] for c in CAMS])
    m = C.mean(0)
    _, _, Vt = np.linalg.svd(C - m)
    u, v = Vt[0], np.cross(Vt[2], Vt[0])
    order = sorted(
        CAMS, key=lambda c: np.arctan2((c2w[c][:3, 3] - m) @ v, (c2w[c][:3, 3] - m) @ u)
    )
    return c2w, order


def ring_pose(c2w, order, s):
    """world-to-camera at arc-length fraction s in [0, 8] around the ring: position linear and rotation slerped
    between the two neighboring real cameras, so the virtual camera moves at constant speed and passes through
    every real pose"""
    L = np.array(
        [
            np.linalg.norm(c2w[order[(g + 1) % 8]][:3, 3] - c2w[order[g]][:3, 3])
            for g in range(8)
        ]
    )
    cum = np.concatenate([[0], np.cumsum(L)])
    d = s / 8 * cum[-1]
    g = min(int(np.searchsorted(cum, d, side="right") - 1), 7)
    a = min(max((d - cum[g]) / L[g], 0.0), 1.0)
    ca, cb = order[g], order[(g + 1) % 8]
    m = np.eye(4)
    m[:3, :3] = Slerp([0, 1], Rotation.from_matrix([c2w[ca][:3, :3], c2w[cb][:3, :3]]))(
        a
    ).as_matrix()
    m[:3, 3] = (1 - a) * c2w[ca][:3, 3] + a * c2w[cb][:3, 3]
    return np.linalg.inv(m)


def look_at(eye, target, up):
    """world-to-camera for a camera at eye looking at target (x right, y down, z forward)"""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return m
