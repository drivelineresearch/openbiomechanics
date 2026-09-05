"""Theia3D segment transforms in the rig frame. The C3D stores one 4x4 transform per body segment per frame at
360 Hz, in the lab frame in millimeters. theia_alignment.json holds the similarity that takes the lab frame into
the rig frame (fitted once on this trial, see the README) and the frame offset between the C3D and the video."""

import json
from pathlib import Path

import ezc3d
import numpy as np

ALIGN = json.load(open(Path(__file__).resolve().parents[1] / "theia_alignment.json"))
SCALE, R, T = float(ALIGN["scale"]), np.array(ALIGN["rotation"]), np.array(ALIGN["translation"])
FRAME_OFFSET = int(ALIGN["c3d_frame_minus_video_frame"])
UP = R[:, 2]  # the lab's vertical axis in rig coordinates


def load_segments(path):
    """T (frames, segments, 4, 4) in meters in the rig frame, and the segment names"""
    c = ezc3d.c3d(str(path))
    names = [label.replace("_4X4", "") for label in c["parameters"]["ROTATION"]["LABELS"]["value"]]
    raw = np.transpose(c["data"]["rotations"], (3, 2, 0, 1))
    assert raw.shape[1:] == (len(names), 4, 4) and np.isfinite(raw).all(), raw.shape
    T_all = np.tile(np.eye(4), (raw.shape[0], raw.shape[1], 1, 1))
    T_all[..., :3, :3] = R @ raw[..., :3, :3]
    T_all[..., :3, 3] = SCALE * (raw[..., :3, 3] @ R.T) + T
    return T_all, names
