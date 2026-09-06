"""Theia3D segment transforms in the rig frame. The C3D stores one 4x4 transform per body segment per frame at
360 Hz, in the lab frame in millimeters. theia_alignment.json holds the similarity that takes the lab frame into
the rig frame (fitted once on this trial, see the README) and the frame offset between the C3D and the video."""

import hashlib
import json
from pathlib import Path

import ezc3d
import numpy as np

ALIGN_PATH = Path(__file__).resolve().parents[1] / "theia_alignment.json"
ALIGN = json.loads(ALIGN_PATH.read_text())
ALIGNMENT_SHA256 = hashlib.sha256(ALIGN_PATH.read_bytes()).hexdigest()
SCALE, R, T = (
    float(ALIGN["scale"]),
    np.array(ALIGN["rotation"]),
    np.array(ALIGN["translation"]),
)
FRAME_OFFSET = int(ALIGN["c3d_frame_minus_video_frame"])
UP = R[:, 2]  # the lab's vertical axis in rig coordinates


def load_segments(path):
    """T (frames, segments, 4, 4) in meters in the rig frame, and the segment names"""
    c = ezc3d.c3d(str(path))
    names = [
        label.replace("_4X4", "")
        for label in c["parameters"]["ROTATION"]["LABELS"]["value"]
    ]
    raw = np.transpose(c["data"]["rotations"], (3, 2, 0, 1))
    assert raw.shape[1:] == (len(names), 4, 4) and np.isfinite(raw).all(), raw.shape
    if (
        SCALE <= 0
        or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(R), 1.0)
    ):
        raise ValueError("Alignment must have a positive scale and a proper rotation")
    T_all = np.tile(np.eye(4), (raw.shape[0], raw.shape[1], 1, 1))
    T_all[..., :3, :3] = R @ raw[..., :3, :3]
    T_all[..., :3, 3] = SCALE * (raw[..., :3, 3] @ R.T) + T
    return T_all, names


def video_to_c3d_index(video_frame, frame_count):
    """Apply the contributor-supplied array-index offset, rejecting out-of-range frames."""
    index = video_frame + FRAME_OFFSET
    if not 0 <= index < frame_count:
        raise ValueError(
            f"Video frame {video_frame} with offset {FRAME_OFFSET} is outside {frame_count} C3D frames"
        )
    return index


def evaluation_protocol(holdout, frames):
    """Describe the scope of the supplied-pose rendering experiment."""
    return {
        "evaluation": "appearance holdout conditioned on supplied Theia poses and fitted alignment",
        "appearance_excluded_cameras": [holdout] if holdout else [],
        "alignment_fit_cameras": ALIGN["fit_cameras"],
        "alignment_sha256": ALIGNMENT_SHA256,
        "alignment_fit_code_available": ALIGN["fit_code_available"],
        "theia_input_camera_provenance": "not established by this pipeline",
        "strict_independent_camera_evaluation": False,
        "c3d_frame_minus_video_frame": FRAME_OFFSET,
        "training_video_frames": list(frames),
        "scored_video_frames": list(frames)[::10],
        "scoring_images": "extracted RGB PNG frames",
        "scoring_mask": "BiRefNet at source resolution, threshold > 127/255",
    }
