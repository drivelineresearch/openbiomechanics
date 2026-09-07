"""Compare relative C3D-to-video offsets with a refitted similarity at each shift.

This module applies `C3D index = video index + 1` and the splat contribution applies 0;
interpolating the C3D and refitting on the supplied 30-frame reference window puts the
in-sample residual minimum near half a frame. This relative fit does not establish physical
sample synchronization, a triangulation-error floor, or effects on rendering scores.
The shipped transforms and integer offsets are unchanged.

Reference positions are the joint centers triangulated from the eight cameras over the
release window, shipped as release_keypoints.csv in the camera-19 metric frame.

    python computer_vision/4d_reconstruction/offset_study.py --c3d <work>/theia.c3d
"""

import argparse
import csv
from pathlib import Path

import ezc3d
import numpy as np

SEGMENTS = [
    "l_thigh",
    "r_thigh",
    "l_shank",
    "r_shank",
    "l_foot",
    "r_foot",
    "l_uarm",
    "r_uarm",
    "l_larm",
    "r_larm",
    "l_hand",
    "r_hand",
]
KEYPOINTS = Path(__file__).resolve().parent / "release_keypoints.csv"
SHIFTS = np.arange(-0.5, 1.75, 0.125)


def load_keypoints():
    """(frames, 12, 3) metres in the rig frame, and the video frame numbers."""
    with KEYPOINTS.open() as stream:
        rows = list(csv.DictReader(stream))
    frames = sorted({int(r["video_frame"]) for r in rows})
    index = {(int(r["video_frame"]), r["segment"]): r for r in rows}
    joints = np.array(
        [
            [[float(index[(f, s)][a]) for a in ("x_m", "y_m", "z_m")] for s in SEGMENTS]
            for f in frames
        ]
    )
    assert joints.shape == (len(frames), len(SEGMENTS), 3) and np.isfinite(joints).all()
    return joints, frames


def umeyama(src, dst):
    """Similarity (scale, rotation, translation) taking src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ correction @ Vt
    s = np.trace(np.diag(sig) @ correction) / (S**2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c3d", required=True, help="theia.c3d, downloaded by obp4d fetch"
    )
    args = parser.parse_args()

    obs, frames = load_keypoints()
    c = ezc3d.c3d(args.c3d)
    labels = c["parameters"]["ROTATION"]["LABELS"]["value"]
    columns = [labels.index(f"{s}_4X4") for s in SEGMENTS]
    origins = c["data"]["rotations"][:3, 3, :, :]  # (3, segment, frame) millimetres

    def theia_at(shift):
        """Segment origins at a fractional C3D index, linearly interpolated."""
        out = []
        for f in frames:
            i = int(np.floor(f + shift))
            w = f + shift - i
            out.append(
                ((1 - w) * origins[:, columns, i] + w * origins[:, columns, i + 1]).T
            )
        return np.stack(out)

    def residuals(shift):
        """Per-frame per-joint distance in centimetres, similarity refitted at this shift."""
        X = theia_at(shift).reshape(-1, 3)
        s, R, t = umeyama(X, obs.reshape(-1, 3))
        return (
            np.linalg.norm((s * (R @ X.T).T + t).reshape(obs.shape) - obs, axis=2) * 100
        )

    hand, forearm = SEGMENTS.index("l_hand"), SEGMENTS.index("l_larm")
    table = np.array(
        [
            [np.median(r), np.median(r[:, hand]), np.median(r[:, forearm])]
            for r in (residuals(d) for d in SHIFTS)
        ]
    )

    print(
        f"{'offset':>7} {'all joints':>11} {'throwing hand':>14} {'throwing forearm':>17}"
    )
    for shift, (a, h, f) in zip(SHIFTS, table):
        print(f"{shift:+7.3f} {a:10.2f}cm {h:13.2f}cm {f:16.2f}cm")
    for column, name in enumerate(("all joints", "throwing hand", "throwing forearm")):
        print(
            f"minimum for {name:17s} at {SHIFTS[table[:, column].argmin()]:+.3f} frames"
        )

    best = SHIFTS[table[:, 0].argmin()]
    minimum_residual = table[:, 0].min()
    assert 0.25 <= best <= 0.75, (
        f"expected a sub-frame shift near half a frame, got {best}"
    )
    print(
        f"\nbest whole-body shift {best:+.3f} frames = {best / 360 * 1000:.2f} ms; "
        f"integer 0 adds {table[SHIFTS == 0, 0][0] - minimum_residual:.2f} cm and "
        f"integer +1 adds {table[SHIFTS == 1, 0][0] - minimum_residual:.2f} cm "
        "to the in-sample median residual after refitting at each shift"
    )
    print(
        "This relative fit does not establish physical synchronization, "
        "a triangulation-error floor, or effects on rendering scores."
    )


if __name__ == "__main__":
    main()
