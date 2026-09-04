"""Item 3: compare our 8-camera triangulated skeleton against Driveline's Theia3D output for the same trial.

Theia's C3D carries 19 segment 4x4 poses per frame at 360 Hz in Theia's lab frame (mm). Segment origins are
joint centres, so hips, knees, ankles, shoulders, elbows and wrists correspond to COCO joints. We fit ONE
similarity transform (scale, rotation, translation) from our camera-19 frame to Theia's lab frame on all
frames at once, then report per-joint residuals. The transform is also the lab-frame alignment the repo's
roadmap asks for (Theia's frame has gravity along +Z).

  python theia_compare.py --c3d data/theia/BaseballThrow_001.c3d --frames data/throw_full --first_video_frame 820
Writes theia_joints.npz (Theia joints in our frame for every dataset frame) and prints the residual table.
"""
import argparse
import json
from pathlib import Path

import ezc3d
import numpy as np

# COCO index -> Theia segment whose origin is that joint
MAP = {5: "l_uarm_4X4", 6: "r_uarm_4X4", 7: "l_larm_4X4", 8: "r_larm_4X4", 9: "l_hand_4X4", 10: "r_hand_4X4",
       11: "l_thigh_4X4", 12: "r_thigh_4X4", 13: "l_shank_4X4", 14: "r_shank_4X4", 15: "l_foot_4X4", 16: "r_foot_4X4"}
# extra segment origins with no COCO counterpart, appended after index 16: head, toes, pelvis, torso
EXTRA = ["head_4X4", "l_toes_4X4", "r_toes_4X4", "pelvis_4X4", "torso_4X4"]
NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src); U, sig, Vt = np.linalg.svd(cov)
    d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ d @ Vt; s = np.trace(np.diag(sig) @ d) / (S ** 2).sum() * len(src)
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--c3d", required=True); ap.add_argument("--frames", required=True); ap.add_argument("--first_video_frame", type=int, required=True)
    ap.add_argument("--out", default=""); a = ap.parse_args()
    c = ezc3d.c3d(a.c3d); labels = list(c["parameters"]["ROTATION"]["LABELS"]["value"]); rot = c["data"]["rotations"]   # (4,4,19,T)
    theia = {k: rot[:3, 3, labels.index(v), :].T / 1000.0 for k, v in MAP.items()}                                     # metres, (T,3)
    frames = sorted(Path(a.frames).glob("t0*")); ours, ref, frame_ids, joint_ids = [], [], [], []
    for i, f in enumerate(frames):
        vf = json.load(open(f / "meta.json"))["frame"]   # dataset frame index == video frame index == Theia frame index (genlocked)
        s = json.load(open(f / "skeleton.json"))
        for j, seg in MAP.items():
            if s["joints"][j] is None or not np.isfinite(s["reproj_px"][j]): continue
            ours.append(s["joints"][j]); ref.append(theia[j][vf]); frame_ids.append(i); joint_ids.append(j)
    ours, ref = np.array(ours), np.array(ref)
    s, R, t = umeyama(ours, ref)
    res = np.linalg.norm((s * (R @ ours.T).T + t) - ref, axis=1)
    print(f"correspondences {len(ours)} over {len(frames)} frames; similarity scale {s:.4f} (1.0 = our calibration is metric)")
    print(f"residual vs Theia3D: median {np.median(res)*100:.1f} cm, mean {res.mean()*100:.1f} cm, p90 {np.percentile(res,90)*100:.1f} cm")
    jid = np.array(joint_ids)
    for j in MAP:
        m = jid == j; print(f"  {NAMES[j]:11s} n={m.sum():4d} median {np.median(res[m])*100:5.1f} cm  p90 {np.percentile(res[m],90)*100:5.1f} cm")
    # gravity direction of Theia's lab frame expressed in our camera-19 frame (Theia: +Z up)
    up_ours = R.T @ np.array([0, 0, 1.0]); print("lab up-vector in cam19 frame:", np.round(up_ours, 3))
    if a.out:
        Rinv, sinv = R.T, 1.0 / s
        J = np.full((len(frames), 17 + len(EXTRA), 3), np.nan, np.float32)
        ex = {17 + k: rot[:3, 3, labels.index(v), :].T / 1000.0 for k, v in enumerate(EXTRA)}
        for i, f in enumerate(frames):
            vf = json.load(open(f / "meta.json"))["frame"]
            for j in MAP: J[i, j] = sinv * (Rinv @ (theia[j][vf] - t))
            for j, arr in ex.items(): J[i, j] = sinv * (Rinv @ (arr[vf] - t))
        np.savez(a.out, joints=J, scale=s, R=R, t=t, up_cam19=up_ours, extra_names=EXTRA, frame_video=[json.load(open(f / "meta.json"))["frame"] for f in frames])
        print("extra joints appended:", {17 + k: v for k, v in enumerate(EXTRA)})
        print("wrote", a.out)


if __name__ == "__main__":
    main()
