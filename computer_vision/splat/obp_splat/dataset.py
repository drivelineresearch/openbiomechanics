"""Convert OpenBiomechanics' provisional OptiTrack calibration + one synchronized 8-camera recording into a
COLMAP-format dataset (one frame index per dataset) that gsplat's simple_trainer can consume.

World frame = camera 19 frame, converted from millimetres to metres, with the checkerboard-audit
intrinsics (square pixels, zero distortion for all eight OptiTrack cameras). No SfM points exist; the
trainer is run with random initialization inside the capture volume (--init_type random).

Usage:
  python obp_to_colmap.py --videos data/prime_color/event13 --frame 1800 --out data/colmap/throw_f1800
  (videos named opti_<anything>_cam15.mp4 ... cam22.mp4; frame index is the same in every genlocked feed)
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAL = HERE.parents[1] / "calibration" / "results"   # computer_vision/calibration/results


def rot_to_quat(R):
    """COLMAP quaternion (qw, qx, qy, qz) for a rotation matrix."""
    q = np.empty(4); t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; q[0] = 0.25 * s; q[1] = (R[2, 1] - R[1, 2]) / s; q[2] = (R[0, 2] - R[2, 0]) / s; q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q[0] = (R[2, 1] - R[1, 2]) / s; q[1] = 0.25 * s; q[2] = (R[0, 1] + R[1, 0]) / s; q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q[0] = (R[0, 2] - R[2, 0]) / s; q[1] = (R[0, 1] + R[1, 0]) / s; q[2] = 0.25 * s; q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q[0] = (R[1, 0] - R[0, 1]) / s; q[1] = (R[0, 2] + R[2, 0]) / s; q[2] = (R[1, 2] + R[2, 1]) / s; q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def load_calibration():
    intr = {}
    for r in csv.DictReader(open(CAL / "intrinsics_summary.csv")):
        if r["camera"].startswith("optitrack_"):
            intr[r["camera"].split("_")[1]] = dict(fx=float(r["fx_px"]), fy=float(r["fy_px"]), cx=float(r["cx_px"]), cy=float(r["cy_px"]), dist=json.loads(r["distortion"]))
    rig = json.load(open(CAL / "optitrack_rig_provisional.json"))
    poses = {k: (np.array(v["reference_to_camera_rotation_matrix"]), np.array(v["reference_to_camera_translation_mm"]) / 1000.0) for k, v in rig["camera_poses"].items()}
    return intr, poses


def grab_frame(video, idx):
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = cap.read(); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if not ok:
        raise RuntimeError(f"frame {idx} not readable in {video} ({n} frames)")
    return f, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True); ap.add_argument("--frame", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=1, help="number of frames to export as separate time folders")
    ap.add_argument("--stride", type=int, default=1, help="source-frame stride between exported frames")
    a = ap.parse_args()
    intr, poses = load_calibration()
    vids = {m.group(1): p for p in Path(a.videos).glob("*.mp4") for m in [re.search(r"cam(\d+)", p.name)] if m}
    cams = sorted(set(vids) & set(intr) & set(poses), key=int)
    assert len(cams) >= 3, f"found cameras {sorted(vids)}; calibrated {sorted(intr)}"
    for t in range(a.frames):
        fi = a.frame + t * a.stride
        out = Path(a.out) if a.frames == 1 else Path(a.out) / f"t{t:04d}"
        (out / "images").mkdir(parents=True, exist_ok=True); (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        cam_lines, img_lines = [], []
        for i, c in enumerate(cams, 1):
            img, n = grab_frame(vids[c], fi); h, w = img.shape[:2]
            cv2.imwrite(str(out / "images" / f"cam{c}.png"), img)
            k = intr[c]; cam_lines.append(f"{i} PINHOLE {w} {h} {k['fx']} {k['fy']} {k['cx']} {k['cy']}")
            R, tt = poses[c]; q = rot_to_quat(R)
            img_lines.append(f"{i} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} {tt[0]:.6f} {tt[1]:.6f} {tt[2]:.6f} {i} cam{c}.png\n")
        (out / "sparse" / "0" / "cameras.txt").write_text("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + "\n".join(cam_lines) + "\n")
        (out / "sparse" / "0" / "images.txt").write_text("# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + "\n".join(img_lines))
        (out / "sparse" / "0" / "points3D.txt").write_text("# empty: no SfM points; use random init inside the capture volume\n")
        # capture-volume hint: centroid of camera centers and the mean camera-to-centroid distance
        C = np.array([-poses[c][0].T @ poses[c][1] for c in cams]); cen = C.mean(0)
        json.dump(dict(frame=fi, cameras=cams, scene_center_m=cen.tolist(), camera_radius_m=float(np.linalg.norm(C - cen, axis=1).mean()), n_frames_in_video=int(n)), open(out / "meta.json", "w"), indent=1)
        print(f"wrote {out}: {len(cams)} cameras, frame {fi}/{n}, center {np.round(cen,2)}")


if __name__ == "__main__":
    main()
