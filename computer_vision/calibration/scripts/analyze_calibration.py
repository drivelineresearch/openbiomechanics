#!/usr/bin/env python3
"""Empirical calibration audit for the public OpenBiomechanics CV videos."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/tmp/obp-calibration-assets")
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
PATTERN = (7, 4)
SQUARE_MM = 100.0
OBJ = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
OBJ[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    result = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return result


def detect_video(path: Path, stride: int) -> tuple[dict, dict[int, np.ndarray]]:
    info = video_info(path)
    cap = cv2.VideoCapture(str(path))
    found: dict[int, np.ndarray] = {}
    sampled = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = min(1.0, 720.0 / gray.shape[1])
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok2, corners = cv2.findChessboardCornersSB(small, PATTERN, flags)
            if ok2:
                corners = corners.astype(np.float32) / scale
                cv2.cornerSubPix(
                    gray,
                    corners,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )
                found[idx] = corners.reshape(-1, 2)
        idx += 1
    cap.release()
    info.update({"stride": stride, "sampled": sampled, "detected": len(found)})
    return info, found


def board_features(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    hull = cv2.convexHull(corners.astype(np.float32))
    area = cv2.contourArea(hull) / (width * height)
    center = corners.mean(axis=0) / np.array([width, height])
    v = corners[PATTERN[0] - 1] - corners[0]
    angle = math.atan2(float(v[1]), float(v[0])) / math.pi
    return np.array([center[0], center[1], math.sqrt(max(area, 0)), angle], dtype=float)


def select_diverse(dets: dict[int, np.ndarray], width: int, height: int, limit: int = 120) -> list[int]:
    ids = sorted(dets)
    if len(ids) <= limit:
        return ids
    feats = np.array([board_features(dets[i], width, height) for i in ids])
    lo = feats.min(axis=0)
    span = np.maximum(feats.max(axis=0) - lo, 1e-9)
    z = (feats - lo) / span
    chosen = [int(np.argmax(z[:, 2]))]
    min_d = np.full(len(ids), np.inf)
    for _ in range(1, limit):
        d = np.sum((z - z[chosen[-1]]) ** 2, axis=1)
        min_d = np.minimum(min_d, d)
        min_d[chosen] = -1
        chosen.append(int(np.argmax(min_d)))
    return sorted(ids[i] for i in chosen)


def reprojection_errors(obj, img, rvecs, tvecs, K, D) -> np.ndarray:
    vals = []
    for ip, rv, tv in zip(img, rvecs, tvecs):
        pred, _ = cv2.projectPoints(obj, rv, tv, K, D)
        vals.extend(np.linalg.norm(pred.reshape(-1, 2) - ip.reshape(-1, 2), axis=1))
    return np.asarray(vals)


def calibrate_camera(dets: dict[int, np.ndarray], info: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    selected = select_diverse(dets, info["width"], info["height"])
    test_ids = selected[::5]
    train_ids = [i for i in selected if i not in set(test_ids)]
    obj_train = [OBJ.copy() for _ in train_ids]
    img_train = [dets[i].reshape(-1, 1, 2).astype(np.float32) for i in train_ids]
    # Use a deliberately low-dimensional Brown model. The board does not reach
    # every image corner in every camera, so unconstrained k3/tangential terms
    # can fit the observed region while becoming physically implausible outside it.
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_train, img_train, (info["width"], info["height"]), None, None, flags=flags
    )
    train_e = reprojection_errors(OBJ, img_train, rvecs, tvecs, K, D)
    test_e = []
    solved_test = 0
    for i in test_ids:
        ip = dets[i].reshape(-1, 1, 2).astype(np.float32)
        ok, rv, tv = cv2.solvePnP(OBJ, ip, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            pred, _ = cv2.projectPoints(OBJ, rv, tv, K, D)
            test_e.extend(np.linalg.norm(pred.reshape(-1, 2) - ip.reshape(-1, 2), axis=1))
            solved_test += 1
    test_e = np.asarray(test_e)
    out = {
        "n_detected": len(dets),
        "n_selected": len(selected),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_test_solved": solved_test,
        "opencv_rms_px": float(rms),
        "train_median_px": float(np.median(train_e)),
        "train_p95_px": float(np.percentile(train_e, 95)),
        "test_median_px": float(np.median(test_e)),
        "test_p95_px": float(np.percentile(test_e, 95)),
        "fx_px": float(K[0, 0]),
        "fy_px": float(K[1, 1]),
        "cx_px": float(K[0, 2]),
        "cy_px": float(K[1, 2]),
        "distortion": D.reshape(-1).tolist(),
        "selected_frames": selected,
        "test_frames": test_ids,
    }
    return out, K, D


def rotation_angle(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def stereo_pair(
    da: dict[int, np.ndarray], db: dict[int, np.ndarray], ia: dict, ib: dict,
    Ka: np.ndarray, Da: np.ndarray, Kb: np.ndarray, Db: np.ndarray,
) -> dict:
    shared = sorted(set(da) & set(db))
    if len(shared) > 100:
        shared = shared[:: max(1, len(shared) // 100)][:100]
    if len(shared) < 5:
        return {"n_shared": len(shared), "status": "insufficient_overlap"}
    objs = [OBJ.copy() for _ in shared]
    ipa = [da[i].reshape(-1, 1, 2).astype(np.float32) for i in shared]
    ipb = [db[i].reshape(-1, 1, 2).astype(np.float32) for i in shared]
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objs, ipa, ipb, Ka.copy(), Da.copy(), Kb.copy(), Db.copy(),
        (ia["width"], ia["height"]),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )
    # Independent per-frame estimates quantify stability and expose ordering failures.
    rot_d, trans_d = [], []
    for i in shared:
        oka, rva, tva = cv2.solvePnP(OBJ, da[i], Ka, Da)
        okb, rvb, tvb = cv2.solvePnP(OBJ, db[i], Kb, Db)
        if not (oka and okb):
            continue
        Ra, _ = cv2.Rodrigues(rva)
        Rb, _ = cv2.Rodrigues(rvb)
        Rij = Rb @ Ra.T
        Tij = tvb - Rij @ tva
        rot_d.append(rotation_angle(Rij @ R.T))
        trans_d.append(float(np.linalg.norm(Tij.reshape(3, 1) - T.reshape(3, 1))))
    return {
        "status": "fit",
        "n_shared": len(shared),
        "stereo_rms_px": float(rms),
        "rotation_deg": rotation_angle(R),
        "rotation_matrix": R.tolist(),
        "translation_mm": T.reshape(-1).tolist(),
        "baseline_mm": float(np.linalg.norm(T)),
        "per_frame_rotation_delta_median_deg": float(np.median(rot_d)),
        "per_frame_rotation_delta_p95_deg": float(np.percentile(rot_d, 95)),
        "per_frame_translation_delta_median_mm": float(np.median(trans_d)),
        "per_frame_translation_delta_p95_mm": float(np.percentile(trans_d, 95)),
    }


def main() -> None:
    jobs = []
    for cam in range(15, 23):
        jobs.append(("optitrack", str(cam), ROOT / f"opti_dynamic_checkerboard_cam{cam}.mp4", 40))
    for cam in range(1, 5):
        jobs.append(("edgertronic", str(cam), ROOT / f"edge_dynamic_checkerboard_cam{cam}.mp4", 6))
    jobs.append(("iphone", "iphone", ROOT / "iphone_dynamic_checkerboard.MOV", 6))

    detections = {}
    infos = {}
    for system, camera, path, stride in jobs:
        key = f"{system}_{camera}"
        print(f"detect {key}: {path.name}", flush=True)
        info, det = detect_video(path, stride)
        infos[key] = info
        detections[key] = det
        print(f"  {len(det)}/{info['sampled']} sampled frames", flush=True)

    calibrations, matrices = {}, {}
    for key, det in detections.items():
        print(f"calibrate {key}", flush=True)
        if len(det) < 12:
            calibrations[key] = {"status": "insufficient_detections", "n_detected": len(det)}
            continue
        result, K, D = calibrate_camera(det, infos[key])
        result["status"] = "fit"
        calibrations[key] = result
        matrices[key] = (K, D)
        print(f"  held-out median={result['test_median_px']:.3f}px p95={result['test_p95_px']:.3f}px", flush=True)

    pairs = {}
    for system, cams in (("optitrack", [str(x) for x in range(15, 23)]), ("edgertronic", [str(x) for x in range(1, 5)])):
        for ai, a in enumerate(cams):
            for b in cams[ai + 1:]:
                ka, kb = f"{system}_{a}", f"{system}_{b}"
                label = f"{ka}__{kb}"
                if ka not in matrices or kb not in matrices:
                    pairs[label] = {"status": "missing_intrinsics"}
                    continue
                print(f"stereo {label}", flush=True)
                pairs[label] = stereo_pair(
                    detections[ka], detections[kb], infos[ka], infos[kb],
                    *matrices[ka], *matrices[kb],
                )
                print(f"  {pairs[label]}", flush=True)

    payload = {"pattern_inner_corners": PATTERN, "square_mm": SQUARE_MM,
               "video_info": infos, "calibrations": calibrations, "stereo_pairs": pairs}
    (OUT / "calibration_results.json").write_text(json.dumps(payload, indent=2))

    with (OUT / "intrinsics_summary.csv").open("w", newline="") as f:
        cols = ["camera", "status", "n_detected", "n_selected", "opencv_rms_px", "test_median_px",
                "test_p95_px", "fx_px", "fy_px", "cx_px", "cy_px", "distortion"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key, row in calibrations.items():
            w.writerow({c: key if c == "camera" else row.get(c) for c in cols})
    with (OUT / "extrinsics_summary.csv").open("w", newline="") as f:
        cols = ["pair", "status", "n_shared", "stereo_rms_px", "baseline_mm",
                "per_frame_rotation_delta_median_deg", "per_frame_rotation_delta_p95_deg",
                "per_frame_translation_delta_median_mm", "per_frame_translation_delta_p95_mm"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key, row in pairs.items():
            w.writerow({c: key if c == "pair" else row.get(c) for c in cols})
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
