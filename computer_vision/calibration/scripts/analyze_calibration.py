#!/usr/bin/env python3
"""Coverage-aware intrinsic and stereo calibration for the OBP-CV videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ASSETS = Path(os.environ.get("OBP_CV_ASSETS", "/tmp/obp-calibration-assets"))
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results"
PATTERN = (7, 4)
SQUARE_MM = 100.0
OBJ = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
OBJ[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

ZERO_TANGENT = cv2.CALIB_ZERO_TANGENT_DIST
FIX_K1 = cv2.CALIB_FIX_K1
FIX_K2 = cv2.CALIB_FIX_K2
FIX_K3 = cv2.CALIB_FIX_K3
FIX_ASPECT = cv2.CALIB_FIX_ASPECT_RATIO
MODEL_SPECS = (
    {"name": "zero_square", "flags": ZERO_TANGENT | FIX_K1 | FIX_K2 | FIX_K3 | FIX_ASPECT, "complexity": 0},
    {"name": "k1_square", "flags": ZERO_TANGENT | FIX_K2 | FIX_K3 | FIX_ASPECT, "complexity": 1},
    {"name": "k1_k2_square", "flags": ZERO_TANGENT | FIX_K3 | FIX_ASPECT, "complexity": 2},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-intrinsic-views", type=int, default=120)
    parser.add_argument("--max-stereo-views", type=int, default=120)
    parser.add_argument(
        "--reuse-intrinsics",
        action="store_true",
        help="Reuse matching schema-v2 intrinsics in output-dir and recompute only stereo results",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required calibration asset is missing: {path}")
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def software_versions() -> dict:
    try:
        ffprobe = subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        ffprobe = None
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ffprobe": ffprobe,
        "platform": platform.platform(),
    }


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    result = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if result["width"] <= 0 or result["height"] <= 0:
        raise RuntimeError(f"Video has invalid dimensions: {path}")
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
    vector = corners[PATTERN[0] - 1] - corners[0]
    angle = math.atan2(float(vector[1]), float(vector[0]))
    return np.array(
        [center[0], center[1], math.sqrt(max(area, 0)), math.sin(angle), math.cos(angle)], dtype=float
    )


def select_diverse(dets: dict[int, np.ndarray], width: int, height: int, limit: int = 120) -> list[int]:
    ids = sorted(dets)
    if len(ids) <= limit:
        return ids
    features = np.array([board_features(dets[i], width, height) for i in ids])
    lo = features.min(axis=0)
    span = np.maximum(features.max(axis=0) - lo, 1e-9)
    normalized = (features - lo) / span
    chosen = [int(np.argmax(normalized[:, 2]))]
    min_distance = np.full(len(ids), np.inf)
    for _ in range(1, limit):
        distance = np.sum((normalized - normalized[chosen[-1]]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen] = -1
        chosen.append(int(np.argmax(min_distance)))
    return sorted(ids[i] for i in chosen)


def select_stratified(ids: list[int], limit: int) -> list[int]:
    """Return time-stratified IDs including both endpoints."""
    if len(ids) <= limit:
        return list(ids)
    indexes = np.linspace(0, len(ids) - 1, num=limit)
    return [ids[i] for i in sorted(set(np.rint(indexes).astype(int).tolist()))]


def fit_camera(ids: list[int], dets: dict[int, np.ndarray], info: dict, flags: int):
    objects = [OBJ.copy() for _ in ids]
    images = [dets[i].reshape(-1, 1, 2).astype(np.float32) for i in ids]
    return cv2.calibrateCamera(objects, images, (info["width"], info["height"]), None, None, flags=flags)


def per_view_errors(ids, dets, rvecs, tvecs, K, D) -> dict[int, float]:
    result = {}
    for frame, rvec, tvec in zip(ids, rvecs, tvecs):
        predicted, _ = cv2.projectPoints(OBJ, rvec, tvec, K, D)
        error = np.linalg.norm(predicted.reshape(-1, 2) - dets[frame].reshape(-1, 2), axis=1)
        result[frame] = float(np.sqrt(np.mean(error**2)))
    return result


def heldout_errors(ids, dets, K, D) -> list[float]:
    errors = []
    for frame in ids:
        image = dets[frame].reshape(-1, 1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(OBJ, image, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        predicted, _ = cv2.projectPoints(OBJ, rvec, tvec, K, D)
        errors.extend(np.linalg.norm(predicted.reshape(-1, 2) - image.reshape(-1, 2), axis=1))
    return errors


def image_radius(K: np.ndarray, width: int, height: int) -> float:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return max(math.hypot((x - cx) / fx, (y - cy) / fy) for x in (0, width - 1) for y in (0, height - 1))


def radial_validity(K: np.ndarray, D: np.ndarray, width: int, height: int) -> dict:
    k1, k2 = float(D.reshape(-1)[0]), float(D.reshape(-1)[1])
    maximum = image_radius(K, width, height)
    radii = np.linspace(0.0, maximum, 4097)
    r2 = radii**2
    scale = 1.0 + k1 * r2 + k2 * r2**2
    derivative = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2**2
    bad = np.flatnonzero((scale <= 0.0) | (derivative <= 0.0))
    first_invalid = float(radii[bad[0]]) if len(bad) else None
    return {
        "full_frame_max_normalized_radius": float(maximum),
        "minimum_radial_scale": float(scale.min()),
        "minimum_radial_derivative": float(derivative.min()),
        "first_invalid_normalized_radius": first_invalid,
        "monotonic_over_full_frame": not len(bad),
    }


def coverage_metrics(ids, dets, info, K) -> dict:
    points = np.concatenate([dets[i] for i in ids])
    normalized_radius = np.hypot(
        (points[:, 0] - K[0, 2]) / K[0, 0],
        (points[:, 1] - K[1, 2]) / K[1, 1],
    )
    centers = np.array([dets[i].mean(axis=0) for i in ids])
    areas = np.array([
        cv2.contourArea(cv2.convexHull(dets[i].astype(np.float32))) / (info["width"] * info["height"])
        for i in ids
    ])
    return {
        "observed_max_normalized_radius": float(normalized_radius.max()),
        "observed_p99_normalized_radius": float(np.percentile(normalized_radius, 99)),
        "center_x_range_fraction": [float(centers[:, 0].min() / info["width"]), float(centers[:, 0].max() / info["width"])],
        "center_y_range_fraction": [float(centers[:, 1].min() / info["height"]), float(centers[:, 1].max() / info["height"])],
        "board_area_fraction_range": [float(areas.min()), float(areas.max())],
    }


def robust_view_filter(ids, dets, info) -> tuple[list[int], list[int], dict]:
    spec = next(item for item in MODEL_SPECS if item["name"] == "k1_square")
    rms, K, D, rvecs, tvecs = fit_camera(ids, dets, info, spec["flags"])
    errors = per_view_errors(ids, dets, rvecs, tvecs, K, D)
    values = np.array(list(errors.values()))
    median = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values - median)))
    threshold = max(0.75, median + 4.0 * sigma)
    retained = [frame for frame in ids if errors[frame] <= threshold]
    rejected = [frame for frame in ids if errors[frame] > threshold]
    if len(retained) < 12:
        retained, rejected = list(ids), []
    return retained, rejected, {
        "pilot_model": spec["name"],
        "pilot_rms_px": float(rms),
        "median_view_rmse_px": median,
        "robust_sigma_px": sigma,
        "rejection_threshold_px": threshold,
        "per_view_rmse_px": {str(key): value for key, value in errors.items()},
    }


def cross_validate_model(ids, dets, info, spec, folds: int = 3) -> dict:
    all_errors = []
    parameters = []
    physical = []
    solved_views = 0
    for fold in range(folds):
        test = [frame for index, frame in enumerate(ids) if index % folds == fold]
        train = [frame for index, frame in enumerate(ids) if index % folds != fold]
        rms, K, D, _, _ = fit_camera(train, dets, info, spec["flags"])
        errors = heldout_errors(test, dets, K, D)
        all_errors.extend(errors)
        solved_views += len(errors) // len(OBJ)
        parameters.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D.reshape(-1)[:2]])
        physical.append(radial_validity(K, D, info["width"], info["height"]))
    params = np.asarray(parameters)
    focal_mean = np.maximum(np.abs(params[:, :2].mean(axis=0)), 1e-9)
    return {
        "model": spec["name"],
        "complexity_rank": spec["complexity"],
        "folds": folds,
        "heldout_views_solved": solved_views,
        "heldout_median_px": float(np.median(all_errors)),
        "heldout_p95_px": float(np.percentile(all_errors, 95)),
        "fold_fx_cv_pct": float(params[:, 0].std(ddof=1) / focal_mean[0] * 100),
        "fold_fy_cv_pct": float(params[:, 1].std(ddof=1) / focal_mean[1] * 100),
        "fold_principal_point_max_distance_px": float(max(np.linalg.norm(row[2:4] - params[:, 2:4].mean(axis=0)) for row in params)),
        "all_folds_monotonic_full_frame": all(row["monotonic_over_full_frame"] for row in physical),
        "fold_parameters": params.tolist(),
    }


def choose_model(candidates: list[dict]) -> str:
    valid = [row for row in candidates if row["all_folds_monotonic_full_frame"]]
    if not valid:
        raise RuntimeError("No intrinsic candidate is monotonic over the full frame in every fold")
    best_median = min(row["heldout_median_px"] for row in valid)
    best_p95 = min(row["heldout_p95_px"] for row in valid)
    eligible = [
        row for row in valid
        if row["heldout_median_px"] <= best_median + max(0.02, best_median * 0.10)
        and row["heldout_p95_px"] <= best_p95 + max(0.05, best_p95 * 0.10)
    ]
    return min(eligible, key=lambda row: (row["complexity_rank"], row["heldout_p95_px"]))["model"]


def calibrate_camera(dets: dict[int, np.ndarray], info: dict, limit: int = 120) -> tuple[dict, np.ndarray, np.ndarray]:
    selected = select_diverse(dets, info["width"], info["height"], limit)
    retained, rejected, rejection = robust_view_filter(selected, dets, info)
    candidates = [cross_validate_model(retained, dets, info, spec) for spec in MODEL_SPECS]
    model_name = choose_model(candidates)
    spec = next(item for item in MODEL_SPECS if item["name"] == model_name)
    rms, K, D, rvecs, tvecs = fit_camera(retained, dets, info, spec["flags"])
    final_physical = radial_validity(K, D, info["width"], info["height"])
    if not final_physical["monotonic_over_full_frame"]:
        raise RuntimeError(f"Selected model {model_name} is not monotonic in its final fit")
    selected_cv = next(row for row in candidates if row["model"] == model_name)
    view_errors = per_view_errors(retained, dets, rvecs, tvecs, K, D)
    quality = "strong"
    if selected_cv["heldout_p95_px"] > 2.0 or selected_cv["fold_fx_cv_pct"] > 5.0:
        quality = "weak"
    elif selected_cv["heldout_p95_px"] > 1.0 or selected_cv["fold_fx_cv_pct"] > 2.0:
        quality = "caution"
    output = {
        "status": "fit",
        "quality": quality,
        "model": model_name,
        "n_detected": len(dets),
        "n_selected": len(selected),
        "n_retained": len(retained),
        "n_rejected_views": len(rejected),
        "rejected_frames": rejected,
        "opencv_rms_px": float(rms),
        "train_view_rmse_median_px": float(np.median(list(view_errors.values()))),
        "train_view_rmse_p95_px": float(np.percentile(list(view_errors.values()), 95)),
        "test_median_px": selected_cv["heldout_median_px"],
        "test_p95_px": selected_cv["heldout_p95_px"],
        "fx_px": float(K[0, 0]),
        "fy_px": float(K[1, 1]),
        "cx_px": float(K[0, 2]),
        "cy_px": float(K[1, 2]),
        "distortion": D.reshape(-1).tolist(),
        "selected_frames": selected,
        "retained_frames": retained,
        "view_rejection": rejection,
        "coverage": coverage_metrics(retained, dets, info, K),
        "physical_validity": final_physical,
        "cross_validation": selected_cv,
        "candidate_models": candidates,
    }
    return output, K, D


def rotation_angle(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def transform_deltas(ids, da, db, Ka, Da, Kb, Db, R, T) -> dict:
    rotations, translations = [], []
    for frame in ids:
        oka, rva, tva = cv2.solvePnP(OBJ, da[frame], Ka, Da)
        okb, rvb, tvb = cv2.solvePnP(OBJ, db[frame], Kb, Db)
        if not (oka and okb):
            continue
        Ra, _ = cv2.Rodrigues(rva)
        Rb, _ = cv2.Rodrigues(rvb)
        per_frame_R = Rb @ Ra.T
        per_frame_T = tvb - per_frame_R @ tva
        rotations.append(rotation_angle(per_frame_R @ R.T))
        translations.append(float(np.linalg.norm(per_frame_T.reshape(3, 1) - T.reshape(3, 1))))
    if not rotations:
        return {"n_solved": 0}
    return {
        "n_solved": len(rotations),
        "rotation_delta_median_deg": float(np.median(rotations)),
        "rotation_delta_p95_deg": float(np.percentile(rotations, 95)),
        "translation_delta_median_mm": float(np.median(translations)),
        "translation_delta_p95_mm": float(np.percentile(translations, 95)),
    }


def fit_stereo(ids, da, db, image_size, Ka, Da, Kb, Db):
    objects = [OBJ.copy() for _ in ids]
    images_a = [da[i].reshape(-1, 1, 2).astype(np.float32) for i in ids]
    images_b = [db[i].reshape(-1, 1, 2).astype(np.float32) for i in ids]
    return cv2.stereoCalibrate(
        objects,
        images_a,
        images_b,
        Ka.copy(),
        Da.copy(),
        Kb.copy(),
        Db.copy(),
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )


def stereo_pair(da, db, ia, ib, Ka, Da, Kb, Db, valid_a=None, valid_b=None, limit: int = 120) -> dict:
    shared_detected = sorted(set(da) & set(db))
    allowed_a = set(da) if valid_a is None else set(valid_a)
    allowed_b = set(db) if valid_b is None else set(valid_b)
    shared_qualified = sorted(set(shared_detected) & allowed_a & allowed_b)
    selected = select_stratified(shared_qualified, limit)
    if len(selected) < 8:
        return {
            "n_shared_raw": len(shared_detected),
            "n_shared_after_intrinsic_filter": len(shared_qualified),
            "n_selected": len(selected),
            "status": "insufficient_overlap",
        }
    validation = selected[::5]
    training = [frame for frame in selected if frame not in set(validation)]
    if len(validation) < 2 or len(training) < 5:
        return {
            "n_shared_raw": len(shared_detected),
            "n_shared_after_intrinsic_filter": len(shared_qualified),
            "n_selected": len(selected),
            "status": "insufficient_validation_split",
        }
    train_result = fit_stereo(training, da, db, (ia["width"], ia["height"]), Ka, Da, Kb, Db)
    train_rms, _, _, _, _, train_R, train_T, _, _ = train_result
    training_diagnostics = transform_deltas(training, da, db, Ka, Da, Kb, Db, train_R, train_T)
    validation_diagnostics = transform_deltas(validation, da, db, Ka, Da, Kb, Db, train_R, train_T)
    final_result = fit_stereo(selected, da, db, (ia["width"], ia["height"]), Ka, Da, Kb, Db)
    final_rms, _, _, _, _, R, T, _, _ = final_result
    return {
        "status": "fit",
        "n_shared_raw": len(shared_detected),
        "n_shared_after_intrinsic_filter": len(shared_qualified),
        "n_selected": len(selected),
        "n_train": len(training),
        "n_validation": len(validation),
        "selected_frame_range": [selected[0], selected[-1]],
        "train_stereo_rms_px": float(train_rms),
        "stereo_rms_px": float(final_rms),
        "rotation_deg": rotation_angle(R),
        "rotation_matrix": R.tolist(),
        "translation_mm": T.reshape(-1).tolist(),
        "baseline_mm": float(np.linalg.norm(T)),
        "training_diagnostics": training_diagnostics,
        "validation_diagnostics": validation_diagnostics,
    }


def write_summaries(output: Path, calibrations: dict, pairs: dict) -> None:
    with (output / "intrinsics_summary.csv").open("w", newline="") as target:
        columns = [
            "camera", "status", "quality", "model", "n_detected", "n_retained", "n_rejected_views",
            "opencv_rms_px", "test_median_px", "test_p95_px", "fold_fx_cv_pct",
            "observed_max_normalized_radius", "full_frame_max_normalized_radius", "fx_px", "fy_px",
            "cx_px", "cy_px", "distortion",
        ]
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for key, row in calibrations.items():
            flattened = dict(row)
            flattened["camera"] = key
            flattened["fold_fx_cv_pct"] = row.get("cross_validation", {}).get("fold_fx_cv_pct")
            flattened["observed_max_normalized_radius"] = row.get("coverage", {}).get("observed_max_normalized_radius")
            flattened["full_frame_max_normalized_radius"] = row.get("physical_validity", {}).get("full_frame_max_normalized_radius")
            writer.writerow({column: flattened.get(column) for column in columns})
    with (output / "extrinsics_summary.csv").open("w", newline="") as target:
        columns = [
            "pair", "status", "n_shared_raw", "n_shared_after_intrinsic_filter", "n_selected", "n_train",
            "n_validation", "stereo_rms_px",
            "baseline_mm", "validation_rotation_median_deg", "validation_rotation_p95_deg",
            "validation_translation_median_mm", "validation_translation_p95_mm",
        ]
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for key, row in pairs.items():
            validation = row.get("validation_diagnostics", {})
            flattened = dict(row)
            flattened.update({
                "pair": key,
                "validation_rotation_median_deg": validation.get("rotation_delta_median_deg"),
                "validation_rotation_p95_deg": validation.get("rotation_delta_p95_deg"),
                "validation_translation_median_mm": validation.get("translation_delta_median_mm"),
                "validation_translation_p95_mm": validation.get("translation_delta_p95_mm"),
            })
            writer.writerow({column: flattened.get(column) for column in columns})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for camera in range(15, 23):
        jobs.append(("optitrack", str(camera), args.assets_dir / f"opti_dynamic_checkerboard_cam{camera}.mp4", 40))
    for camera in range(1, 5):
        jobs.append(("edgertronic", str(camera), args.assets_dir / f"edge_dynamic_checkerboard_cam{camera}.mp4", 6))
    jobs.append(("iphone", "iphone", args.assets_dir / "iphone_dynamic_checkerboard.MOV", 6))

    provenance = {path.name: file_provenance(path) for _, _, path, _ in jobs}
    detections, infos = {}, {}
    for system, camera, path, stride in jobs:
        key = f"{system}_{camera}"
        print(f"detect {key}: {path.name}", flush=True)
        info, detected = detect_video(path, stride)
        infos[key], detections[key] = info, detected
        print(f"  {len(detected)}/{info['sampled']} sampled frames", flush=True)

    calibrations, matrices, valid_frames = {}, {}, {}
    if args.reuse_intrinsics:
        previous_path = args.output_dir / "calibration_results.json"
        previous = json.loads(previous_path.read_text())
        if previous.get("schema_version") != 2 or previous.get("input_provenance") != provenance:
            raise RuntimeError("Cannot reuse intrinsics: schema or source hashes do not match")
        calibrations = previous["calibrations"]
        print(f"reuse intrinsics from {previous_path}", flush=True)
        for key, result in calibrations.items():
            if result.get("status") != "fit":
                continue
            K = np.array([
                [result["fx_px"], 0.0, result["cx_px"]],
                [0.0, result["fy_px"], result["cy_px"]],
                [0.0, 0.0, 1.0],
            ])
            matrices[key] = (K, np.asarray(result["distortion"], dtype=float))
            valid_frames[key] = set(result["retained_frames"])
    else:
        for key, detected in detections.items():
            print(f"calibrate {key}", flush=True)
            if len(detected) < 12:
                calibrations[key] = {"status": "insufficient_detections", "n_detected": len(detected)}
                continue
            result, K, D = calibrate_camera(detected, infos[key], args.max_intrinsic_views)
            calibrations[key], matrices[key] = result, (K, D)
            valid_frames[key] = set(result["retained_frames"])
            print(
                f"  {result['model']} {result['quality']}: held-out median={result['test_median_px']:.3f}px "
                f"p95={result['test_p95_px']:.3f}px, rejected={result['n_rejected_views']}",
                flush=True,
            )

    pairs = {}
    systems = (("optitrack", [str(x) for x in range(15, 23)]), ("edgertronic", [str(x) for x in range(1, 5)]))
    for system, cameras in systems:
        for index, camera_a in enumerate(cameras):
            for camera_b in cameras[index + 1:]:
                key_a, key_b = f"{system}_{camera_a}", f"{system}_{camera_b}"
                label = f"{key_a}__{key_b}"
                if key_a not in matrices or key_b not in matrices:
                    pairs[label] = {"status": "missing_intrinsics"}
                    continue
                print(f"stereo {label}", flush=True)
                pairs[label] = stereo_pair(
                    detections[key_a], detections[key_b], infos[key_a], infos[key_b],
                    *matrices[key_a], *matrices[key_b], valid_frames[key_a], valid_frames[key_b],
                    args.max_stereo_views,
                )
                print(f"  {pairs[label]['status']}, selected={pairs[label].get('n_selected', 0)}", flush=True)

    payload = {
        "schema_version": 2,
        "generated_by": Path(__file__).name,
        "software_versions": software_versions(),
        "input_provenance": provenance,
        "configuration": {
            "pattern_inner_corners": PATTERN,
            "square_mm": SQUARE_MM,
            "max_intrinsic_views": args.max_intrinsic_views,
            "max_stereo_views": args.max_stereo_views,
            "pixel_aspect_ratio": 1.0,
            "cross_validation_folds": 3,
            "intrinsic_models": [{"name": row["name"], "complexity_rank": row["complexity"]} for row in MODEL_SPECS],
        },
        "video_info": infos,
        "calibrations": calibrations,
        "stereo_pairs": pairs,
    }
    (args.output_dir / "calibration_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_summaries(args.output_dir, calibrations, pairs)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
