#!/usr/bin/env python3
"""Interleaved intrinsic-fit stability check for the rolling-shutter iPhone feed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import analyze_calibration as calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=calibration.DEFAULT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=calibration.DEFAULT_OUT)
    return parser.parse_args()


def fit(ids, detections, info, flags):
    rms, K, D, _, _ = calibration.fit_camera(ids, detections, info, flags)
    return {
        "rms_px": float(rms),
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "physical_validity": calibration.radial_validity(K, D, info["width"], info["height"]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.assets_dir / "iphone_dynamic_checkerboard.MOV"
    calibration_data = json.loads((args.output_dir / "calibration_results.json").read_text())
    model_name = calibration_data["calibrations"]["iphone_iphone"]["model"]
    spec = next(row for row in calibration.MODEL_SPECS if row["name"] == model_name)
    info, detections = calibration.detect_video(video, 6)
    ids = sorted(detections)
    split_a = [frame for index, frame in enumerate(ids) if index % 2 == 0]
    split_b = [frame for index, frame in enumerate(ids) if index % 2 == 1]
    fit_a = fit(split_a, detections, info, spec["flags"])
    fit_b = fit(split_b, detections, info, spec["flags"])
    Ka, Kb = np.asarray(fit_a["K"]), np.asarray(fit_b["K"])
    Da, Db = np.asarray(fit_a["D"]), np.asarray(fit_b["D"])
    payload = {
        "schema_version": 2,
        "model": model_name,
        "software_versions": calibration.software_versions(),
        "input_provenance": calibration.file_provenance(video),
        "split_method": "alternating detected frames after stride-6 sampling",
        "split_a": fit_a,
        "split_b": fit_b,
        "relative_fx_difference_pct": float(abs(Ka[0, 0] - Kb[0, 0]) / np.mean([Ka[0, 0], Kb[0, 0]]) * 100),
        "relative_fy_difference_pct": float(abs(Ka[1, 1] - Kb[1, 1]) / np.mean([Ka[1, 1], Kb[1, 1]]) * 100),
        "principal_point_difference_px": float(np.linalg.norm(Ka[:2, 2] - Kb[:2, 2])),
        "distortion_l2_difference": float(np.linalg.norm(Da - Db)),
        "detections_per_split": [len(split_a), len(split_b)],
        "warning": "Interleaved stability does not estimate or correct rolling-shutter readout time.",
    }
    target = args.output_dir / "iphone_split_stability.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
