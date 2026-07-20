#!/usr/bin/env python3
"""Build and validate a provisional OptiTrack rig in camera-19 coordinates."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"
REFERENCE = "19"
THRESHOLDS = {
    "stereo_rms_px_lt": 0.5,
    "validation_rotation_median_deg_lt": 1.0,
    "validation_rotation_p95_deg_lt": 1.5,
    "validation_translation_median_mm_lt": 30.0,
    "validation_translation_p95_mm_lt": 30.0,
    "validation_frames_gte": 4,
}
SERIALIZATION_DECIMALS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def rejection_reasons(row: dict) -> list[str]:
    if row.get("status") != "fit":
        return [row.get("status", "not_fit")]
    validation = row.get("validation_diagnostics", {})
    checks = (
        (row.get("stereo_rms_px", math.inf) < THRESHOLDS["stereo_rms_px_lt"], "stereo_rms"),
        (
            validation.get("rotation_delta_median_deg", math.inf)
            < THRESHOLDS["validation_rotation_median_deg_lt"],
            "validation_rotation_median",
        ),
        (
            validation.get("rotation_delta_p95_deg", math.inf)
            < THRESHOLDS["validation_rotation_p95_deg_lt"],
            "validation_rotation_p95",
        ),
        (
            validation.get("translation_delta_median_mm", math.inf)
            < THRESHOLDS["validation_translation_median_mm_lt"],
            "validation_translation_median",
        ),
        (
            validation.get("translation_delta_p95_mm", math.inf)
            < THRESHOLDS["validation_translation_p95_mm_lt"],
            "validation_translation_p95",
        ),
        (row.get("n_validation", 0) >= THRESHOLDS["validation_frames_gte"], "validation_frame_count"),
    )
    return [name for passed, name in checks if not passed]


def edge_weight(row: dict) -> float:
    validation = row["validation_diagnostics"]
    return float(
        row["stereo_rms_px"]
        + validation["rotation_delta_median_deg"]
        + validation["translation_delta_median_mm"] / 100.0
    )


def canonicalize_floats(value, decimals: int = SERIALIZATION_DECIMALS):
    """Normalize derived floats across BLAS/platform implementations."""
    if isinstance(value, dict):
        return {key: canonicalize_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_floats(item, decimals) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), decimals)
    return value


def build_payload(data: dict) -> dict:
    accepted, rejected = {}, {}
    graph = {str(camera): [] for camera in range(15, 23)}
    for name, row in data["stereo_pairs"].items():
        if not name.startswith("optitrack"):
            continue
        reasons = rejection_reasons(row)
        if reasons:
            rejected[name] = {"reasons": reasons, "metrics": row}
            continue
        left, right = name.split("__")
        camera_a, camera_b = left.rsplit("_", 1)[1], right.rsplit("_", 1)[1]
        rotation = np.asarray(row["rotation_matrix"], dtype=float)
        translation = np.asarray(row["translation_mm"], dtype=float).reshape(3, 1)
        weight = edge_weight(row)
        edge_name = f"{camera_a}_to_{camera_b}"
        accepted[edge_name] = {
            "source_pair": name,
            "raw_shared_frames": row["n_shared_raw"],
            "qualified_shared_frames": row["n_shared_after_intrinsic_filter"],
            "selected_frames": row["n_selected"],
            "validation_frames": row["n_validation"],
            "stereo_rms_px": row["stereo_rms_px"],
            "rotation_matrix": rotation.tolist(),
            "translation_mm": translation.reshape(-1).tolist(),
            "validation_diagnostics": row["validation_diagnostics"],
            "path_weight": weight,
        }
        graph[camera_a].append((camera_b, weight, rotation, translation, edge_name))
        graph[camera_b].append((camera_a, weight, rotation.T, -rotation.T @ translation, edge_name))

    distance = {camera: float("inf") for camera in graph}
    distance[REFERENCE] = 0.0
    poses = {REFERENCE: (np.eye(3), np.zeros((3, 1)))}
    parent = {REFERENCE: None}
    queue = [(0.0, REFERENCE)]
    while queue:
        cost, camera = heapq.heappop(queue)
        if cost != distance[camera]:
            continue
        camera_rotation, camera_translation = poses[camera]
        for neighbor, weight, edge_rotation, edge_translation, edge_name in graph[camera]:
            candidate = cost + weight
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                poses[neighbor] = (
                    edge_rotation @ camera_rotation,
                    edge_rotation @ camera_translation + edge_translation,
                )
                parent[neighbor] = {"camera": camera, "accepted_edge": edge_name}
                heapq.heappush(queue, (candidate, neighbor))

    if set(poses) != set(graph):
        raise RuntimeError(
            f"Accepted graph is disconnected: reached {sorted(poses)}; "
            f"missing {sorted(set(graph) - set(poses))}"
        )

    camera_poses = {}
    for camera in sorted(poses, key=int):
        rotation, translation = poses[camera]
        center = -rotation.T @ translation
        camera_poses[camera] = {
            "reference_to_camera_rotation_matrix": rotation.tolist(),
            "reference_to_camera_translation_mm": translation.reshape(-1).tolist(),
            "camera_center_in_reference_mm": center.reshape(-1).tolist(),
            "shortest_path_validation_cost": distance[camera],
            "tree_parent": parent[camera],
        }

    closures = []
    for edge_name, row in accepted.items():
        camera_a, camera_b = edge_name.split("_to_")
        rotation_a, translation_a = poses[camera_a]
        rotation_b, translation_b = poses[camera_b]
        predicted_rotation = rotation_b @ rotation_a.T
        predicted_translation = translation_b - predicted_rotation @ translation_a
        observed_rotation = np.asarray(row["rotation_matrix"])
        observed_translation = np.asarray(row["translation_mm"]).reshape(3, 1)
        closures.append({
            "edge": edge_name,
            "rotation_closure_error_deg": angle_deg(predicted_rotation @ observed_rotation.T),
            "translation_closure_error_mm": float(np.linalg.norm(predicted_translation - observed_translation)),
            "is_tree_edge": any(value and value["accepted_edge"] == edge_name for value in parent.values()),
        })

    non_tree = [row for row in closures if not row["is_tree_edge"]]
    return {
        "schema_version": 2,
        "status": "provisional_not_bundle_adjusted",
        "serialization_precision_decimal_places": SERIALIZATION_DECIMALS,
        "source_calibration_schema_version": data.get("schema_version"),
        "source_input_provenance": data.get("input_provenance"),
        "reference_camera": "optitrack_19",
        "coordinate_convention": (
            "A 3-D point expressed in camera-19 coordinates is mapped to camera i by "
            "X_i = R_i19 @ X_19 + t_i19. Camera centers are expressed in camera-19 coordinates."
        ),
        "units": "millimetres",
        "acceptance_thresholds": THRESHOLDS,
        "accepted_pair_count": len(accepted),
        "rejected_pair_count": len(rejected),
        "accepted_pairs": accepted,
        "rejected_pairs": rejected,
        "camera_poses": camera_poses,
        "loop_closure_checks": closures,
        "non_tree_loop_closure_summary": {
            "count": len(non_tree),
            "max_rotation_error_deg": max((row["rotation_closure_error_deg"] for row in non_tree), default=0.0),
            "max_translation_error_mm": max((row["translation_closure_error_mm"] for row in non_tree), default=0.0),
        },
        "warning": (
            "Do not treat this shortest-path pose graph as the final released lab calibration. "
            "Run orientation-aware all-frame bundle adjustment, validate loop closure, and align "
            "the result to annotated cube/CS-200 lab coordinates first."
        ),
    }


def main() -> None:
    args = parse_args()
    source = args.results_dir / "calibration_results.json"
    data = json.loads(source.read_text())
    payload = canonicalize_floats(build_payload(data))
    target = args.results_dir / "optitrack_rig_provisional.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["non_tree_loop_closure_summary"], indent=2))


if __name__ == "__main__":
    main()
