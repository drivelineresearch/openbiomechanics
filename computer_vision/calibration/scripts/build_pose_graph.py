#!/usr/bin/env python3
"""Build a provisional OptiTrack rig in camera-19 coordinates."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
data = json.loads((RESULTS / "calibration_results.json").read_text())
reference = "19"


def angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


accepted = {}
graph = {str(camera): [] for camera in range(15, 23)}
for name, row in data["stereo_pairs"].items():
    if not name.startswith("optitrack") or row.get("status") != "fit":
        continue
    strict = (
        row["stereo_rms_px"] < 0.5
        and row["per_frame_rotation_delta_median_deg"] < 1.0
        and row["per_frame_translation_delta_median_mm"] < 30.0
    )
    if not strict:
        continue
    left, right = name.split("__")
    a, b = left.rsplit("_", 1)[1], right.rsplit("_", 1)[1]
    R = np.asarray(row["rotation_matrix"], dtype=float)
    t = np.asarray(row["translation_mm"], dtype=float).reshape(3, 1)
    accepted[f"{a}_to_{b}"] = {
        "source_pair": name,
        "shared_sampled_frames": row["n_shared"],
        "stereo_rms_px": row["stereo_rms_px"],
        "rotation_matrix": R.tolist(),
        "translation_mm": t.reshape(-1).tolist(),
        "median_rotation_drift_deg": row["per_frame_rotation_delta_median_deg"],
        "median_translation_drift_mm": row["per_frame_translation_delta_median_mm"],
    }
    graph[a].append((b, row["stereo_rms_px"], R, t, f"{a}_to_{b}"))
    graph[b].append((a, row["stereo_rms_px"], R.T, -R.T @ t, f"{a}_to_{b}"))

# Minimum-error shortest-path tree. This is intentionally not presented as a
# substitute for final robust pose-graph optimization / bundle adjustment.
distance = {camera: float("inf") for camera in graph}
distance[reference] = 0.0
poses = {reference: (np.eye(3), np.zeros((3, 1)))}
parent = {reference: None}
queue = [(0.0, reference)]
while queue:
    cost, camera = heapq.heappop(queue)
    if cost != distance[camera]:
        continue
    R_camera, t_camera = poses[camera]
    for neighbor, weight, R_edge, t_edge, edge_name in graph[camera]:
        candidate = cost + weight
        if candidate < distance[neighbor]:
            distance[neighbor] = candidate
            poses[neighbor] = (R_edge @ R_camera, R_edge @ t_camera + t_edge)
            parent[neighbor] = {"camera": camera, "accepted_edge": edge_name}
            heapq.heappush(queue, (candidate, neighbor))

if set(poses) != set(graph):
    raise RuntimeError(f"Accepted graph is disconnected: reached {sorted(poses)}")

camera_poses = {}
for camera in sorted(poses, key=int):
    R, t = poses[camera]
    center = -R.T @ t
    camera_poses[camera] = {
        "reference_to_camera_rotation_matrix": R.tolist(),
        "reference_to_camera_translation_mm": t.reshape(-1).tolist(),
        "camera_center_in_reference_mm": center.reshape(-1).tolist(),
        "shortest_path_cost_px": distance[camera],
        "tree_parent": parent[camera],
    }

closures = []
for edge_name, row in accepted.items():
    a, b = edge_name.split("_to_")
    Ra, ta = poses[a]
    Rb, tb = poses[b]
    predicted_R = Rb @ Ra.T
    predicted_t = tb - predicted_R @ ta
    observed_R = np.asarray(row["rotation_matrix"])
    observed_t = np.asarray(row["translation_mm"]).reshape(3, 1)
    closures.append({
        "edge": edge_name,
        "rotation_closure_error_deg": angle_deg(predicted_R @ observed_R.T),
        "translation_closure_error_mm": float(np.linalg.norm(predicted_t - observed_t)),
        "is_tree_edge": any(v and v["accepted_edge"] == edge_name for v in parent.values()),
    })

non_tree = [row for row in closures if not row["is_tree_edge"]]
payload = {
    "status": "provisional_not_bundle_adjusted",
    "reference_camera": "optitrack_19",
    "coordinate_convention": (
        "A 3-D point expressed in camera-19 coordinates is mapped to camera i by "
        "X_i = R_i19 @ X_19 + t_i19. Camera centers are expressed in camera-19 coordinates."
    ),
    "units": "millimetres",
    "acceptance_thresholds": {
        "stereo_rms_px_lt": 0.5,
        "median_rotation_drift_deg_lt": 1.0,
        "median_translation_drift_mm_lt": 30.0,
    },
    "accepted_pair_count": len(accepted),
    "accepted_pairs": accepted,
    "camera_poses": camera_poses,
    "loop_closure_checks": closures,
    "non_tree_loop_closure_summary": {
        "count": len(non_tree),
        "max_rotation_error_deg": max((x["rotation_closure_error_deg"] for x in non_tree), default=0.0),
        "max_translation_error_mm": max((x["translation_closure_error_mm"] for x in non_tree), default=0.0),
    },
    "warning": (
        "Do not treat this shortest-path pose graph as the final released lab calibration. "
        "Run orientation-aware all-frame bundle adjustment, validate loop closure, and align "
        "the result to annotated cube/CS-200 lab coordinates first."
    ),
}
(RESULTS / "optitrack_rig_provisional.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload["non_tree_loop_closure_summary"], indent=2))
