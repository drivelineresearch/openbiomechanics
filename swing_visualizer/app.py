"""Run the local OpenBiomechanics skeleton overlay application."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from obp import c3d_dir

APP_DIR = pathlib.Path(__file__).resolve().parent
SEGMENTS = (
    ("LFHD", "RFHD"),
    ("LFHD", "LBHD"),
    ("RFHD", "RBHD"),
    ("LBHD", "RBHD"),
    ("C7", "CLAV"),
    ("CLAV", "STRN"),
    ("C7", "T10"),
    ("STRN", "T10"),
    ("LSHO", "RSHO"),
    ("LSHO", "LELB"),
    ("LELB", "LWRA"),
    ("LWRA", "LWRB"),
    ("LWRA", "LFIN"),
    ("RSHO", "RELB"),
    ("RELB", "RWRA"),
    ("RWRA", "RWRB"),
    ("RWRA", "RFIN"),
    ("LASI", "RASI"),
    ("LPSI", "RPSI"),
    ("LASI", "LPSI"),
    ("RASI", "RPSI"),
    ("LSHO", "LASI"),
    ("RSHO", "RASI"),
    ("LASI", "LKNE"),
    ("LKNE", "LANK"),
    ("LANK", "LHEE"),
    ("LANK", "LTOE"),
    ("RASI", "RKNE"),
    ("RKNE", "RANK"),
    ("RANK", "RHEE"),
    ("RANK", "RTOE"),
)


def _finite(value: Any) -> float | None:
    number = float(value)
    return round(number, 3) if math.isfinite(number) else None


def _ground_height(
    frames: list[list[list[float | None]]], index: dict[str, int]
) -> float:
    markers = [
        index[label] for label in ("LHEE", "RHEE", "LTOE", "RTOE") if label in index
    ]
    heights = sorted(
        point[2]
        for frame in frames
        for marker in markers
        if (point := frame[marker])[2] is not None
    )
    return round(heights[max(0, int(len(heights) * 0.03) - 1)], 3) if heights else 0.0


def _estimated_ball(
    path: pathlib.Path,
    poses: list[dict[str, list[float]] | None],
    contact: int | None,
    rate: float,
) -> list[list[float] | None]:
    """Estimate flight from the reconstructed barrel tip at estimated contact."""
    ball: list[list[float] | None] = [None] * len(poses)
    if contact is None or poses[contact] is None:
        return ball

    # Ball direction is not measured. Use a conservative center-field flight
    # instead of projecting the barrel's lateral hand path as spray direction.
    launch_angle = math.radians(8.0)
    direction = np.asarray([math.cos(launch_angle), 0.0, math.sin(launch_angle)])
    match = re.search(r"_(\d+)\.c3d$", path.name, re.IGNORECASE)
    exit_speed = (int(match.group(1)) / 10 * 0.44704) if match else 40.0
    origin = np.asarray(poses[contact]["barrel"])
    for frame in range(contact, len(poses)):
        elapsed = (frame - contact) / rate
        position = origin + direction * exit_speed * elapsed
        position[2] -= 4.905 * elapsed**2
        ball[frame] = [round(float(value), 3) for value in position]
    return ball


def _bat_grip(
    frames: list[list[list[float | None]]], bat: list[int], barrel: int | None
) -> int | None:
    """Choose one stable handle marker: farthest from the known barrel marker."""
    if barrel is None:
        return None
    distances: dict[int, list[float]] = {
        marker: [] for marker in bat if marker != barrel
    }
    for frame in frames:
        barrel_point = frame[barrel]
        if any(value is None for value in barrel_point):
            continue
        for marker, marker_distances in distances.items():
            point = frame[marker]
            if any(value is None for value in point):
                continue
            marker_distances.append(
                sum((point[axis] - barrel_point[axis]) ** 2 for axis in range(3))
            )
    averages = {
        marker: sum(values) / len(values)
        for marker, values in distances.items()
        if values
    }
    return max(averages, key=averages.get) if averages else None


def _bat_poses(
    raw_points: Any, labels: list[str], bat: list[int]
) -> list[dict[str, list[float]] | None]:
    """Fit a stable bat axis and extend it to regulation-scale length."""
    hands = [
        labels.index(label)
        for label in ("LFIN", "RFIN", "LWRA", "RWRA")
        if label in labels
    ]
    poses: list[dict[str, list[float]] | None] = []
    for frame in range(raw_points.shape[2]):
        bat_points = raw_points[:3, bat, frame].T
        valid_bat = np.isfinite(bat_points).all(axis=1)
        if raw_points.shape[0] > 3:
            valid_bat &= raw_points[3, bat, frame] >= 0
        bat_points = bat_points[valid_bat]
        if len(bat_points) < 2:
            poses.append(None)
            continue
        center = bat_points.mean(axis=0)
        _, _, axes = np.linalg.svd(bat_points - center, full_matrices=False)
        axis = axes[0]
        projections = (bat_points - center) @ axis
        low = center + axis * projections.min()
        high = center + axis * projections.max()
        hand_points = raw_points[:3, hands, frame].T
        valid_hands = np.isfinite(hand_points).all(axis=1)
        if raw_points.shape[0] > 3:
            valid_hands &= raw_points[3, hands, frame] >= 0
        hand_points = hand_points[valid_hands]
        hand = hand_points.mean(axis=0) if len(hand_points) else center
        if np.linalg.norm(low - hand) <= np.linalg.norm(high - hand):
            grip, direction = low, axis
        else:
            grip, direction = high, -axis
        # OBP hitting metadata reports bats around 32–34 inches. The rigid-body
        # markers span only part of the implement, so render a full 34-inch bat.
        barrel = grip + direction * 0.864
        poses.append(
            {
                "grip": [round(float(value), 4) for value in grip],
                "barrel": [round(float(value), 4) for value in barrel],
            }
        )
    return poses


def _motion_analysis(
    path: pathlib.Path,
    poses: list[dict[str, list[float]] | None],
    contact: int | None,
    rate: float,
) -> dict[str, Any]:
    """Derive review metrics from the reconstructed barrel trajectory."""
    speeds: list[float | None] = [None] * len(poses)
    for frame in range(1, len(poses) - 1):
        before, after = poses[frame - 1], poses[frame + 1]
        if before is None or after is None:
            continue
        delta = np.asarray(after["barrel"]) - np.asarray(before["barrel"])
        speeds[frame] = float(np.linalg.norm(delta) * rate / 2)

    valid = [(speed, frame) for frame, speed in enumerate(speeds) if speed is not None]
    peak_speed, peak_frame = max(valid, default=(0.0, 0))
    if contact is None:
        contact = peak_frame if valid else None
    contact_speed = speeds[contact] if contact is not None else None

    swing_start = 0
    if contact is not None and peak_speed:
        threshold = peak_speed * 0.15
        prior = [
            frame
            for frame in range(contact)
            if speeds[frame] is not None and speeds[frame] < threshold
        ]
        swing_start = min(contact, (prior[-1] + 1) if prior else 0)

    attack_angle = None
    bat_direction = None
    if contact is not None and 0 < contact < len(poses) - 1:
        before, after = poses[contact - 1], poses[contact + 1]
        if before is not None and after is not None:
            velocity = np.asarray(after["barrel"]) - np.asarray(before["barrel"])
            horizontal = float(np.linalg.norm(velocity[:2]))
            attack_angle = math.degrees(math.atan2(float(velocity[2]), horizontal))
            bat_direction = math.degrees(
                math.atan2(float(velocity[1]), float(velocity[0]))
            )

    neighborhood = (
        poses[max(0, contact - 5) : min(len(poses), contact + 6)]
        if contact is not None
        else []
    )
    coverage = (
        sum(pose is not None for pose in neighborhood) / len(neighborhood)
        if neighborhood
        else 0.0
    )
    confidence = "High" if coverage >= 0.9 else "Medium" if coverage >= 0.65 else "Low"
    exit_match = re.search(r"_(\d+)\.c3d$", path.name, re.IGNORECASE)
    return {
        "events": {
            "swingStart": swing_start,
            "peakSpeed": peak_frame,
            "contact": contact,
            "followThrough": (
                min(len(poses) - 1, contact + round(rate * 0.15))
                if contact is not None
                else None
            ),
        },
        "metrics": {
            "peakBarrelSpeedMph": round(peak_speed * 2.23694, 1),
            "contactBarrelSpeedMph": (
                round(contact_speed * 2.23694, 1) if contact_speed is not None else None
            ),
            "timeToContactMs": (
                round((contact - swing_start) / rate * 1000)
                if contact is not None
                else None
            ),
            "attackAngleDeg": round(attack_angle, 1)
            if attack_angle is not None
            else None,
            "batDirectionDeg": (
                round(bat_direction, 1) if bat_direction is not None else None
            ),
            "recordedExitVelocityMph": (
                round(int(exit_match.group(1)) / 10, 1) if exit_match else None
            ),
        },
        "confidence": confidence,
    }


def load_motion(path: pathlib.Path) -> dict[str, Any]:
    """Read one C3D into the compact structure consumed by the browser."""
    try:
        import ezc3d
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ezc3d is missing; install the repository requirements"
        ) from error

    c3d = ezc3d.c3d(str(path))
    point = c3d["parameters"]["POINT"]
    labels = [str(label).strip() for label in point["LABELS"]["value"]]
    raw_points = c3d["data"]["points"]
    points = raw_points[:3]
    if points.shape[1] != len(labels) or points.shape[2] < 2:
        raise ValueError("C3D marker labels and frames are inconsistent")
    rate = float(point["RATE"]["value"][0])
    if rate <= 0:
        raise ValueError("C3D marker rate must be positive")

    index = {label: position for position, label in enumerate(labels)}
    has_residuals = raw_points.shape[0] > 3
    frames = [
        [
            (
                [_finite(points[axis, marker, frame]) for axis in range(3)]
                if not has_residuals or float(raw_points[3, marker, frame]) >= 0
                else [None, None, None]
            )
            for marker in range(len(labels))
        ]
        for frame in range(points.shape[2])
    ]
    barrel = index.get("Marker5")
    bat = [i for i, label in enumerate(labels) if label.startswith("Marker")]
    grip = _bat_grip(frames, bat, barrel)
    bat_poses = _bat_poses(raw_points, labels, bat)
    analysis = _motion_analysis(path, bat_poses, None, rate)
    contact = analysis["events"]["contact"]
    ball = _estimated_ball(path, bat_poses, contact, rate)
    return {
        "name": path.name,
        "rate": rate,
        "labels": labels,
        "frames": frames,
        "segments": [
            [index[a], index[b]] for a, b in SEGMENTS if a in index and b in index
        ],
        "bat": bat,
        "batPoses": bat_poses,
        "barrel": barrel,
        "grip": grip,
        "ground": _ground_height(frames, index),
        "contact": contact,
        "ball": ball,
        "analysis": analysis,
        "anchor": [index[label] for label in ("LASI", "RASI") if label in index],
    }


def list_trials(root: pathlib.Path) -> list[dict[str, str]]:
    """List non-static C3Ds relative to the configured data root."""
    return [
        {"path": path.relative_to(root).as_posix(), "name": path.stem}
        for path in sorted(root.glob("**/*.c3d"))
        if "model" not in path.name.lower()
    ]


def resolve_trial(root: pathlib.Path, requested: str) -> pathlib.Path:
    """Resolve a client-supplied relative trial path without allowing traversal."""
    candidate = (root / requested).resolve()
    root = root.resolve()
    if candidate.suffix.lower() != ".c3d" or not candidate.is_relative_to(root):
        raise ValueError("Invalid trial path")
    if not candidate.is_file():
        raise FileNotFoundError("Trial not found")
    return candidate


class ApplicationServer(ThreadingHTTPServer):
    data_root: pathlib.Path


class Handler(BaseHTTPRequestHandler):
    """Serve the application and its local C3D API."""

    server: ApplicationServer

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(
            status, "application/json; charset=utf-8", json.dumps(payload).encode()
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/trials":
                self._json(
                    HTTPStatus.OK, {"trials": list_trials(self.server.data_root)}
                )
                return
            if parsed.path == "/api/motion":
                requested = parse_qs(parsed.query).get("path", [""])[0]
                motion = load_motion(resolve_trial(self.server.data_root, requested))
                self._json(HTTPStatus.OK, motion)
                return
            if parsed.path in ("/", "/index.html"):
                self._send(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    (APP_DIR / "index.html").read_bytes(),
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except (FileNotFoundError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (IndexError, KeyError, RuntimeError, TypeError) as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local skeleton overlay application."
    )
    parser.add_argument("--data-root", type=pathlib.Path, default=c3d_dir("hitting"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the browser automatically"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = ApplicationServer((args.host, args.port), Handler)
    server.data_root = args.data_root.resolve()
    url = f"http://{args.host}:{args.port}"
    print(f"Skeleton overlay app: {url}")
    print(f"C3D data root:        {server.data_root}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
