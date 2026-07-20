#!/usr/bin/env python3
"""Presentation-timestamp audit for files used by the calibration study."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyze_calibration as calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=calibration.DEFAULT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=calibration.DEFAULT_OUT)
    return parser.parse_args()


def presentation_timestamps(path: Path) -> np.ndarray:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path),
    ]
    text = subprocess.check_output(command, text=True)
    timestamps = np.array([float(value.strip().strip(",")) for value in text.splitlines() if value.strip().strip(",")])
    if len(timestamps) < 2:
        raise RuntimeError(f"Fewer than two presentation timestamps in {path}")
    return timestamps


def summarize(timestamps: np.ndarray) -> dict:
    intervals = np.diff(timestamps)
    median = float(np.median(intervals))
    return {
        "frames_with_pts": len(timestamps),
        "first_pts_s": float(timestamps[0]),
        "last_pts_s": float(timestamps[-1]),
        "duration_from_pts_s": float(timestamps[-1] - timestamps[0]),
        "timestamps_strictly_increasing": bool(np.all(intervals > 0)),
        "median_interval_ms": median * 1000,
        "minimum_interval_ms": float(intervals.min() * 1000),
        "p05_interval_ms": float(np.percentile(intervals, 5) * 1000),
        "p95_interval_ms": float(np.percentile(intervals, 95) * 1000),
        "maximum_interval_ms": float(intervals.max() * 1000),
        "intervals_over_1_5x_median": int(np.sum(intervals > 1.5 * median)),
        "unique_intervals_ms": sorted(set(np.round(intervals * 1000, 3).tolist())),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ["iphone_dynamic_checkerboard.MOV", "iphone_color_palette.MOV", "iphone_grayscale_palette.MOV"]
    videos = [args.assets_dir / name for name in names]
    payload = {
        "schema_version": 2,
        "software_versions": calibration.software_versions(),
        "input_provenance": {path.name: calibration.file_provenance(path) for path in videos},
        "videos": {path.name: summarize(presentation_timestamps(path)) for path in videos},
    }
    target = args.output_dir / "timing_results.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
