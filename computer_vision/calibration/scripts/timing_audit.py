#!/usr/bin/env python3
"""Frame timestamp audit for files used by the calibration study."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("/tmp/obp-calibration-assets")
OUT = Path(__file__).resolve().parents[1] / "results" / "timing_results.json"


def pts(path: Path) -> np.ndarray:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)]
    text = subprocess.check_output(cmd, text=True)
    return np.array([float(x.strip().strip(",")) for x in text.splitlines() if x.strip().strip(",")])


payload = {}
for name in ["iphone_dynamic_checkerboard.MOV", "iphone_color_palette.MOV", "iphone_grayscale_palette.MOV"]:
    t = pts(ROOT / name)
    d = np.diff(t)
    payload[name] = {
        "frames_with_pts": len(t),
        "duration_from_pts_s": float(t[-1] - t[0]),
        "median_interval_ms": float(np.median(d) * 1000),
        "p05_interval_ms": float(np.percentile(d, 5) * 1000),
        "p95_interval_ms": float(np.percentile(d, 95) * 1000),
        "unique_intervals_ms": sorted(set(np.round(d * 1000, 3).tolist()))[:20],
    }
OUT.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
