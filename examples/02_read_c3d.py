"""Read a raw C3D motion-capture file with ezc3d.

Resolves the pitching C3D directory through ``obp.c3d_dir``, opens one of the
sample files with ezc3d, prints the marker count and names, and saves a 3D plot
of a single marker's trajectory across the capture.

Requires ezc3d (``pip install ezc3d``). Sample files ship at
``baseball_pitching/data/c3d/000822/``; the full set comes from
``scripts/download_data.sh``.

Run:
    python3 examples/02_read_c3d.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ezc3d

from obp import c3d_dir

# c3d_dir gives the directory; athlete C3Ds live in per-athlete subfolders.
# Static model files (``*_model.c3d``) hold no motion, so skip them.
c3d_files = sorted(p for p in c3d_dir("pitching").glob("*/*.c3d")
                   if "model" not in p.name)
assert c3d_files, "No sample C3D files found. Run scripts/download_data.sh."

path = c3d_files[0]
print(f"Opening {path.relative_to(c3d_dir('pitching').parents[2])}")

c3d = ezc3d.c3d(str(path))
labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
points = c3d["data"]["points"]  # shape: (4, n_markers, n_frames) -> x,y,z,residual
rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
n_frames = points.shape[2]

print(f"Markers: {len(labels)}   Frames: {n_frames}   Rate: {rate:.0f} Hz")
print(f"Marker names: {', '.join(labels)}")

# Plot the throwing-hand-adjacent RWRA marker if present, else the first marker.
marker = "RWRA" if "RWRA" in labels else labels[0]
idx = labels.index(marker)
xs, ys, zs = points[0, idx], points[1, idx], points[2, idx]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs, ys, zs, lw=1.2)
ax.scatter(xs[0], ys[0], zs[0], c="green", s=40, label="start")
ax.scatter(xs[-1], ys[-1], zs[-1], c="red", s=40, label="end")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title(f"Marker '{marker}' trajectory  ({n_frames} frames @ {rate:.0f} Hz)")
ax.legend()
fig.tight_layout()

out = pathlib.Path(__file__).resolve().parent / "figures" / "02_marker_trajectory.png"
fig.savefig(out, dpi=120)
print(f"Wrote {out}")
