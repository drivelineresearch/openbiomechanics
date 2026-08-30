"""Read a raw C3D motion-capture file with ezc3d.

Resolves the pitching C3D directory through ``obp.c3d_dir``, opens one downloaded
file with ezc3d, prints the marker count and names, and saves a 3D plot of a
single marker's trajectory across the capture.

Requires ezc3d (``python3 -m pip install ezc3d``) and the pitching C3D release:
``scripts/download_data.sh --discipline pitching``.

Run:
    python3 examples/02_read_c3d.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obp import REPO_ROOT, c3d_dir


def main(
    c3d_root: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> int:
    """Read and plot the first trial under ``c3d_root``.

    The optional paths make the example testable with a tiny synthetic C3D
    fixture while retaining the downloaded-data defaults for normal use.
    """
    root = c3d_root if c3d_root is not None else c3d_dir("pitching")
    # Athlete C3Ds live in per-athlete folders. Static model files have no
    # motion, so select the first trial capture instead.
    c3d_files = sorted(
        path for path in root.glob("*/*.c3d") if "model" not in path.name.lower()
    )
    if not c3d_files:
        print("No downloaded pitching C3D files were found.")
        print("Run: scripts/download_data.sh --discipline pitching")
        return 0

    try:
        import ezc3d
    except ModuleNotFoundError as error:
        raise SystemExit(
            "ezc3d is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from error

    path = c3d_files[0]
    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    print(f"Opening {display_path}")

    c3d = ezc3d.c3d(str(path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    # shape: (4, n_markers, n_frames) -> x, y, z, residual
    points = c3d["data"]["points"]
    rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    n_frames = points.shape[2]

    print(f"Markers: {len(labels)}   Frames: {n_frames}   Rate: {rate:.0f} Hz")
    print(f"Marker names: {', '.join(labels)}")

    # Plot the throwing-hand-adjacent RWRA marker if present, else the first.
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

    out = output_path or (
        pathlib.Path(__file__).resolve().parent / "figures" / "02_marker_trajectory.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
