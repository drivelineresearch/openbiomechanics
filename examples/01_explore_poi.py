"""Explore the pitching point-of-interest (POI) metrics.

Loads the per-pitch POI table and the session/athlete metadata through the
``obp`` loader, prints a few summary statistics, and saves a scatter plot of
pitch velocity against peak elbow varus moment (a headline arm-stress metric).

Run:
    python3 examples/01_explore_poi.py
"""

import pathlib
import sys

# Make the repo root importable so ``import obp`` works from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a screen
import matplotlib.pyplot as plt

from obp import load_metadata, load_poi


def main() -> int:
    poi = load_poi("pitching")
    meta = load_metadata("pitching")

    print(f"POI table:      {poi.shape[0]} pitches x {poi.shape[1]} metrics")
    print(f"Metadata table: {meta.shape[0]} pitches x {meta.shape[1]} columns")

    # Every pitch is keyed by session_pitch. validate= catches accidental
    # duplicate keys instead of silently multiplying rows.
    analysis = poi.merge(
        meta[["session_pitch", "age_yrs", "playing_level"]],
        on="session_pitch",
        how="inner",
        validate="one_to_one",
    )
    if len(analysis) != len(poi):
        raise ValueError("POI and metadata session_pitch keys do not fully match")

    print(f"\nPitch types thrown:\n{analysis['pitch_type'].value_counts().to_string()}")
    print(
        f"\nPitch speed (mph): mean={analysis['pitch_speed_mph'].mean():.1f} "
        f"min={analysis['pitch_speed_mph'].min():.1f} "
        f"max={analysis['pitch_speed_mph'].max():.1f}"
    )
    print(
        "Elbow varus moment (Nm): "
        f"mean={analysis['elbow_varus_moment'].mean():.1f} "
        f"max={analysis['elbow_varus_moment'].max():.1f}"
    )
    print(f"Athlete age (years): median={analysis['age_yrs'].median():.1f}")

    # This is a descriptive trial-level correlation, not a causal estimate.
    corr = analysis["pitch_speed_mph"].corr(analysis["elbow_varus_moment"])
    print(f"\nPearson r (pitch_speed_mph vs elbow_varus_moment): {corr:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        analysis["pitch_speed_mph"],
        analysis["elbow_varus_moment"],
        s=18,
        alpha=0.5,
        edgecolor="none",
    )
    ax.set_xlabel("Pitch speed (mph)")
    ax.set_ylabel("Peak elbow varus moment (Nm)")
    ax.set_title(f"Arm stress vs. velocity  (n={len(analysis)}, r={corr:.2f})")
    fig.tight_layout()

    out = (
        pathlib.Path(__file__).resolve().parent
        / "figures"
        / "01_speed_vs_varus_moment.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
