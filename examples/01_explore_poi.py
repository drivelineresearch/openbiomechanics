"""Explore the pitching point-of-interest (POI) metrics.

Loads the per-pitch POI table and the session/athlete metadata through the
``obp`` loader, prints a few summary statistics, and saves a scatter plot of
pitch velocity against peak elbow varus moment (a headline arm-stress metric).

Run:
    python3 examples/01_explore_poi.py
"""

import sys
import pathlib

# Make the repo root importable so ``import obp`` works from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # headless: render to file, never to a screen
import matplotlib.pyplot as plt

from obp import load_poi, load_metadata

poi = load_poi("pitching")
meta = load_metadata("pitching")

print(f"POI table:      {poi.shape[0]} pitches x {poi.shape[1]} metrics")
print(f"Metadata table: {meta.shape[0]} pitches x {meta.shape[1]} columns")

# Every pitch is keyed by session_pitch; both tables share it.
print(f"\nPitch types thrown:\n{poi['pitch_type'].value_counts().to_string()}")
print(f"\nPitch speed (mph): mean={poi['pitch_speed_mph'].mean():.1f} "
      f"min={poi['pitch_speed_mph'].min():.1f} max={poi['pitch_speed_mph'].max():.1f}")
print(f"Elbow varus moment (Nm): mean={poi['elbow_varus_moment'].mean():.1f} "
      f"max={poi['elbow_varus_moment'].max():.1f}")

# Correlation of the two headline metrics.
corr = poi["pitch_speed_mph"].corr(poi["elbow_varus_moment"])
print(f"\nPearson r (pitch_speed_mph vs elbow_varus_moment): {corr:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(poi["pitch_speed_mph"], poi["elbow_varus_moment"],
           s=18, alpha=0.5, edgecolor="none")
ax.set_xlabel("Pitch speed (mph)")
ax.set_ylabel("Peak elbow varus moment (Nm)")
ax.set_title(f"Arm stress vs. velocity  (n={len(poi)}, r={corr:.2f})")
fig.tight_layout()

out = pathlib.Path(__file__).resolve().parent / "figures" / "01_speed_vs_varus_moment.png"
fig.savefig(out, dpi=120)
print(f"\nWrote {out}")
