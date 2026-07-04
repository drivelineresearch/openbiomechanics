"""Explore the high-performance (force-plate assessment) table.

Loads ``high_performance/data/hp_obp.csv`` through ``obp.load_hp`` and plots the
distribution of countermovement-jump (CMJ) jump height across athletes -- a
standard lower-body power screen that pairs with the pitching/hitting data via
athlete UID.

Run:
    python3 examples/04_hp_assessment.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obp import load_hp

hp = load_hp()
print(f"HP table: {hp.shape[0]} assessments x {hp.shape[1]} columns")

# Force-plate metrics are suffixed by test: _cmj (countermovement jump),
# _sj (squat jump), _imtp (isometric mid-thigh pull), etc.
metric = "jump_height_(imp-mom)_[cm]_mean_cmj"
vals = hp[metric].dropna()

print(f"\n{metric}")
print(f"  n={len(vals)}  mean={vals.mean():.1f} cm  sd={vals.std():.1f}  "
      f"min={vals.min():.1f}  max={vals.max():.1f}")
print(f"\nAssessments by playing level:\n{hp['playing_level'].value_counts().to_string()}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(vals, bins=20, edgecolor="white")
ax.axvline(vals.mean(), color="red", ls="--", lw=1.5, label=f"mean {vals.mean():.1f} cm")
ax.set_xlabel("CMJ jump height (cm)")
ax.set_ylabel("Athletes")
ax.set_title(f"Countermovement-jump height distribution  (n={len(vals)})")
ax.legend()
fig.tight_layout()

out = pathlib.Path(__file__).resolve().parent / "figures" / "04_cmj_jump_height.png"
fig.savefig(out, dpi=120)
print(f"\nWrote {out}")
