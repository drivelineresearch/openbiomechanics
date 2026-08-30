"""Explore the high-performance (force-plate assessment) table.

Loads ``high_performance/data/hp_obp.csv`` through ``obp.load_hp`` and plots the
assessment-level distribution of countermovement-jump (CMJ) jump height. Some
athletes have repeated assessments, so this is not an athlete-level estimate.

Run:
    python3 examples/04_hp_assessment.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obp import load_hp


def main() -> int:
    hp = load_hp()
    print(f"HP table: {hp.shape[0]} assessments x {hp.shape[1]} columns")
    print(f"Unique athletes: {hp['athlete_uid'].nunique()}")

    # Force-plate metrics are suffixed by test: _cmj (countermovement jump),
    # _sj (squat jump), _imtp (isometric mid-thigh pull), etc.
    metric = "jump_height_(imp-mom)_[cm]_mean_cmj"
    vals = hp[metric].dropna()

    print(f"\n{metric}")
    print(
        f"  assessments={len(vals)}  mean={vals.mean():.1f} cm  "
        f"sd={vals.std():.1f}  min={vals.min():.1f}  max={vals.max():.1f}"
    )
    print(
        "\nAssessments by playing level:\n"
        f"{hp['playing_level'].value_counts().to_string()}"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=20, edgecolor="white")
    ax.axvline(
        vals.mean(),
        color="red",
        ls="--",
        lw=1.5,
        label=f"mean {vals.mean():.1f} cm",
    )
    ax.set_xlabel("CMJ jump height (cm)")
    ax.set_ylabel("Assessments")
    ax.set_title(f"Assessment-level CMJ jump height  (n={len(vals)})")
    ax.legend()
    fig.tight_layout()

    out = pathlib.Path(__file__).resolve().parent / "figures" / "04_cmj_jump_height.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
