"""Join two full-signal time-series tables on session_pitch + time.

The full-signal tables (``joint_angles``, ``joint_velos``, ``forces_moments``,
``energy_flow``, ``force_plate``, ``landmarks``) are large and are NOT stored in
git. Fetch them with ``obp.download()`` (wraps scripts/download_data.sh); they
land as ``.zip`` archives in ``data/full_sig/`` and must be unzipped first.

Per the pitching README, marker-derived tables share the 360 Hz clock and join
cleanly on ``session_pitch`` + ``time``. (Force-plate data are sampled at
1,080 Hz, so joining those to marker data can drop rows -- avoided here by
joining two marker-derived tables.)

Run:
    python3 examples/03_join_fullsig.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from obp import full_sig_dir

fs = full_sig_dir("pitching")
angles_csv = fs / "joint_angles.csv"
velos_csv = fs / "joint_velos.csv"

if not (angles_csv.exists() and velos_csv.exists()):
    print("Full-signal tables not found locally.")
    print(f"Expected: {angles_csv}")
    print(f"          {velos_csv}")
    print()
    print("Fetch them first, then unzip the archives:")
    print("    scripts/download_data.sh          # or:  python3 -c 'import obp; obp.download()'")
    print(f"    cd {fs} && unzip -o 'joint_angles.zip' && unzip -o 'joint_velos.zip'")
    sys.exit(0)

# Both tables are keyed by (session_pitch, time) at the same sample rate.
angles = pd.read_csv(angles_csv)
velos = pd.read_csv(velos_csv)
print(f"joint_angles: {angles.shape}")
print(f"joint_velos:  {velos.shape}")

# joint_velos repeats the shared event-time columns; keep only the join keys
# plus its measurement columns so the merge doesn't duplicate them.
shared = set(angles.columns) & set(velos.columns)
velo_cols = ["session_pitch", "time"] + [c for c in velos.columns if c not in shared]

merged = angles.merge(velos[velo_cols], on=["session_pitch", "time"], how="inner")
print(f"merged:       {merged.shape}")
assert len(merged) == len(angles), "row count changed -- unexpected time misalignment"

print(f"\nPitches represented: {merged['session_pitch'].nunique()}")
print(f"Columns after join:  {merged.shape[1]}")
