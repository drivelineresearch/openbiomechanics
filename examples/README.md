# Examples

Runnable, commented Python scripts showing real use of the OpenBiomechanics
dataset through the [`obp`](../obp/README.md) loader package. Each script adds
the repo root to `sys.path` and imports `obp` -- no install step.

Figures are written to `examples/figures/` (Agg backend, `savefig` only).

| Script | What it shows | Needs downloaded data? |
| --- | --- | --- |
| [`01_explore_poi.py`](01_explore_poi.py) | Loads pitching POI + metadata, prints summary stats, scatters pitch speed vs. peak elbow varus moment. | No -- CSVs ship in git. |
| [`02_read_c3d.py`](02_read_c3d.py) | Opens a raw C3D with `ezc3d`, prints marker count/names, plots one marker's 3D trajectory. | No -- sample C3Ds ship in `baseball_pitching/data/c3d/000822/`. Requires `ezc3d`. |
| [`03_join_fullsig.py`](03_join_fullsig.py) | Joins two full-signal time-series tables on `session_pitch` + `time`. | **Yes** -- full-signal tables are not in git. |
| [`04_hp_assessment.py`](04_hp_assessment.py) | Loads the high-performance force-plate table, plots the CMJ jump-height distribution. | No -- `hp_obp.csv` ships in git. |

## Running

```bash
# From the repository root:
python3 examples/01_explore_poi.py
python3 examples/02_read_c3d.py      # needs ezc3d
python3 examples/03_join_fullsig.py  # prints download instructions if data is absent
python3 examples/04_hp_assessment.py
```

## Getting the large data (for script 03)

The raw C3D set and the full-signal tables live on GitHub Releases, not in git.
Fetch them from the repo root:

```bash
scripts/download_data.sh
# or:  python3 -c "import obp; obp.download()"
```

Full-signal tables arrive as `.zip` archives in `<discipline>/data/full_sig/`
and must be unzipped before use. Script `03` prints the exact `unzip` commands
if the CSVs are missing.

Per the pitching README, marker-derived tables (`joint_angles`, `joint_velos`,
`forces_moments`, `energy_flow`, `landmarks`) share a 360 Hz clock and join
cleanly on `session_pitch` + `time`. Force-plate data are 1,080 Hz, so joining
those to marker data can drop rows -- join with care.
