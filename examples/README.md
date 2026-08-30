# Examples

Runnable, commented Python scripts showing real use of the OpenBiomechanics
dataset through the [`obp`](../obp/README.md) loader. Each script makes the
repository root importable, so no package-install step is required.

Figures are written to an automatically created, Git-ignored
`examples/figures/` directory (Agg backend, `savefig` only).

| Script | What it shows | Needs downloaded data? |
| --- | --- | --- |
| [`01_explore_poi.py`](01_explore_poi.py) | Validates a one-to-one pitching POI/metadata join, prints descriptive statistics, and plots pitch speed vs. peak elbow varus moment. | No — CSVs ship in Git. |
| [`02_read_c3d.py`](02_read_c3d.py) | Opens a downloaded C3D with `ezc3d`, prints marker count/names, and plots one marker's 3D trajectory. | **Yes** — run the pitching download first. |
| [`03_join_fullsig.py`](03_join_fullsig.py) | Validates a one-to-one join between two marker-derived full-signal tables on `session_pitch` + `time`. | **Yes** — full-signal tables are not in Git. |
| [`04_hp_assessment.py`](04_hp_assessment.py) | Loads the high-performance table and plots an assessment-level CMJ jump-height distribution with repeated-measures context. | No — `hp_obp.csv` ships in Git. |

## Running

```bash
# From the repository root:
python3 examples/01_explore_poi.py
python3 examples/02_read_c3d.py      # needs the pitching C3D download
python3 examples/03_join_fullsig.py  # prints download instructions if data is absent
python3 examples/04_hp_assessment.py
```

## Getting the large data (for scripts 02 and 03)

Raw C3Ds and full-signal tables live on GitHub Releases, not in Git. Fetch only
the pitching assets needed by examples 02 and 03 from the repository root:

```bash
scripts/download_data.sh --discipline pitching
# or: python3 -c 'import obp; obp.download("pitching")'
```

The downloader verifies checksums and extracts C3Ds. Full-signal tables remain
as `.zip` archives in `<discipline>/data/full_sig/` and must be unzipped before
use. Script 03 prints the exact `unzip` commands if its CSVs are missing.

Per the pitching README, marker-derived tables (`joint_angles`, `joint_velos`,
`forces_moments`, `energy_flow`, `landmarks`) share a 360 Hz clock and join
cleanly on `session_pitch` + `time`. Force-plate data are 1,080 Hz, so joining
those to marker data can drop rows -- join with care.
