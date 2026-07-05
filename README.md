# OpenBiomechanics Project (OBP) Documentation

The OpenBiomechanics Project is an initiative started by [Driveline Baseball Research & Development](https://drivelinebaseball.com/mission-and-purpose/) to provide raw (in the form of cleaned C3D files) and processed (full signal + point of interest) sports biomechanics data to the general public. For more information, read the documentation below and visit the [project homepage](https://openbiomechanics.org).

![IMG_5797.JPG](imgs/IMG_5797.jpg)

## What's New (2026-07-04 repository overhaul)

> [!WARNING]
> **Breaking — git history was rewritten.** The repository history was purged to
> remove ~2 GB of large binaries, shrinking `.git` from **2.2 GB to ~25 MB** (a
> fresh clone is now ~28 MB). **If you have an existing clone or fork, delete it
> and re-clone** — old commit hashes no longer exist and a `git pull` will not
> reconcile. Nothing in the current file tree changed; only history was rewritten.

> [!IMPORTANT]
> **The large data no longer lives in git.** Raw C3D files, processed
> full-signal archives, computer-vision demo media, and the Mokka installer are
> now distributed via [GitHub Releases](https://github.com/drivelineresearch/openbiomechanics/releases).
> Run [`scripts/download_data.sh`](scripts/download_data.sh) to fetch them. POI
> and metadata CSVs remain in-repo.

**Major changes**

- **History purge** — large binaries removed from all git history (see the warning above).
- **Data → GitHub Releases** — `dataset-v1` (raw C3D + full-signal), `cv-media-v1` (demo media), `tools-mokka-0.6.2` (C3D viewer). Fetch with `scripts/download_data.sh`.
- **Dual license split** — code is now MIT ([`LICENSE-CODE.md`](LICENSE-CODE.md)); data + documentation stay CC BY-NC-SA 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)), including the professional-sports-organization / financial-firm exclusion.

**Minor changes / additions**

- **New tooling** — [`obp/`](obp/README.md) pandas loader package and runnable [`examples/`](examples/README.md).
- **New metadata** — machine-readable [data dictionaries](#data-dictionaries--datasheet) (`<module>/data/data_dictionary.csv` + `data_dictionary.json`), a [`DATASHEET.md`](DATASHEET.md), and [`CITATION.cff`](CITATION.cff).
- **Reproducibility** — pinned `requirements.txt` per module; agent guide in [`CLAUDE.md`](CLAUDE.md).
- **Docs & code fixes** — corrected POI/metadata dictionaries (fixed a broken entry, added undocumented columns, fixed the `rejc` landmark), removed copyrighted textbook PDFs in favor of [cited references](additional_resources/README.md), de-duplicated and parameterized the computer-vision scripts, and removed committed cruft (`.DS_Store`, hardcoded paths/PII in notebooks).

## Contents

- [What's New](#whats-new-2026-07-04-repository-overhaul)
- [Modules](#modules)
- [Dataset at a glance](#dataset-at-a-glance)
- [Getting the Data](#getting-the-data)
- [Quickstart](#quickstart)
- [Data dictionaries & datasheet](#data-dictionaries--datasheet)
- [Citing](#citing)
- [License](#license)
- [IRB information](#irb-information)
- [Updates](#updates)

## Modules

| Module | Description | Link |
| --- | --- | --- |
| Baseball Pitching | Full-signal + point-of-interest kinematics/kinetics for fastball trials. | [`baseball_pitching/`](baseball_pitching/README.md) |
| Baseball Hitting | Full-signal + point-of-interest biomechanics for baseball swings. | [`baseball_hitting/`](baseball_hitting/README.md) |
| High Performance | Force-plate and physical-assessment data paired with the mocap athletes. | [`high_performance/`](high_performance/README.md) |
| Computer Vision | OBP-CV markerless/2D examples, calibration, and pose tutorials. | [`computer_vision/`](computer_vision/TUTORIAL.md) |
| Additional Resources | Tutorials, references, and supporting material. | [`additional_resources/`](additional_resources/README.md) |

## Dataset at a glance

Approximate figures — see each module's README for exact counts and definitions.

- **Pitching:** ~411 fastball trials across ~100 athletes.
- **Hitting:** swing trials across ~99 athletes.
- **High Performance:** force-plate and physical-assessment data for the participating athletes.
- **Population:** most participants are collegiate-level.

## Getting the Data

To keep the repository lightweight, the large archives are distributed via [GitHub Releases](https://github.com/drivelineresearch/openbiomechanics/releases) rather than tracked in git:

- **`dataset-v1`** — processed full-signal archives (pitching + hitting) plus the raw `pitching_c3d.zip` and `hitting_c3d.zip`.
- **`cv-media-v1`** — `cv_media.zip`, the computer-vision demo media; restore by unzipping at the repo root.
- **`tools-mokka-0.6.2`** — Mokka installers for viewing C3D files.

Run [`scripts/download_data.sh`](scripts/download_data.sh) to fetch and unpack the release assets into the correct locations.

Point-of-interest (POI) and metadata CSVs remain in-repo under each module's `data/` folder, so summary analyses work without downloading the full archives.

## Quickstart

A small helper package, [`obp/`](obp/README.md), resolves data paths and loads the
in-repo CSVs with pandas:

```python
import sys; sys.path.insert(0, "path/to/openbiomechanics")
import obp

poi = obp.load_poi("pitching")     # 411 fastball trials x 81 POI metrics
meta = obp.load_metadata("hitting")
hp = obp.load_hp()
obp.download(media=True)            # fetch the release archives via scripts/download_data.sh
```

Runnable, commented scripts live in [`examples/`](examples/README.md):

- `01_explore_poi.py` — load POI + metadata, plot pitch speed vs. elbow varus moment
- `02_read_c3d.py` — open a C3D with `ezc3d`, plot a marker trajectory
- `03_join_fullsig.py` — join full-signal tables on `session_pitch` + `time`
- `04_hp_assessment.py` — plot a countermovement-jump distribution

## Data dictionaries & datasheet

- **Data dictionaries** — machine-readable column references generated from the
  actual CSV headers: `<module>/data/data_dictionary.csv` and the aggregated
  [`data_dictionary.json`](data_dictionary.json). Regenerate with
  [`scripts/build_data_dictionary.py`](scripts/build_data_dictionary.py).
- **Datasheet** — [`DATASHEET.md`](DATASHEET.md) documents motivation, composition
  (with real counts), collection, and distribution, following the
  *Datasheets for Datasets* framework.

## Citing

If you use OBP, please cite it — see [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button reads this file).

## License

OBP is dual-licensed:

- **Code** — MIT License. See [`LICENSE-CODE.md`](LICENSE-CODE.md).
- **Data + biomechanics documentation** — Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). See [`LICENSE-DATA.md`](LICENSE-DATA.md) and the OBP site under "Usage Terms." (`license.txt` is retained as a pointer.)

The data license is non-commercial, and it carries one additional specific exclusion beyond the standard CC BY-NC-SA terms:

> While the license is clear that this data cannot be used for commercial purposes (which includes but is not limited to for-profit organizations, corporations, and sole proprietorships with the intent to profit now or in the future), there is also one additional specific exclusion where this data cannot be used in any form without a specific written commercial (paid) license: ***Any employee or contractor employed by, associated with, or a significant shareholder of a professional sports organization or financial analysis firm is forbidden to use The OpenBiomechanics Project data for any use whatsoever.***

See [`LICENSE-DATA.md`](LICENSE-DATA.md) for the full terms.

## IRB information

[Western IRB](https://www.wcgirb.com/) provided ethical approval for all data collection procedures (Western IRB # WB-DLR-115).

## Updates

### Update 2024-07-30

[High Performance](https://github.com/drivelineresearch/openbiomechanics/tree/main/high_performance) module added! Find it in the `high_performance` folder.

### Update 2023-11-02

OBP Computer Vision (OBP-CV) added! Examples can be found in the `computer_vision` folder and the README and data can be found at the following Google Sheet link:

[OBP-CV README, Shot List, and More Information](https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q/edit?usp=sharing)
