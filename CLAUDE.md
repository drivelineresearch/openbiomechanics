# OpenBiomechanics Project — Agent Guide

Public research dataset from [Driveline Baseball R&D](https://drivelinebaseball.com/mission-and-purpose/):
motion-capture biomechanics for baseball **pitching** and **hitting**, plus
force-plate **high-performance** assessments and **computer-vision** tutorials.
This file onboards coding agents. Keep it under 300 lines; put detail in the
files it points to.

## WHAT — repository map

| Path | What lives here |
| --- | --- |
| `baseball_pitching/` | Pitching data (`data/`) + processing code (`code/`) + `README.md` (marker set, coordinate system, sign conventions, POI + metadata dictionaries) |
| `baseball_hitting/` | Same structure as pitching, for swings |
| `high_performance/` | Force-plate + assessment data (`data/hp_obp.csv`) + scrape notebook + `README.md` |
| `computer_vision/` | OpenCV / YOLO / calibration tutorials (`TUTORIAL.md`) |
| `additional_resources/` | Cited references + `tutorials/` (ezc3d, ISBS) |
| `scripts/download_data.sh` | Fetches the large data artifacts from GitHub Releases |

Each `baseball_*/data/` holds: `metadata.csv`, `poi/` (in-repo, small) — and,
after download, `c3d/` (raw trials) and `full_sig/*.zip` (processed signals).

## WHY — purpose & constraints

- Open dataset for biomechanics research. **Dual-licensed:** code under MIT
  ([`LICENSE-CODE.md`](LICENSE-CODE.md)), data + biomechanics docs under
  CC BY-NC-SA 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)).
- The data is a public sample of a larger IRB-approved dataset (Western IRB
  # WB-DLR-115); the full set is commercially licensed.

## HOW — working in this repo

- **Get the data:** `scripts/download_data.sh` (needs the `gh` CLI). Add
  `--with-media` for CV demo videos, `--with-mokka` for the C3D viewer.
- **Dependencies:** each area has its own `requirements.txt`
  (`computer_vision/`, `<module>/code/`); root `requirements.txt` is the
  minimal data-analysis stack.
- **Join keys:** pitching tables join on `session_pitch`, hitting on
  `session_swing`; full-signal tables also need `time`. Force-plate (1080 Hz)
  and marker data (360 Hz) sample at different rates — join with care.

## Rules

- **Never commit large binaries (raw C3D, `full_sig` zips, video, installers).**
  `.gitignore` blocks them; new large artifacts belong in a GitHub Release, not
  git — history bloat here is permanent. → [Getting the Data](README.md)
- **Keep the license split intact.** The professional-sports-organization /
  financial-firm exclusion in `LICENSE-DATA.md` is load-bearing and must stay
  verbatim; do not soften or drop it.
- **The `*/code/py/*.ipynb` notebooks are internal provenance scripts.** They
  need Driveline DB access + `CLUSTER_*` env vars and will not run externally —
  they document how the released data was produced, not a user pipeline. Don't
  "fix" them to run standalone, and don't add try/except (they fail loudly by
  design).
- **Docs that cite column names must match the real CSV headers.** Pitching
  metadata uses metric units (`session_height_m`, `session_mass_kg`); hitting
  uses imperial (`session_height_in`, `session_mass_lbs`). They differ — verify
  before documenting.

## Verification

- After editing any notebook, confirm it is still valid JSON:
  `python -m json.tool <file> >/dev/null`.
- After editing a README's POI or metadata dictionary, cross-check the entries
  against the actual `data/poi/*.csv` / `data/metadata.csv` headers — the docs
  have drifted from the data before.

<!-- AGENT-MANAGED SECTION — Claude and other agents may add learnings below. -->
<!-- Lifecycle: (1) write the full why/history into the relevant README or a  -->
<!-- notes file FIRST; (2) add a one-line pointer here; (3) once stable,       -->
<!-- graduate it into the matching section above and delete it here. Keep this -->
<!-- an inbox, not an archive.                                                 -->

## Discovered Patterns

_(none yet)_
