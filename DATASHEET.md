# Datasheet for the OpenBiomechanics Project (OBP)

This datasheet follows the structure proposed in *Datasheets for Datasets* (Gebru et al., 2018). It documents the OpenBiomechanics Project dataset released by Driveline Baseball Research & Development. All composition statistics below were computed directly from the in-repository point-of-interest (POI) and metadata CSVs; the exact pandas snippet used is included at the end of this document.

The dataset comprises three data modules — **Baseball Pitching**, **Baseball Hitting**, and **High Performance** — plus supporting computer-vision and tutorial material.

---

## Motivation

**For what purpose was the dataset created?**
The OpenBiomechanics Project was created by [Driveline Baseball Research & Development](https://drivelinebaseball.com/mission-and-purpose/) to provide raw (cleaned C3D marker files) and processed (full-signal time series + point-of-interest summaries) sports-biomechanics data to the general public. The goal is to lower the barrier to biomechanics research in baseball by openly distributing marker-based motion-capture, force-plate, and physical-assessment data that are otherwise difficult and expensive to collect. Project homepage: [openbiomechanics.org](https://openbiomechanics.org).

**Who created the dataset and who funded it?**
The dataset was created and funded by Driveline Baseball's Research & Development group.

---

## Composition

**What do the instances represent?**
Instances are individual baseball movement trials (pitches or swings) and athlete-level physical-assessment records. Each pitching/hitting trial is linked to marker-derived kinematics/kinetics and, where applicable, ground reaction forces. High-performance records are per-assessment force-plate and range-of-motion summaries.

### Baseball Pitching (fastball trials)

Computed from `baseball_pitching/data/poi/poi_metrics.csv` (411 rows) and `baseball_pitching/data/metadata.csv` (411 rows):

- **Pitch trials:** 411
- **Unique athletes** (`user`): 100
- **Unique sessions:** 100
- **Pitch type:** all 411 trials are fastballs (`FF`)
- **Handedness:** 328 right-handed, 83 left-handed
- **Pitch speed (mph):** min 69.5 / median 85.3 / max 94.4
- **Athlete age at collection (years):** min 17.7 / median 20.7 / max 27.6
- **Playing-level breakdown** (`playing_level`):

  | Playing level | Trials |
  | --- | --- |
  | College | 314 |
  | Independent | 42 |
  | High school | 32 |
  | MiLB | 23 |

### Baseball Hitting (swing trials)

Computed from `baseball_hitting/data/poi/poi_metrics.csv` (677 rows) and `baseball_hitting/data/metadata.csv` (677 rows):

- **Swing trials:** 677
- **Unique athletes** (`user`): 98
- **Unique sessions:** 98
- **Hitting side:** 496 right-handed, 181 left-handed
- **Exit velocity (mph):** min 48.8 / median 91.0 / max 107.0
- **Athlete age at collection (years):** min 17 / median 20 / max 31
- **Paired HitTrax ball-in-play records:** 604 rows (`baseball_hitting/data/poi/hittrax.csv`)
- **Playing-level breakdown** (`highest_playing_level`):

  | Playing level | Trials |
  | --- | --- |
  | College | 517 |
  | High school | 83 |
  | MiLB | 47 |
  | Independent | 30 |

### High Performance (force-plate + physical assessment)

Computed from `high_performance/data/hp_obp.csv`:

- **Assessment records:** 1,934
- **Unique athletes** (`athlete_uid`): 1,162
- **Playing-level breakdown** (`playing_level`): College 904, High School 816, Pro 214
- **Force-plate / assessment tests present** (by column suffix, with count of records containing data for each):

  | Test | Abbreviation | Records with data |
  | --- | --- | --- |
  | Countermovement Jump | CMJ | 1,913 |
  | Squat Jump | SJ | 1,827 |
  | Repeated Hop Test | HT | 1,803 |
  | Isometric Mid-Thigh Pull | IMTP | 1,518 |
  | Plyo Pushup | PP | 438 |

  The module also includes table range-of-motion (thoracic-spine mobility) and supine shoulder internal/external-rotation strength (dynamometer) measurements. The Plyo Pushup is not present for the entire dataset (438 of 1,934 records).

**Does the dataset contain all possible instances or a sample?**
The dataset is a sample of athletes assessed at Driveline Baseball facilities. Most participants are collegiate-level. The pitching module is restricted to fastball trials.

**What data does each instance consist of?**
Three representations are provided:

1. **Raw:** cleaned C3D marker files (per-athlete folders, with static model files). Pitching uses a marker set with three embedded force plates under the mound turf; hitting uses a marker set plus a 10-marker bat rigid body and four force plates under the batter's box. See each module README for marker-set images, global coordinate systems, and force-plate layouts.
2. **Full signal:** filtered joint angles, joint velocities, joint forces/moments, energy-flow, ground reaction forces, and landmark (joint-center) time series. Kinematics/kinetics are computed by the right-hand rule with coach/player-intuition sign adjustments, then filtered with a 4th-order Butterworth low-pass filter (20 Hz for marker-derived signals, 40 Hz for ground reaction forces). Marker-derived data sampled at 360 Hz; force-plate data at 1,080 Hz.
3. **Point of interest (POI):** one-row-per-trial summary metrics (e.g., peak shoulder internal-rotation velocity, elbow varus moment, stride length, ground-reaction-force peaks for pitching; bat speed, attack angle, hip-shoulder separation for hitting). Full variable dictionaries are in the module READMEs.

**Is there a label or target?**
There is no single target. Common outcome variables are `pitch_speed_mph` (pitching) and `exit_velo_mph_x` (hitting). The High Performance module additionally links pitching/hitting performance (e.g., `pitch_speed_mph`, `bat_speed_mph`) to force-plate metrics.

**Are relationships between instances made explicit?**
Yes. Trials link to athletes and sessions via `user` / `session`, and to full-signal/POI data via `session_pitch` (pitching) or `session_swing` (hitting). Full-signal tables join on `session_pitch`/`session_swing` + `time`. Event times (foot contact, foot plant, maximum external rotation, ball release, maximum internal rotation for pitching; front foot contact, front foot plant, contact for hitting) are joined into each table for convenience.

**Are there recommended data splits?**
No canonical train/test split is defined. Users should account for repeated measures — multiple trials per athlete and per session — when constructing splits to avoid athlete leakage.

**Are there errors, sources of noise, or redundancies?**
The data are marker-based motion capture and are subject to normal soft-tissue-artifact and modeling limitations. Some legacy markers (LIC/RIC iliac-crest markers) are no longer used and may be absent in some C3D files; RKNEE2/LKNEE2 are labeled RMKNE/LMKNE. Some force/moment sign conventions at the shoulder and hip are not reported (noted as "convention not reported" in the module tables). The Plyo Pushup test is missing for most High Performance records. Caution is advised when joining force-plate (1,080 Hz) and marker-derived (360 Hz) data because of the differing sample rates.

**Is the dataset self-contained?**
The POI and metadata CSVs are in-repo. The large raw C3D files and full-signal archives are **not** stored in git; they are distributed via GitHub Releases (see Distribution).

**Does the dataset contain confidential or sensitive data?**
Athlete identifiers are anonymized (numeric `user`/`session` IDs, UUID `athlete_uid` in the High Performance module, and anonymized C3D filenames). The data are human-subjects biomechanics measurements collected under IRB approval (see Collection Process). No names or direct identifiers are included.

---

## Collection Process

**How was the data acquired?**
Marker-based optical motion capture was collected at Driveline Baseball facilities. Pitching trials were thrown on an instrumented mound with three force plates embedded under the turf (average turf thickness ~0.5 in). Hitting trials were collected in an instrumented batter's box with four force plates under the turf; hitters started with one or both feet on the back plate and landed with the lead leg on the front plate.

**Hitting stimulus (pitching-machine setup):** all swings were collected while hitting off a pitching machine set at ~65 mph from ~40 ft from home plate.

High-performance data were collected using force-plate protocols — Countermovement Jump, Squat Jump, Isometric Mid-Thigh Pull, Repeated Hop Test, and Plyo Pushup — paired with table range-of-motion and dynamometer shoulder-strength testing. Protocol videos are linked in the High Performance README.

**Who was involved and over what timeframe?**
Data were collected by Driveline Baseball R&D staff. High Performance assessment dates in the data span at least 2021 onward (see the `test_date` column).

**Were the individuals notified and did they consent? Was there ethical review?**
Yes. Western IRB (WCG IRB) provided ethical approval for all data-collection procedures under **Western IRB # WB-DLR-115**.

---

## Preprocessing / Cleaning / Labeling

**Was preprocessing done?**
Yes. C3D files were cleaned and organized into per-athlete folders with static model files. Full-signal joint angles, velocities, forces, and moments were computed by the right-hand rule and then adjusted for coach/player intuition (righty/lefty symmetry, sign negations, ±90° offsets). Marker-derived signals were filtered with a 4th-order Butterworth low-pass filter at 20 Hz; ground reaction forces at 40 Hz. Forces and moments are internal and, by default, expressed in the proximal segment's coordinate system, with dual-resolved ("double-dipped") kinetics provided at the shoulders and hips. POI metrics were extracted at defined biomechanical events.

**Was the raw data saved?**
Yes. Cleaned C3D files are provided so users can run their own pipeline start-to-finish and link results to the provided full-signal data via the metadata CSV.

---

## Uses

**What has the dataset been used for / could it be used for?**
Baseball pitching and hitting biomechanics research; force-plate/performance analytics; computer-vision and markerless-pose work (see the Computer Vision module); and methodological studies (e.g., processing the provided C3D through a custom pipeline and comparing against the released full-signal data).

**What should users be aware of?**
Users should respect the repeated-measures structure (multiple trials per athlete/session), the differing sample rates of force-plate vs. marker data, the fastball-only scope of the pitching module, the predominantly collegiate population, and the missing Plyo Pushup data in the High Performance module.

**Are there tasks for which the dataset should not be used?**
Any commercial use is prohibited by the data license (see Distribution). The license additionally forbids use by any employee, contractor, associate, or significant shareholder of a professional sports organization or financial-analysis firm without a separate written commercial license.

---

## Distribution

**How is the dataset distributed?**
The dataset is distributed publicly through the OpenBiomechanics GitHub repository and [GitHub Releases](https://github.com/drivelineresearch/openbiomechanics/releases):

- **POI + metadata CSVs** remain in-repo under each module's `data/` folder, so summary analyses work without downloading the large archives.
- **Raw C3D files and full-signal tables are NOT stored in git.** They are distributed as release assets:
  - `dataset-v1` — processed full-signal archives (pitching + hitting) plus raw `pitching_c3d.zip` and `hitting_c3d.zip`.
  - `cv-media-v1` — `cv_media.zip`, computer-vision demo media.
  - `tools-mokka-0.6.2` — Mokka installers for viewing C3D files.
- Full-signal tables ship as `.zip` archives (e.g., `joint_angles.zip`) and must be unzipped before use.
- Run `scripts/download_data.sh` from the repository root to fetch and unpack the release assets into the correct locations.

**License.**
OBP is dual-licensed:

- **Code** — MIT License (`LICENSE-CODE.md`).
- **Data + biomechanics documentation** — Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (**CC BY-NC-SA 4.0**; `LICENSE-DATA.md`).

The data license is non-commercial and carries one additional exclusion beyond standard CC BY-NC-SA terms: any employee or contractor employed by, associated with, or a significant shareholder of a professional sports organization or financial-analysis firm is forbidden from using OBP data for any purpose without a separate written commercial (paid) license. See `LICENSE-DATA.md` for the full terms.

---

## Maintenance

**Who maintains the dataset?**
Driveline Baseball Research & Development maintains the dataset via the public GitHub repository.

**How is it versioned and updated?**
Updates are recorded in the root `README.md` and released through GitHub Releases (current data tag: `dataset-v1`). Notable updates include the addition of the High Performance module (2024-07-30) and the Computer Vision module (2023-11-02).

**How can users contribute or report issues?**
Contributions and issues are handled through the GitHub repository (see `CONTRIBUTING.md`). For more information, visit [openbiomechanics.org](https://openbiomechanics.org).

---

## Reproducing the Composition Statistics

All Composition numbers above were computed with the following pandas snippet, run against the in-repo CSVs:

```python
import pandas as pd, re

# Pitching
pm = pd.read_csv("baseball_pitching/data/poi/poi_metrics.csv")
pmeta = pd.read_csv("baseball_pitching/data/metadata.csv")
len(pm)                                  # 411 trials
pmeta['user'].nunique()                  # 100 athletes
pmeta['playing_level'].value_counts()    # college 314, independent 42, high_school 32, milb 23
pm['pitch_speed_mph'].agg(['min','median','max'])   # 69.5 / 85.3 / 94.4

# Hitting
hm = pd.read_csv("baseball_hitting/data/poi/poi_metrics.csv")
hmeta = pd.read_csv("baseball_hitting/data/metadata.csv")
len(hm)                                  # 677 trials
hmeta['user'].nunique()                  # 98 athletes
hmeta['highest_playing_level'].value_counts()  # college 517, high_school 83, milb 47, independent 30
hm['exit_velo_mph_x'].agg(['min','median','max'])   # 48.8 / 91.0 / 107.0
len(pd.read_csv("baseball_hitting/data/poi/hittrax.csv"))  # 604 HitTrax rows

# High Performance
hp = pd.read_csv("high_performance/data/hp_obp.csv")
len(hp)                                  # 1934 records
hp['athlete_uid'].nunique()              # 1162 athletes
hp['playing_level'].value_counts()       # College 904, High School 816, Pro 214
for suf in ['cmj','sj','ht','imtp','pp']:
    cols = [c for c in hp.columns if c.endswith('_'+suf)]
    print(suf, hp[cols].notna().any(axis=1).sum())  # cmj 1913, sj 1827, ht 1803, imtp 1518, pp 438
```
