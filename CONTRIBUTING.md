# Contributing to the OpenBiomechanics Project

Thank you for helping make OBP clearer, safer, and easier to reproduce. Useful contributions include documentation corrections, small runnable analyses, loader improvements, data-quality reports, and reproducible computer-vision work.

## Before you start

- Search existing issues and pull requests to avoid duplicating work.
- Keep one focused change per branch.
- Do not include athlete identities, private Driveline data, credentials, or machine-local paths.
- Never commit raw C3Ds, full-signal exports, videos, model weights, or installers. Large public artifacts belong in a GitHub Release; see [`RELEASING.md`](RELEASING.md).
- Read the [license split](#licensing-contributions) before contributing code, data, or biomechanics documentation.

## Local setup

The public surface is tested on Python 3.10 and 3.12. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\activate`. The release downloader requires Git Bash or WSL plus `gh`, `unzip`, and a SHA-256 utility. Authenticate GitHub CLI with `gh auth login` (or provide `GH_TOKEN` in automation) before exercising downloads.

Install [ShellCheck](https://www.shellcheck.net/) separately for the downloader gate (`apt install shellcheck`, `brew install shellcheck`, or your platform's package manager); it is a system executable rather than a Python dependency.

The requirement files declare minimum compatible versions, not a complete lock. The exact calibration environment is the exception: `computer_vision/calibration/requirements.txt` records the versions used to generate its committed artifacts.

## Repository map

| Path | Contributor-facing purpose |
| --- | --- |
| `baseball_pitching/`, `baseball_hitting/` | Public CSVs, biomechanics conventions, and internal provenance code |
| `high_performance/` | Public assessment table and protocol documentation |
| `obp/` | Repository-local pandas and path helpers |
| `examples/` | Small public workflows that must run from a fresh clone |
| `computer_vision/` | Educational examples plus separately validated calibration work |
| `scripts/` | Release download/integrity tooling and dictionary generation |
| `tests/` | Public API, schema, docs-link, notebook, and manifest checks |

The notebooks under `baseball_*/code/py/` and `high_performance/code/` require private Driveline database access. They document release provenance and intentionally fail without the necessary environment; do not disguise that boundary with fallback data or broad exception handling.

## Verification

Run the repository-wide checks from the root:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/build_data_dictionary.py --check
ruff check obp examples scripts tests computer_vision/hello_world computer_vision/utils.py
ruff format --check obp examples scripts tests computer_vision/hello_world computer_vision/utils.py
python3 -m compileall -q obp examples scripts computer_vision
bash -n scripts/download_data.sh
shellcheck scripts/download_data.sh

python3 examples/01_explore_poi.py
python3 examples/02_read_c3d.py
python3 examples/03_join_fullsig.py
python3 examples/04_hp_assessment.py
```

Examples 02 and 03 print actionable download instructions and exit successfully when large release data are absent. Do not download the roughly 1.1 GB dataset merely to run the default CI gate.

After editing a notebook, validate its JSON:

```bash
for file in $(git ls-files '*.ipynb'); do
  python3 -m json.tool "$file" >/dev/null
done
```

Changes under `computer_vision/calibration/` also require:

```bash
python3 -m pip install -r computer_vision/calibration/requirements.txt
python3 -m unittest discover -s computer_vision/calibration/tests -v
python3 computer_vision/calibration/scripts/build_pose_graph.py
git diff --exit-code -- computer_vision/calibration/results/optitrack_rig_provisional.json
```

## Data and documentation changes

- Treat the CSV headers as authoritative. Pitching and hitting use different metadata units and field names.
- If a README's POI or metadata dictionary changes, regenerate the machine-readable dictionaries:

  ```bash
  python3 scripts/build_data_dictionary.py
  git diff -- baseball_pitching/data/data_dictionary.csv \
    baseball_hitting/data/data_dictionary.csv \
    high_performance/data/data_dictionary.csv \
    data_dictionary.json
  ```

- Preserve unknown scientific definitions as explicitly unknown. Do not infer coordinate conventions, metric formulas, or athlete crosswalks that are not supported by the release documentation.
- Account for repeated trials per athlete and the 360 Hz marker / 1,080 Hz force-plate sampling-rate difference in examples and analyses.
- Use repository-relative, clickable links and commands that work from a fresh clone.

## Pull requests

1. Branch from current `main`.
2. Make the smallest coherent change and include tests or reproducible evidence.
3. Run the relevant verification commands above.
4. Complete the pull-request template, including public behavior, data/docs impact, and license scope.
5. Address review feedback without force-pushing over another contributor's work.

Maintainers may ask that large dataset changes be published as a new immutable release tag rather than committed to Git.

## Licensing contributions

- Code contributions are licensed under the [MIT License](LICENSE-CODE.md).
- Data and biomechanics-documentation contributions use the terms in [`LICENSE-DATA.md`](LICENSE-DATA.md), including its additional professional-sports-organization / financial-firm exclusion.

By submitting a contribution, you agree that it can be distributed under the license applicable to the material you changed.

## Collaboration and commercial licensing

For research collaboration, commercial data licensing, or questions about the larger IRB-approved dataset, email **sportsscience@drivelinebaseball.com** and CC **gretchen@drivelinebaseball.com**.

Report security or sensitive-content issues privately according to [`SECURITY.md`](SECURITY.md).
