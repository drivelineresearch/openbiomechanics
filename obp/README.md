# obp

Small repository-local loader for the OpenBiomechanics dataset. It provides
validated path helpers and pandas loaders without hiding the underlying files.
Run Python or Jupyter from the repository root, or add that root to `sys.path`.

```python
import sys

sys.path.insert(0, "/path/to/openbiomechanics")

import obp

# CSV metrics as pandas DataFrames
poi = obp.load_poi("pitching")  # baseball_pitching/data/poi/poi_metrics.csv
meta = obp.load_metadata("hitting")  # baseball_hitting/data/metadata.csv
htx = obp.load_hittrax()  # baseball_hitting/data/poi/hittrax.csv
hp = obp.load_hp()  # high_performance/data/hp_obp.csv

# Paths only for the large signals — call ezc3d yourself
import ezc3d

path = obp.c3d_path("pitching", "000822/000822_1.c3d")
c3d = ezc3d.c3d(str(path))

obp.c3d_dir("pitching")  # baseball_pitching/data/c3d
obp.full_sig_dir("pitching")  # baseball_pitching/data/full_sig

# Fetch verified release artifacts (raw C3D + full-signal zips)
obp.download()  # pitching + hitting
obp.download("pitching")  # pitching only
obp.download(["pitching", "hitting"])  # explicit selection
obp.download("pitching", media=True)  # add CV demo media
obp.download(data=False, media=True)  # CV media only; skip the dataset
obp.download(data=False, mokka=True)  # Mokka installers only
```

`discipline` must be `"pitching"` or `"hitting"`; invalid values raise an
actionable `ValueError`. CSV loader functions pass additional keyword arguments
to `pandas.read_csv`, so calls such as `obp.load_poi("pitching", nrows=10)` work.
`c3d_path` accepts only paths contained by the selected discipline's C3D
directory. `REPO_ROOT`, `Discipline`, and the `DISCIPLINES` map are exported for
path and typing work.

The downloader requires Bash, an authenticated `gh` (`gh auth login` or
`GH_TOKEN`), and a SHA-256 utility; dataset/media downloads also require
`unzip`. On Windows, run it from Git Bash or WSL. Full-signal archives are
intentionally left zipped.
