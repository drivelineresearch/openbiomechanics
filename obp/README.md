# obp

Tiny importable loader for the OpenBiomechanics dataset. No build step — add
the repo root to `sys.path` and import.

```python
import sys
sys.path.insert(0, "/path/to/openbiomechanics")

import obp

# CSV metrics as pandas DataFrames
poi   = obp.load_poi("pitching")      # baseball_pitching/data/poi/poi_metrics.csv
meta  = obp.load_metadata("hitting")  # baseball_hitting/data/metadata.csv
htx   = obp.load_hittrax()            # baseball_hitting/data/poi/hittrax.csv
hp    = obp.load_hp()                 # high_performance/data/hp_obp.csv

# Paths only for the large signals — call ezc3d yourself
import ezc3d
path = obp.c3d_path("pitching", "000822/000822_1.c3d")
c3d  = ezc3d.c3d(str(path))

obp.c3d_dir("pitching")       # baseball_pitching/data/c3d
obp.full_sig_dir("pitching")  # baseball_pitching/data/full_sig

# Fetch the large release artifacts (raw C3D + full_sig zips)
obp.download()                 # pitching + hitting
obp.download(media=True, mokka=True)
```

`discipline` is `"pitching"` or `"hitting"`. `REPO_ROOT` and the
`DISCIPLINES` map are exported for path work.
