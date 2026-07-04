"""Path-correct loaders for the OpenBiomechanics dataset.

Consumers add the repo root to ``sys.path`` and ``from obp import ...``.
CSV loaders return pandas DataFrames; C3D / full_sig helpers return resolved
``Path`` objects so callers use ezc3d (or anything else) directly.
"""

import subprocess
from pathlib import Path

import pandas as pd

# obp/core.py -> obp/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

DISCIPLINES = {
    "pitching": "baseball_pitching",
    "hitting": "baseball_hitting",
}


def _folder(discipline):
    return REPO_ROOT / DISCIPLINES[discipline]


def load_poi(discipline):
    return pd.read_csv(_folder(discipline) / "data" / "poi" / "poi_metrics.csv")


def load_metadata(discipline):
    return pd.read_csv(_folder(discipline) / "data" / "metadata.csv")


def load_hittrax():
    return pd.read_csv(REPO_ROOT / "baseball_hitting" / "data" / "poi" / "hittrax.csv")


def load_hp():
    return pd.read_csv(REPO_ROOT / "high_performance" / "data" / "hp_obp.csv")


def c3d_dir(discipline):
    return _folder(discipline) / "data" / "c3d"


def c3d_path(discipline, filename):
    return c3d_dir(discipline) / filename


def full_sig_dir(discipline):
    return _folder(discipline) / "data" / "full_sig"


def download(disciplines=None, media=False, mokka=False):
    """Fetch the large release artifacts via scripts/download_data.sh.

    ``disciplines`` is accepted for API symmetry; the download script always
    restores both pitching and hitting from dataset-v1.
    """
    cmd = [str(REPO_ROOT / "scripts" / "download_data.sh")]
    if media:
        cmd.append("--with-media")
    if mokka:
        cmd.append("--with-mokka")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
