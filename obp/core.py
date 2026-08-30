"""Small, path-correct loaders for the OpenBiomechanics dataset.

The package is intentionally repository-local: CSV loaders return pandas
``DataFrame`` objects, while C3D and full-signal helpers return resolved
``Path`` objects for callers to open with their preferred tools.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# obp/core.py -> obp/ -> repository root
REPO_ROOT = Path(__file__).resolve().parents[1]

Discipline = Literal["pitching", "hitting"]
DISCIPLINES: dict[Discipline, str] = {
    "pitching": "baseball_pitching",
    "hitting": "baseball_hitting",
}


def _folder(discipline: Discipline) -> Path:
    """Return a discipline directory, with an actionable error for bad input."""
    try:
        folder = DISCIPLINES[discipline]
    except KeyError:
        allowed = ", ".join(repr(name) for name in DISCIPLINES)
        raise ValueError(
            f"Unknown discipline {discipline!r}; expected one of: {allowed}"
        ) from None
    return REPO_ROOT / folder


def _normalize_disciplines(
    disciplines: Discipline | Iterable[Discipline] | None,
) -> tuple[Discipline, ...]:
    if disciplines is None:
        requested = tuple(DISCIPLINES)
    elif isinstance(disciplines, str):
        requested = (disciplines,)
    else:
        requested = tuple(disciplines)

    if not requested:
        raise ValueError("At least one discipline is required")

    # Validate and de-duplicate while retaining caller order.
    unique: list[Discipline] = []
    for discipline in requested:
        _folder(discipline)
        if discipline not in unique:
            unique.append(discipline)
    return tuple(unique)


def load_poi(discipline: Discipline, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load the pitching or hitting point-of-interest table."""
    path = _folder(discipline) / "data" / "poi" / "poi_metrics.csv"
    return pd.read_csv(path, **read_csv_kwargs)


def load_metadata(discipline: Discipline, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load the pitching or hitting trial metadata table."""
    path = _folder(discipline) / "data" / "metadata.csv"
    return pd.read_csv(path, **read_csv_kwargs)


def load_hittrax(**read_csv_kwargs: Any) -> pd.DataFrame:
    """Load the hitting HitTrax table."""
    path = REPO_ROOT / "baseball_hitting" / "data" / "poi" / "hittrax.csv"
    return pd.read_csv(path, **read_csv_kwargs)


def load_hp(**read_csv_kwargs: Any) -> pd.DataFrame:
    """Load the high-performance assessment table."""
    path = REPO_ROOT / "high_performance" / "data" / "hp_obp.csv"
    return pd.read_csv(path, **read_csv_kwargs)


def c3d_dir(discipline: Discipline) -> Path:
    """Return the directory populated by the raw C3D download."""
    return _folder(discipline) / "data" / "c3d"


def c3d_path(discipline: Discipline, filename: str | Path) -> Path:
    """Resolve a C3D path while preventing escape from its data directory."""
    base = c3d_dir(discipline).resolve()
    relative = Path(filename)
    if relative.is_absolute():
        raise ValueError("filename must be relative to the discipline's C3D directory")

    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("filename must stay inside the discipline's C3D directory")
    return resolved


def full_sig_dir(discipline: Discipline) -> Path:
    """Return the directory populated by the full-signal download."""
    return _folder(discipline) / "data" / "full_sig"


def download(
    disciplines: Discipline | Iterable[Discipline] | None = None,
    media: bool = False,
    mokka: bool = False,
    *,
    data: bool = True,
) -> None:
    """Fetch verified release artifacts through ``download_data.sh``.

    ``disciplines`` may be ``"pitching"``, ``"hitting"``, an iterable of both,
    or ``None`` (the default, meaning both). The optional media and Mokka assets
    are added to the selected dataset download. Set ``data=False`` with
    ``media=True`` and/or ``mokka=True`` to fetch optional assets without the
    roughly 1.1 GB dataset release.
    """
    if data:
        requested = _normalize_disciplines(disciplines)
    else:
        if disciplines is not None:
            raise ValueError("disciplines cannot be set when data=False")
        if not (media or mokka):
            raise ValueError("data=False requires media=True or mokka=True")
        requested = ()

    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError(
            "Downloading requires Bash plus the gh and unzip commands. "
            "On Windows, run from Git Bash or WSL."
        )

    cmd = [bash, str(REPO_ROOT / "scripts" / "download_data.sh")]
    if data:
        for discipline in requested:
            cmd.extend(("--discipline", discipline))
    else:
        cmd.append("--skip-data")
    if media:
        cmd.append("--with-media")
    if mokka:
        cmd.append("--with-mokka")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
