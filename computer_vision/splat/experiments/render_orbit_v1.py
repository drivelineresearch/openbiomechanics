"""Re-render the orbit video for an existing checkpoint (novel-view path only)."""

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse
from pathlib import Path

import torch
from obp_splat.core import load_views, orbit_video

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=90)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cuda")
splats = torch.nn.ParameterDict(
    {k: torch.nn.Parameter(v) for k, v in ck["splats"].items()}
)
views = load_views(Path(a.data), set(), "cuda")
orbit_video(splats, views, ck["meta"], a.out, n=a.n)
print("wrote", a.out)
