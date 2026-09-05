"""python -m obp4d <step> --work DIR [step options]

  fetch        the eight videos and the Theia3D C3D from the public OBP-CV Drive folder
  dataset      frames for training and for the background plates
  masks        athlete masks (BiRefNet)
  plates       median background plates
  background   DUSt3R seed and Gaussian refinement of the static background
  body         the athlete: one Gaussian body skinned to the Theia3D segments, with per-frame corrections
  render       holdout | quad | ring videos

Every step runs inside the work directory; nothing is written into the repository. Add --help after a step
for its options.
"""

import importlib
import os
import sys
from pathlib import Path

STEPS = {
    "fetch": ("fetch", "main"),
    "dataset": ("dataset", "main"),
    "masks": ("masks", "main"),
    "plates": ("dataset", "plates"),
    "background": ("background", "main"),
    "body": ("body", "main"),
    "render": ("render", "main"),
}


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in STEPS:
        print(__doc__)
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    step, rest = argv[0], argv[1:]
    assert "--work" in rest, "pass --work DIR"
    i = rest.index("--work")
    work = Path(rest[i + 1]).resolve()
    del rest[i : i + 2]
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    module, fn = STEPS[step]
    getattr(importlib.import_module(f"obp4d.{module}"), fn)(rest)
    return 0
