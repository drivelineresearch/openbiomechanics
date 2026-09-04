"""Download the 8-camera OptiTrack recordings needed for splatting from the public OBP-CV Drive folder."""
import os
import sys
from pathlib import Path

import gdown

THROW = {15:"1u4-xPnQ3DCG_ocUZn-bTeieqCGm_W4Ka",16:"1WHEr9MXDQBv0nrEB1ahEfK6Dd-ObP2JL",17:"1b37Da2tNjqytG3-rRqkgawoeQJktXVsS",18:"1Err5ZMBa_M5npiO437LuLV4HFjE3b-_U",19:"1sa2wrKhEHXsEvM59PQdixIDVa_e5QEDI",20:"1xJDj0xlDX7FHfqPYkfBK3hiJnWhPWsLO",21:"1vwbuVrvkbo_wFLTEEYFm2mWRH3RshiWV",22:"1nE1uRyTCxTriwIQMYXC9DjXI5xKRpOuZ"}
CUBE = {15:"1PpDkO9eO8_bA1t-DJQvdJLkAwybifqYJ",16:"1HswSugmA2KCAa_Zk4N0Bg71KQUCMITyb",17:"1tb1tiobmyU5fTRIFvHGEjAZCLzVtslq7",18:"16UNwYQKfYGZTxRvqjy5IblUFDEycBs1L",19:"1fJO6_DOzm9V9yukkNJBCrVWD5OYBCKSk",20:"1raWh8l32TbXpPh0rIbdQAYtto7mF52dU",21:"1oRouPsQoEcDtb2JRQDW1_x0NWlJ_jTlo",22:"1-pCV-UOtXvidnmXEng42MWuAs0DnR40p"}
root = Path(os.environ.get("OBP_SPLAT_WORK", ".")).resolve() / "videos"
root.mkdir(parents=True, exist_ok=True)
ALL = {"throw": THROW, "cube": CUBE}
sets = ALL if len(sys.argv) < 2 else {k: ALL[k] for k in sys.argv[1:] if k in ALL}
for name, ids in sets.items():
    for cam, fid in ids.items():
        out = root / f"{name}_cam{cam}.mp4"
        if out.exists() and out.stat().st_size > 1e6:
            print("have", out.name); continue
        for i in range(6):
            try:
                gdown.download(id=fid, output=str(out), quiet=True, resume=True); break
            except Exception as e:
                print("retry", out.name, i, type(e).__name__, flush=True); __import__("time").sleep(5)
        print("done", out.name, out.stat().st_size // 2**20, "MB", flush=True)
