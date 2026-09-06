"""Download public OBP-CV videos and the matching Theia3D C3D into the work directory."""

import argparse
import hashlib
import json
from pathlib import Path

THROW = {
    15: "1u4-xPnQ3DCG_ocUZn-bTeieqCGm_W4Ka",
    16: "1WHEr9MXDQBv0nrEB1ahEfK6Dd-ObP2JL",
    17: "1b37Da2tNjqytG3-rRqkgawoeQJktXVsS",
    18: "1Err5ZMBa_M5npiO437LuLV4HFjE3b-_U",
    19: "1sa2wrKhEHXsEvM59PQdixIDVa_e5QEDI",
    20: "1xJDj0xlDX7FHfqPYkfBK3hiJnWhPWsLO",
    21: "1vwbuVrvkbo_wFLTEEYFm2mWRH3RshiWV",
    22: "1nE1uRyTCxTriwIQMYXC9DjXI5xKRpOuZ",
}
CUBE = {
    15: "1PpDkO9eO8_bA1t-DJQvdJLkAwybifqYJ",
    16: "1HswSugmA2KCAa_Zk4N0Bg71KQUCMITyb",
    17: "1tb1tiobmyU5fTRIFvHGEjAZCLzVtslq7",
    18: "16UNwYQKfYGZTxRvqjy5IblUFDEycBs1L",
    19: "1fJO6_DOzm9V9yukkNJBCrVWD5OYBCKSk",
    20: "1raWh8l32TbXpPh0rIbdQAYtto7mF52dU",
    21: "1oRouPsQoEcDtb2JRQDW1_x0NWlJ_jTlo",
    22: "1-pCV-UOtXvidnmXEng42MWuAs0DnR40p",
}

C3D = "1EgP-OVVpp5wv2pgorCV8Qs4-vcXZ_NjX"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sets",
        nargs="*",
        default=["throw", "theia"],
        choices=["throw", "cube", "theia"],
    )
    args = ap.parse_args()
    import gdown

    root = Path("videos")
    root.mkdir(exist_ok=True)
    assets = []
    for name in args.sets:
        if name == "theia":
            assets.append((C3D, Path("theia.c3d"), 100_000))
        else:
            ids = {"throw": THROW, "cube": CUBE}[name]
            assets.extend(
                (fid, root / f"{name}_cam{cam}.mp4", 1_000_000)
                for cam, fid in ids.items()
            )
    manifest = {}
    for fid, out, minimum in assets:
        if not out.exists() or out.stat().st_size <= minimum:
            gdown.download(id=fid, output=str(out), quiet=False, resume=True)
        if not out.exists() or out.stat().st_size <= minimum:
            raise RuntimeError(f"Download incomplete: {out}")
        digest = hashlib.sha256()
        with out.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[str(out)] = {
            "drive_id": fid,
            "bytes": out.stat().st_size,
            "observed_sha256": digest.hexdigest(),
        }
        print("have", out, flush=True)
    with Path("download_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
