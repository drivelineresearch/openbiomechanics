"""Download the eight Prime Color recordings of the 62.7 mph throw and its Theia3D C3D from the public OBP-CV
Drive folder, into <work>/videos/cam{15..22}.mp4 and <work>/theia.c3d."""

import argparse
from pathlib import Path

import gdown

VIDEOS = {
    15: "1u4-xPnQ3DCG_ocUZn-bTeieqCGm_W4Ka",
    16: "1WHEr9MXDQBv0nrEB1ahEfK6Dd-ObP2JL",
    17: "1b37Da2tNjqytG3-rRqkgawoeQJktXVsS",
    18: "1Err5ZMBa_M5npiO437LuLV4HFjE3b-_U",
    19: "1sa2wrKhEHXsEvM59PQdixIDVa_e5QEDI",
    20: "1xJDj0xlDX7FHfqPYkfBK3hiJnWhPWsLO",
    21: "1vwbuVrvkbo_wFLTEEYFm2mWRH3RshiWV",
    22: "1nE1uRyTCxTriwIQMYXC9DjXI5xKRpOuZ",
}
C3D = "1EgP-OVVpp5wv2pgorCV8Qs4-vcXZ_NjX"


def fetch(file_id, out, min_bytes):
    if out.exists() and out.stat().st_size > min_bytes:
        print("have", out, flush=True)
        return
    gdown.download(id=file_id, output=str(out), quiet=False, resume=True)
    assert out.exists() and out.stat().st_size > min_bytes, f"download failed: {out}"


def main(argv):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    Path("videos").mkdir(exist_ok=True)
    for c, file_id in VIDEOS.items():
        fetch(file_id, Path(f"videos/cam{c}.mp4"), 10**6)
    fetch(C3D, Path("theia.c3d"), 10**5)
