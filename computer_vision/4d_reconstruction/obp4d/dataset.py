"""Frames from the videos: the pitch window at the full 360 fps for training, and every fifth frame of the whole
recording for the background plates. Then, once the masks exist, the plates themselves: per camera, per pixel,
the median over the sampled frames where the dilated athlete mask is empty, which erases the athlete and his
shadow. Pixels the athlete covers in every sampled frame fall back to a plain median.

  python -m obp4d dataset --work W [--frames 800 1100]
  python -m obp4d plates --work W
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

from .rig import CAMS

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
PLATE_STRIDE = 5


def extract(video, select, first, out_pattern):
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", video, "-vf", f"select='{select}'", "-vsync", "0", "-start_number", str(first), out_pattern], check=True)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, nargs=2, default=[800, 1100], help="video frame range, [first, last)")
    a = ap.parse_args(argv)
    Path("images").mkdir(exist_ok=True)
    Path("plate_frames").mkdir(exist_ok=True)
    for c in CAMS:
        if not Path(f"images/cam{c}_{a.frames[1] - 1:04d}.png").exists():
            extract(f"videos/cam{c}.mp4", f"between(n,{a.frames[0]},{a.frames[1] - 1})", a.frames[0], f"images/cam{c}_%04d.png")
        if not Path(f"plate_frames/cam{c}_0000.png").exists():
            extract(f"videos/cam{c}.mp4", f"not(mod(n,{PLATE_STRIDE}))", 0, f"plate_frames/cam{c}_%04d.png")
        n_train, n_plate = len(list(Path("images").glob(f"cam{c}_*.png"))), len(list(Path("plate_frames").glob(f"cam{c}_*.png")))
        assert n_train == a.frames[1] - a.frames[0], f"camera {c}: {n_train} training frames extracted"
        print(f"cam{c}: {n_train} training frames, {n_plate} plate frames", flush=True)


def plates(argv):
    argparse.ArgumentParser(description="median background plates from plate_frames and plate_masks").parse_args(argv)
    Path("plates").mkdir(exist_ok=True)
    for c in CAMS:
        files = sorted(Path("plate_frames").glob(f"cam{c}_*.png"))
        imgs = torch.stack([torch.from_numpy(np.array(Image.open(f).convert("RGB"))) for f in files]).cuda()
        mk = torch.stack([torch.from_numpy(np.array(Image.open(f"plate_masks/{f.name}").convert("L"))) for f in files]).cuda() > 20
        mk = torch.nn.functional.max_pool2d(mk.float()[:, None], 31, 1, 15)[:, 0] > 0  # dilate 15 px so soft mask edges never leak in
        x = imgs.float()
        x[mk] = float("nan")
        med = torch.nanmedian(x, dim=0).values
        holes = torch.isnan(med[..., 0])
        med[holes] = imgs.float().median(0).values[holes]
        Image.fromarray(med.clamp(0, 255).byte().cpu().numpy()).save(f"plates/cam{c}.png")
        print(f"cam{c}: {len(files)} frames, {int(holes.sum())} always-covered pixels", flush=True)
