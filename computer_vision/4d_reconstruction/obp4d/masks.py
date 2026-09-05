"""Athlete masks with BiRefNet (ZhengPeng7/BiRefNet on Hugging Face) for the training frames and the plate frames:
masks/cam{c}_{frame}.png and plate_masks/cam{c}_{k}.png, 255 = athlete.

  python -m obp4d masks --work W [--batch 8]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

PREP = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).cuda().half().eval()
    for src, dst in (("images", "masks"), ("plate_frames", "plate_masks")):
        Path(dst).mkdir(exist_ok=True)
        todo = [f for f in sorted(Path(src).glob("*.png")) if not (Path(dst) / f.name).exists()]
        print(src, len(todo), "images to mask", flush=True)
        for i in range(0, len(todo), a.batch):
            files = todo[i : i + a.batch]
            ims = [Image.open(f).convert("RGB") for f in files]
            with torch.no_grad():
                pred = model(torch.stack([PREP(im) for im in ims]).cuda().half())[-1].sigmoid().float().cpu()
            for f, im, p in zip(files, ims, pred):
                Image.fromarray((p[0].numpy() * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR).save(Path(dst) / f.name)
            if i % (50 * a.batch) == 0:
                print(i, "/", len(todo), flush=True)
