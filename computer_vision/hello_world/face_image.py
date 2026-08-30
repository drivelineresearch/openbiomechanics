"""Locate faces in a still image without relying on the current directory.

Run from anywhere in the repository:

    python3 computer_vision/hello_world/face_image.py
    python3 computer_vision/hello_world/face_image.py --image path/to/photo.jpg

Use ``--output-dir`` to save each detected face as a separate image. The
``face_recognition`` dependency is intentionally imported after argument
parsing so ``--help`` works before the optional CV environment is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=HERE / "Kyle_Boddy_2009.jpg",
        help="input image (default: the sample portrait beside this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for cropped face images",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")

    try:
        import face_recognition
        from PIL import Image
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install the computer-vision dependencies first: "
            "python3 -m pip install -r "
            "computer_vision/requirements-face-recognition.txt"
        ) from error

    image = face_recognition.load_image_file(str(args.image))
    locations = face_recognition.face_locations(image)
    print(f"Found {len(locations)} face(s) in {args.image}.")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, (top, right, bottom, left) in enumerate(locations, start=1):
        print(f"Face {index}: top={top}, right={right}, bottom={bottom}, left={left}")
        if args.output_dir is not None:
            output = args.output_dir / f"face_{index}.jpg"
            Image.fromarray(image[top:bottom, left:right]).save(output)
            print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
