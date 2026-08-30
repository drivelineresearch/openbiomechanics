"""Identify known faces in a prerecorded video.

The sample portraits beside this script are used as references by default; a
custom ``--training-dir`` may contain ``.jpg`` or ``.jpeg`` files whose stems
become labels. Download optional demo media or provide any local video:

    scripts/download_data.sh --skip-data --with-media
    python3 computer_vision/hello_world/face_video.py --video path/to/video.mp4

Optional CV dependencies are imported only after argument parsing, so
``--help`` works in the repository's minimal analysis environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV_ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="input video")
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=HERE,
        help="reference portraits (default: sample portraits beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "face_identification_output.mp4",
        help="annotated output video",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="recognition resize factor in (0, 1] (default: 0.5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    if not args.training_dir.is_dir():
        parser.error(f"training directory not found: {args.training_dir}")
    if not 0 < args.scale <= 1:
        parser.error("--scale must be greater than 0 and at most 1")

    sys.path.insert(0, str(CV_ROOT))
    try:
        import cv2
        from utils import (
            draw_face_box,
            identify_faces,
            iter_frames,
            load_known_faces,
            make_writer,
        )
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install the computer-vision dependencies first: "
            "python3 -m pip install -r "
            "computer_vision/requirements-face-recognition.txt"
        ) from error

    known_encodings, known_names = load_known_faces(args.training_dir)
    if not known_encodings:
        raise SystemExit(f"No usable face portraits found in {args.training_dir}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(capture, args.output, fps=fps)
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {args.output}")

    try:
        for frame in iter_frames(capture):
            locations, names = identify_faces(
                frame, known_encodings, known_names, scale=args.scale
            )
            for (top, right, bottom, left), name in zip(locations, names):
                draw_face_box(frame, top, right, bottom, left, name)
            writer.write(frame)
    finally:
        writer.release()
        capture.release()

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
