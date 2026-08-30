"""Identify known faces in a live camera feed.

The sample portraits beside this script are used as references by default. Use
``--training-dir`` for custom ``.jpg``/``.jpeg`` portraits and ``--camera`` to
select a different OpenCV camera index. Press ``q`` to quit.

    python3 computer_vision/hello_world/face_tracking.py --camera 0

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
    parser.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default: 0)"
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=HERE,
        help="reference portraits (default: sample portraits beside this script)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="recognition resize factor in (0, 1] (default: 0.25)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.training_dir.is_dir():
        parser.error(f"training directory not found: {args.training_dir}")
    if not 0 < args.scale <= 1:
        parser.error("--scale must be greater than 0 and at most 1")

    sys.path.insert(0, str(CV_ROOT))
    try:
        import cv2
        from utils import draw_face_box, identify_faces, load_known_faces
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install the computer-vision dependencies first: "
            "python3 -m pip install -r "
            "computer_vision/requirements-face-recognition.txt"
        ) from error

    known_encodings, known_names = load_known_faces(args.training_dir)
    if not known_encodings:
        raise SystemExit(f"No usable face portraits found in {args.training_dir}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"Could not open camera index {args.camera}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Camera stopped returning frames")
            locations, names = identify_faces(
                frame, known_encodings, known_names, scale=args.scale
            )
            for (top, right, bottom, left), name in zip(locations, names):
                draw_face_box(frame, top, right, bottom, left, name)
            cv2.imshow("Face identification (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
