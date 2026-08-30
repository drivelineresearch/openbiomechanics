"""Shared OpenCV and face-recognition helpers for educational CV examples."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import face_recognition


def load_known_faces(
    training_dir: str | Path, name: str | None = None
) -> tuple[list[Any], list[str]]:
    """Encode portraits in ``training_dir`` and return encodings plus labels.

    If ``name`` is omitted, each filename stem becomes its label. Supplying a
    name preserves compatibility with older examples that use one subject.
    """
    encodings: list[Any] = []
    names: list[str] = []
    root = Path(training_dir)
    if not root.is_dir():
        print(f"Training directory not found: {root}")
        return encodings, names

    images = sorted(
        path for path in root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}
    )
    for path in images:
        print(f"Processing {path.name}...")
        image = face_recognition.load_image_file(str(path))
        found = face_recognition.face_encodings(image)
        if found:
            encodings.append(found[0])
            names.append(name or path.stem.replace("_", " "))
        else:
            print(f"No faces found in {path.name}")
    print(f"Loaded {len(encodings)} face encoding(s).")
    return encodings, names


def identify_faces(
    frame: Any,
    known_encodings: Sequence[Any],
    known_names: Sequence[str],
    *,
    scale: float = 0.5,
    tolerance: float = 0.6,
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Identify faces in a frame and return full-resolution boxes and labels."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than 0 and at most 1")
    if len(known_encodings) != len(known_names):
        raise ValueError("known face encodings and names must have equal lengths")

    resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    locations = face_recognition.face_locations(resized)
    encodings = face_recognition.face_encodings(resized, locations)
    labels: list[str] = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=tolerance
        )
        distances = face_recognition.face_distance(known_encodings, encoding)
        best = int(distances.argmin())
        labels.append(known_names[best] if matches[best] else "Unknown")

    full_size = [
        tuple(round(coordinate / scale) for coordinate in location)
        for location in locations
    ]
    return full_size, labels


def make_writer(capture: Any, path: str | Path, fps: float = 30.0) -> Any:
    """Create an MP4 writer at ``path`` sized to an open capture."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def iter_frames(capture: Any) -> Iterator[Any]:
    """Yield frames from an open video capture until its stream ends."""
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        yield frame


def draw_face_box(
    frame: Any, top: int, right: int, bottom: int, left: int, name: str
) -> None:
    """Draw a red face box with a filled name label beneath it."""
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(
        frame,
        name,
        (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 255, 255),
        1,
    )
