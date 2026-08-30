"""Dependency-light tests for the promoted computer-vision helper logic."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_utils(cv2: object, face_recognition: object) -> types.ModuleType:
    path = REPO / "computer_vision" / "utils.py"
    spec = importlib.util.spec_from_file_location("obp_test_cv_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load CV helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"cv2": cv2, "face_recognition": face_recognition}
    ):
        spec.loader.exec_module(module)
    return module


class ComputerVisionHelperTests(unittest.TestCase):
    def test_identify_faces_rescales_boxes_and_uses_nearest_match(self) -> None:
        fake_cv2 = types.SimpleNamespace(resize=lambda *_args, **_kwargs: "resized")
        fake_faces = types.SimpleNamespace(
            face_locations=lambda frame: [(1, 4, 3, 2)],
            face_encodings=lambda frame, locations: ["observed"],
            compare_faces=lambda known, observed, tolerance: [False, True],
            face_distance=lambda known, observed: np.array([0.8, 0.1]),
        )
        helpers = load_utils(fake_cv2, fake_faces)

        locations, names = helpers.identify_faces(
            object(), ["first", "second"], ["First", "Second"], scale=0.5
        )

        self.assertEqual(locations, [(2, 8, 6, 4)])
        self.assertEqual(names, ["Second"])

    def test_known_face_loader_is_sorted_and_uses_portable_labels(self) -> None:
        fake_cv2 = types.SimpleNamespace()
        fake_faces = types.SimpleNamespace(
            load_image_file=lambda path: Path(path).name,
            face_encodings=lambda image: [f"encoding:{image}"],
        )
        helpers = load_utils(fake_cv2, fake_faces)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Second_Person.jpeg").write_bytes(b"fixture")
            (root / "First_Person.JPG").write_bytes(b"fixture")
            (root / "ignored.png").write_bytes(b"fixture")

            encodings, names = helpers.load_known_faces(root)

        self.assertEqual(
            encodings,
            ["encoding:First_Person.JPG", "encoding:Second_Person.jpeg"],
        )
        self.assertEqual(names, ["First Person", "Second Person"])


if __name__ == "__main__":
    unittest.main()
