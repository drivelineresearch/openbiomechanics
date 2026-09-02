from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "swing_visualizer_app", REPO / "swing_visualizer" / "app.py"
)
assert SPEC is not None and SPEC.loader is not None
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


class SwingVisualizerTests(unittest.TestCase):
    def test_list_trials_excludes_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "nested").mkdir()
            (root / "swing_954.c3d").touch()
            (root / "nested" / "MODEL.c3d").touch()
            (root / "notes.txt").touch()

            self.assertEqual(
                APP.list_trials(root),
                [{"path": "swing_954.c3d", "name": "swing_954"}],
            )

    def test_resolve_trial_rejects_paths_outside_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = pathlib.Path(directory)
            root = parent / "data"
            root.mkdir()
            trial = root / "swing.c3d"
            trial.touch()

            self.assertEqual(APP.resolve_trial(root, "swing.c3d"), trial)
            with self.assertRaisesRegex(ValueError, "Invalid trial path"):
                APP.resolve_trial(root, "../outside.c3d")
            with self.assertRaisesRegex(ValueError, "Invalid trial path"):
                APP.resolve_trial(root, "notes.txt")

    def test_motion_analysis_uses_peak_speed_as_estimated_contact(self) -> None:
        poses = [
            {"grip": [0.0, 0.0, 0.0], "barrel": [position, 0.0, 0.0]}
            for position in (0.0, 0.1, 0.4, 1.0, 1.2)
        ]

        analysis = APP._motion_analysis(
            pathlib.Path("trial_954.c3d"), poses, None, 100.0
        )

        self.assertEqual(analysis["events"]["contact"], 2)
        self.assertEqual(analysis["metrics"]["recordedExitVelocityMph"], 95.4)
        self.assertEqual(analysis["confidence"], "High")

    def test_bat_poses_require_two_visible_markers(self) -> None:
        points = np.full((4, 2, 2), np.nan)
        points[3, :, :] = -1.0

        self.assertEqual(
            APP._bat_poses(points, ["Marker1", "Marker2"], [0, 1]), [None, None]
        )


if __name__ == "__main__":
    unittest.main()
