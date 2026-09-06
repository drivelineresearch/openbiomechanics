from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

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

    def test_contact_is_not_selected_inside_a_marker_gap(self) -> None:
        poses = [
            {"grip": [0.0, 0.0, 0.0], "barrel": [x, 0.0, 0.0]}
            for x in (0.0, 0.1, 0.2, 0.9, 1.0, 1.05, 1.1)
        ]
        poses[2] = None
        analysis = APP._motion_analysis(pathlib.Path("trial.c3d"), poses, None, 100)
        contact = analysis["events"]["contact"]
        self.assertEqual(contact, 4)
        self.assertIsNotNone(poses[contact])
        self.assertIsNotNone(
            APP._estimated_ball(pathlib.Path("trial.c3d"), poses, contact, 100)[contact]
        )

    def test_absent_bat_has_no_peak_or_contact_event(self) -> None:
        analysis = APP._motion_analysis(
            pathlib.Path("trial.c3d"), [None] * 5, None, 100
        )
        self.assertIsNone(analysis["events"]["peakSpeed"])
        self.assertIsNone(analysis["events"]["contact"])

    def test_athlete_metadata_extracted_from_hitting_c3d(self) -> None:
        trial_path = (
            REPO
            / "baseball_hitting"
            / "data"
            / "c3d"
            / "000004"
            / "000004_000103_75_236_R_003_972.c3d"
        )
        if not trial_path.exists():
            self.skipTest("Optional released hitting C3D is not installed")
        motion = APP.load_motion(trial_path)
        self.assertEqual(motion["athleteHeightIn"], 75.0)
        self.assertEqual(motion["athleteWeightLb"], 236.0)
        self.assertEqual(motion["hitterSide"], "R")
        self.assertEqual(motion["analysis"]["metrics"]["athleteHeightIn"], 75.0)
        self.assertGreater(motion["ground"], 0.0)

    def test_load_motion_extracts_metadata_from_synthetic_c3d(self) -> None:
        labels = ["Marker1", "Marker2", "LASI", "RASI"]
        points = np.zeros((4, len(labels), 5))
        points[3, :, :] = 0.0
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "parameters": {
                    "POINT": {
                        "LABELS": {"value": labels},
                        "RATE": {"value": [100.0]},
                    }
                },
                "data": {"points": points},
            }
        )
        with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
            synthetic_path = pathlib.Path("000004_000103_75_236_R_003_972.c3d")
            motion = APP.load_motion(synthetic_path)
            self.assertEqual(motion["athleteHeightIn"], 75.0)
            self.assertEqual(motion["athleteWeightLb"], 236.0)
            self.assertEqual(motion["hitterSide"], "R")
            self.assertEqual(
                motion["analysis"]["metrics"]["recordedExitVelocityMph"], 97.2
            )


if __name__ == "__main__":
    unittest.main()
