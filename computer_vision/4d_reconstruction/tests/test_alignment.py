"""CPU checks of transform direction, units, offsets, and evaluation scope."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obp4d import theia


class AlignmentTests(unittest.TestCase):
    def test_segment_translation_and_orientation_transform(self):
        raw = np.tile(np.eye(4)[:, :, None, None], (1, 1, 2, 3))
        raw[:3, 3, 0, :] = np.array([[1000], [2000], [3000]])
        c = {
            "parameters": {
                "ROTATION": {"LABELS": {"value": ["pelvis_4X4", "torso_4X4"]}}
            },
            "data": {"rotations": raw},
        }
        with patch.object(theia.ezc3d, "c3d", return_value=c):
            transforms, names = theia.load_segments("synthetic.c3d")
        self.assertEqual(names, ["pelvis", "torso"])
        self.assertEqual(transforms.shape, (3, 2, 4, 4))
        np.testing.assert_allclose(
            transforms[0, 0, :3, 3],
            theia.SCALE * theia.R @ [1000, 2000, 3000] + theia.T,
        )
        np.testing.assert_allclose(transforms[0, 0, :3, :3], theia.R)
        np.testing.assert_allclose(transforms[0, 0, 3], [0, 0, 0, 1])
        self.assertAlmostEqual(
            np.linalg.norm(transforms[0, 0, :3, 3] - transforms[0, 1, :3, 3]),
            theia.SCALE * np.linalg.norm([1000, 2000, 3000]),
        )

    def test_video_offset_is_applied_once_and_bounds_checked(self):
        self.assertEqual(theia.video_to_c3d_index(950, 1200), 951)
        self.assertEqual(theia.video_to_c3d_index(1099, 1101), 1100)
        with self.assertRaises(ValueError):
            theia.video_to_c3d_index(1100, 1101)
        with self.assertRaises(ValueError):
            theia.video_to_c3d_index(-2, 1101)

    def test_alignment_is_not_claimed_independent_of_scored_camera(self):
        protocol = theia.evaluation_protocol(19, range(800, 1100))
        self.assertIn(19, protocol["alignment_fit_cameras"])
        self.assertEqual(protocol["appearance_excluded_cameras"], [19])
        self.assertFalse(protocol["strict_independent_camera_evaluation"])
        self.assertEqual(protocol["scored_video_frames"], list(range(800, 1100, 10)))
        self.assertEqual(
            theia.evaluation_protocol(0, range(3))["appearance_excluded_cameras"], []
        )


if __name__ == "__main__":
    unittest.main()
