"""CPU regressions for observation exclusion and saved skeleton topology."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obp_splat.skeleton import BONES, bone_frames, checkpoint_bones, pose_gaussians
from obp_splat.triangulation import select_views, triangulate


class GeometryTests(unittest.TestCase):
    def test_excluded_camera_cannot_affect_triangulation(self):
        point = np.array([0.2, 0.1, 4.0])
        images = []
        for camera, center in zip((15, 16, 18, 19), (-1.0, 0.0, 1.0, 2.0)):
            P = np.array(
                [[1000, 0, 640, -1000 * center], [0, 1000, 360, 0], [0, 0, 1, 0]]
            )
            xy = P @ np.append(point, 1)
            images.append({"name": f"cam{camera}.png", "P": P, "uv": xy[:2] / xy[2]})

        def solve():
            selected = select_views(images, {"cam19"})
            self.assertEqual(
                [i["name"] for i in selected], ["cam15.png", "cam16.png", "cam18.png"]
            )
            return triangulate([i["P"] for i in selected], [i["uv"] for i in selected])[
                0
            ]

        np.testing.assert_allclose(solve(), point, atol=1e-8)
        images[-1]["uv"] = np.array([1e9, -1e9])
        np.testing.assert_allclose(solve(), point, atol=1e-8)

    def test_insufficient_views_do_not_invent_a_joint(self):
        self.assertIsNone(triangulate([], [])[0])

    def test_expanded_checkpoint_poses_without_dimension_mismatch(self):
        torch.manual_seed(0)
        J = torch.randn(30, 3)
        bones = [*BONES, (15, 26), (16, 27), (0, 25), (28, 29), (29, 0)]
        Rc, tc, _ = bone_frames(J, bones=bones)
        weights = torch.zeros(1, len(bones))
        weights[0, -1] = 1
        ck = {"J": J[None], "bones": bones, "W": weights, "Rc": Rc, "tc": tc}
        restored = checkpoint_bones(ck)
        self.assertEqual(len(restored), 29)
        shift = torch.tensor([0.2, -0.3, 0.1])
        Rf, tf, _ = bone_frames(J + shift, bones=restored)
        can = {
            "means": torch.tensor([[0.1, 0.2, 0.3]]),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "scales": torch.zeros(1, 3),
            "opacities": torch.zeros(1),
            "sh0": torch.zeros(1, 1, 3),
            "shN": torch.zeros(1, 15, 3),
        }
        result = pose_gaussians(can, weights, Rc, tc, Rf, tf)
        torch.testing.assert_close(
            result["means"], can["means"] + shift, atol=1e-5, rtol=1e-5
        )
        del ck["bones"]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            checkpoint_bones(ck)

    def test_legacy_24_bone_checkpoint_remains_supported(self):
        ck = {
            "J": torch.zeros(1, 25, 3),
            "W": torch.zeros(2, 24),
            "Rc": torch.zeros(24, 3, 3),
            "tc": torch.zeros(24, 3),
        }
        self.assertEqual(checkpoint_bones(ck), BONES)
        ck["bones"] = [*BONES[:-1], (0, 999)]
        with self.assertRaisesRegex(ValueError, "joint indices"):
            checkpoint_bones(ck)


if __name__ == "__main__":
    unittest.main()
