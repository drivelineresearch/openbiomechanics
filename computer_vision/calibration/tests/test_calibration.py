"""Regression tests for calibration model and graph safeguards."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


CALIBRATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIBRATION / "scripts"))

import analyze_calibration as analysis  # noqa: E402
import build_pose_graph as pose_graph  # noqa: E402


class CalibrationSafeguardTests(unittest.TestCase):
    def test_stratified_selection_spans_full_sequence(self):
        selected = analysis.select_stratified(list(range(173)), 100)
        self.assertEqual(len(selected), 100)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 172)
        self.assertGreater(selected[50], 80)

    def test_stereo_selection_excludes_intrinsic_outlier_views(self):
        detections = {frame: np.zeros((28, 2), dtype=np.float32) for frame in range(10)}
        info = {"width": 1280, "height": 720}
        K = np.eye(3)
        distortion = np.zeros(5)
        result = analysis.stereo_pair(
            detections, detections, info, info, K, distortion, K, distortion,
            valid_a=set(range(8)), valid_b=set(range(1, 10)), limit=120,
        )
        self.assertEqual(result["status"], "insufficient_overlap")
        self.assertEqual(result["n_shared_raw"], 10)
        self.assertEqual(result["n_shared_after_intrinsic_filter"], 7)

    def test_radial_validity_rejects_optitrack_18_legacy_fold(self):
        K = np.array([[1230.111, 0.0, 632.556], [0.0, 1237.676, 320.454], [0.0, 0.0, 1.0]])
        distortion = np.array([0.1234416, -3.3470246, 0.0, 0.0, 0.0])
        result = analysis.radial_validity(K, distortion, 1280, 720)
        self.assertFalse(result["monotonic_over_full_frame"])
        self.assertLess(result["first_invalid_normalized_radius"], result["full_frame_max_normalized_radius"])

    def test_model_selection_prefers_simplest_near_best_valid_model(self):
        candidates = [
            {"model": "zero_square", "complexity_rank": 0, "heldout_median_px": 0.16,
             "heldout_p95_px": 0.39, "all_folds_monotonic_full_frame": True},
            {"model": "k1_square", "complexity_rank": 1, "heldout_median_px": 0.15,
             "heldout_p95_px": 0.37, "all_folds_monotonic_full_frame": True},
            {"model": "k1_k2", "complexity_rank": 3, "heldout_median_px": 0.14,
             "heldout_p95_px": 0.34, "all_folds_monotonic_full_frame": False},
        ]
        self.assertEqual(analysis.choose_model(candidates), "zero_square")

    def test_pose_edge_rejection_uses_heldout_tail_metrics(self):
        row = {
            "status": "fit",
            "stereo_rms_px": 0.2,
            "n_validation": 8,
            "validation_diagnostics": {
                "rotation_delta_median_deg": 0.2,
                "rotation_delta_p95_deg": 0.8,
                "translation_delta_median_mm": 10.0,
                "translation_delta_p95_mm": 75.0,
            },
        }
        self.assertEqual(pose_graph.rejection_reasons(row), ["validation_translation_p95"])

    def test_committed_results_are_physically_valid_and_provenanced(self):
        results = json.loads((CALIBRATION / "results" / "calibration_results.json").read_text())
        if results.get("schema_version") != 2:
            self.skipTest("Run the schema-v2 calibration regeneration first")
        self.assertEqual(len(results["input_provenance"]), 13)
        for source in results["input_provenance"].values():
            self.assertEqual(len(source["sha256"]), 64)
        for camera in results["calibrations"].values():
            if camera["status"] == "fit":
                self.assertTrue(camera["physical_validity"]["monotonic_over_full_frame"])


if __name__ == "__main__":
    unittest.main()
