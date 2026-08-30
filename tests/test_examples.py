"""Substantive fixture tests for public examples that normally need large data."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_example(filename: str) -> types.ModuleType:
    """Import a numerically named example without executing its CLI guard."""
    path = REPO / "examples" / filename
    spec = importlib.util.spec_from_file_location(f"obp_example_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExampleTests(unittest.TestCase):
    def test_read_c3d_runs_with_small_synthetic_fixture(self) -> None:
        example = load_example("02_read_c3d.py")
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "parameters": {
                    "POINT": {
                        "LABELS": {"value": ["RWRA"]},
                        "RATE": {"value": [100.0]},
                    }
                },
                "data": {
                    "points": np.array(
                        [
                            [[0.0, 1.0, 2.0]],
                            [[1.0, 2.0, 3.0]],
                            [[2.0, 3.0, 4.0]],
                            [[0.0, 0.0, 0.0]],
                        ]
                    )
                },
            }
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "c3d"
            trial = root / "athlete-001" / "trial.c3d"
            trial.parent.mkdir(parents=True)
            trial.write_bytes(b"synthetic fixture")
            output = Path(temp) / "trajectory.png"

            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                self.assertEqual(example.main(root, output), 0)
            self.assertGreater(output.stat().st_size, 0)

    def test_full_signal_join_runs_with_small_csv_fixtures(self) -> None:
        example = load_example("03_join_fullsig.py")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame(
                {
                    "session_pitch": ["p1", "p1"],
                    "time": [0.0, 1 / 360],
                    "joint_angle": [10.0, 11.0],
                }
            ).to_csv(root / "joint_angles.csv", index=False)
            pd.DataFrame(
                {
                    "session_pitch": ["p1", "p1"],
                    "time": [0.0, 1 / 360],
                    "joint_velocity": [20.0, 21.0],
                }
            ).to_csv(root / "joint_velos.csv", index=False)

            self.assertEqual(example.main(root), 0)


if __name__ == "__main__":
    unittest.main()
