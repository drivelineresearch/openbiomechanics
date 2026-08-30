"""Tests for the public repository-local ``obp`` helper."""

from __future__ import annotations

import unittest
from unittest import mock

import obp
import obp.core


class LoaderTests(unittest.TestCase):
    def test_poi_and_metadata_keys_match_one_to_one(self) -> None:
        for discipline, key in (
            ("pitching", "session_pitch"),
            ("hitting", "session_swing"),
        ):
            with self.subTest(discipline=discipline):
                poi = obp.load_poi(discipline)
                metadata = obp.load_metadata(discipline)
                self.assertFalse(poi.empty)
                self.assertFalse(metadata.empty)
                self.assertTrue(poi[key].is_unique)
                self.assertTrue(metadata[key].is_unique)
                self.assertEqual(set(poi[key]), set(metadata[key]))

    def test_read_csv_keywords_are_forwarded(self) -> None:
        self.assertEqual(len(obp.load_poi("pitching", nrows=2)), 2)
        self.assertEqual(len(obp.load_metadata("hitting", nrows=3)), 3)
        self.assertEqual(len(obp.load_hittrax(nrows=4)), 4)
        self.assertEqual(len(obp.load_hp(nrows=5)), 5)

    def test_specialized_tables_have_expected_keys(self) -> None:
        hittrax = obp.load_hittrax()
        hitting = obp.load_poi("hitting")
        hp = obp.load_hp()
        self.assertIn("session_swing", hittrax)
        self.assertTrue(
            set(hittrax["session_swing"]).issubset(hitting["session_swing"])
        )
        self.assertIn("athlete_uid", hp)
        self.assertGreater(hp["athlete_uid"].nunique(), 0)

    def test_invalid_discipline_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Unknown discipline.*'pitching'.*'hitting'"
        ):
            obp.load_poi("Pitching")  # type: ignore[arg-type]


class PathTests(unittest.TestCase):
    def test_data_directories_are_repository_relative(self) -> None:
        self.assertEqual(
            obp.c3d_dir("pitching"),
            obp.REPO_ROOT / "baseball_pitching" / "data" / "c3d",
        )
        self.assertEqual(
            obp.full_sig_dir("hitting"),
            obp.REPO_ROOT / "baseball_hitting" / "data" / "full_sig",
        )

    def test_c3d_path_stays_inside_selected_directory(self) -> None:
        path = obp.c3d_path("pitching", "000001/trial.c3d")
        self.assertEqual(path.name, "trial.c3d")
        self.assertTrue(path.is_relative_to(obp.c3d_dir("pitching").resolve()))

        with self.assertRaisesRegex(ValueError, "must be relative"):
            obp.c3d_path("pitching", obp.REPO_ROOT.parent / "outside.c3d")
        with self.assertRaisesRegex(ValueError, "must stay inside"):
            obp.c3d_path("pitching", "../../outside.c3d")


class DownloadTests(unittest.TestCase):
    @mock.patch("obp.core.subprocess.run")
    @mock.patch("obp.core.shutil.which", return_value="/bin/bash")
    def test_download_forwards_unique_discipline_and_options(
        self, _which: mock.Mock, run: mock.Mock
    ) -> None:
        obp.download(["hitting", "hitting"], media=True, mokka=True)

        run.assert_called_once_with(
            [
                "/bin/bash",
                str(obp.REPO_ROOT / "scripts" / "download_data.sh"),
                "--discipline",
                "hitting",
                "--with-media",
                "--with-mokka",
            ],
            cwd=obp.REPO_ROOT,
            check=True,
        )

    def test_empty_discipline_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            obp.download([])

    @mock.patch("obp.core.subprocess.run")
    @mock.patch("obp.core.shutil.which", return_value="/bin/bash")
    def test_optional_assets_can_be_downloaded_without_dataset(
        self, _which: mock.Mock, run: mock.Mock
    ) -> None:
        obp.download(data=False, media=True)

        run.assert_called_once_with(
            [
                "/bin/bash",
                str(obp.REPO_ROOT / "scripts" / "download_data.sh"),
                "--skip-data",
                "--with-media",
            ],
            cwd=obp.REPO_ROOT,
            check=True,
        )

    def test_data_free_download_requires_optional_asset(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires media=True or mokka=True"):
            obp.download(data=False)
        with self.assertRaisesRegex(ValueError, "disciplines cannot be set"):
            obp.download("pitching", media=True, data=False)

    @mock.patch("obp.core.shutil.which", return_value=None)
    def test_download_explains_windows_bash_requirement(
        self, _which: mock.Mock
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "Git Bash or WSL"):
            obp.download("pitching")


if __name__ == "__main__":
    unittest.main()
