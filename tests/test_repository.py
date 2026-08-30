"""Fast checks for repository documentation, schemas, and tracked artifacts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def tracked_files(pattern: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", pattern],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        REPO / line
        for line in result.stdout.splitlines()
        if line and (REPO / line).exists()
    ]


class DocumentationTests(unittest.TestCase):
    def test_relative_markdown_and_html_links_exist(self) -> None:
        markdown_link = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        html_source = re.compile(r"""(?:src|href)=["']([^"']+)["']""")
        missing: list[str] = []

        for document in tracked_files("*.md"):
            text = document.read_text(encoding="utf-8")
            targets = [match.group(1) for match in markdown_link.finditer(text)]
            targets.extend(match.group(1) for match in html_source.finditer(text))

            for raw_target in targets:
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(
                    ("#", "http://", "https://", "mailto:")
                ):
                    continue
                target = target.split(maxsplit=1)[0]
                path_part = unquote(target.split("#", 1)[0])
                if not path_part:
                    continue
                resolved = (document.parent / path_part).resolve()
                if not resolved.exists():
                    missing.append(f"{document.relative_to(REPO)} -> {raw_target}")

        self.assertEqual(missing, [], "broken relative links:\n" + "\n".join(missing))

    def test_tracked_notebooks_and_json_are_valid(self) -> None:
        files = tracked_files("*.ipynb") + tracked_files("*.json")
        for path in files:
            with self.subTest(path=path.relative_to(REPO)):
                json.loads(path.read_text(encoding="utf-8"))

    def test_tracked_yaml_and_cff_are_valid(self) -> None:
        files = (
            tracked_files("*.yml") + tracked_files("*.yaml") + tracked_files("*.cff")
        )
        for path in files:
            with self.subTest(path=path.relative_to(REPO)):
                yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_promoted_cv_entrypoints_have_portable_help(self) -> None:
        for relative in (
            "computer_vision/hello_world/face_image.py",
            "computer_vision/hello_world/face_video.py",
            "computer_vision/hello_world/face_tracking.py",
        ):
            with self.subTest(script=relative):
                path = REPO / relative
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    cwd=REPO,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"[A-Za-z]:\\")


class DataDictionaryTests(unittest.TestCase):
    SOURCES: ClassVar[dict[str, str]] = {
        "pitching_poi": "baseball_pitching/data/poi/poi_metrics.csv",
        "pitching_metadata": "baseball_pitching/data/metadata.csv",
        "hitting_poi": "baseball_hitting/data/poi/poi_metrics.csv",
        "hitting_metadata": "baseball_hitting/data/metadata.csv",
        "hitting_hittrax": "baseball_hitting/data/poi/hittrax.csv",
        "high_performance": "high_performance/data/hp_obp.csv",
    }

    def test_dictionary_columns_exactly_match_csv_headers(self) -> None:
        dictionaries = pd.concat(
            [
                pd.read_csv(REPO / "baseball_pitching/data/data_dictionary.csv"),
                pd.read_csv(REPO / "baseball_hitting/data/data_dictionary.csv"),
                pd.read_csv(REPO / "high_performance/data/data_dictionary.csv"),
            ],
            ignore_index=True,
        )
        aggregate = json.loads((REPO / "data_dictionary.json").read_text())

        for dataset, source in self.SOURCES.items():
            with self.subTest(dataset=dataset):
                source_columns = list(pd.read_csv(REPO / source, nrows=0).columns)
                dictionary_columns = list(
                    dictionaries.loc[dictionaries["dataset"] == dataset, "column"]
                )
                aggregate_columns = [row["column"] for row in aggregate[dataset]]
                self.assertEqual(dictionary_columns, source_columns)
                self.assertEqual(aggregate_columns, source_columns)

    def test_pitching_and_hitting_columns_are_documented(self) -> None:
        for path in (
            REPO / "baseball_pitching/data/data_dictionary.csv",
            REPO / "baseball_hitting/data/data_dictionary.csv",
        ):
            dictionary = pd.read_csv(path)
            undocumented = dictionary.loc[
                ~dictionary["documented"].astype(bool), ["dataset", "column"]
            ]
            self.assertTrue(
                undocumented.empty,
                f"undocumented columns in {path.relative_to(REPO)}:\n{undocumented}",
            )

    def test_high_performance_dictionary_is_explicitly_schema_only(self) -> None:
        dictionary = pd.read_csv(REPO / "high_performance/data/data_dictionary.csv")
        self.assertEqual(len(dictionary), 53)
        self.assertTrue(dictionary["description"].fillna("").eq("").all())
        self.assertFalse(dictionary["documented"].astype(bool).any())


class ReleaseAndIgnoreTests(unittest.TestCase):
    EXPECTED_ASSETS: ClassVar[set[str]] = {
        "hitting_c3d.zip",
        "hitting_force_plate.zip",
        "hitting_joint_angles.zip",
        "hitting_joint_velos.zip",
        "hitting_landmarks.zip",
        "pitching_c3d.zip",
        "pitching_energy_flow.zip",
        "pitching_forces_moments.zip",
        "pitching_force_plate.zip",
        "pitching_joint_angles.zip",
        "pitching_joint_velos.zip",
        "pitching_landmarks.zip",
        "cv_media.zip",
        "Mokka-0.6.2-Windows.64.zip",
        "Mokka-0.6.2_MacOSX.dmg",
    }

    def test_checksum_manifest_is_complete_and_well_formed(self) -> None:
        manifest = REPO / "scripts/release_checksums.sha256"
        names: list[str] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(maxsplit=1)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            names.append(name)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), self.EXPECTED_ASSETS)

    def test_downloader_help_and_invalid_discipline(self) -> None:
        script = REPO / "scripts/download_data.sh"
        help_result = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--discipline", help_result.stdout)
        self.assertIn("--skip-data", help_result.stdout)

        invalid = subprocess.run(
            ["bash", str(script), "--discipline", "cricket"],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("unknown discipline", invalid.stderr)

        invalid_skip = subprocess.run(
            ["bash", str(script), "--skip-data"],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_skip.returncode, 2)
        self.assertIn("requires --with-media or --with-mokka", invalid_skip.stderr)

    def test_downloader_rejects_incomplete_selected_release(self) -> None:
        script = REPO / "scripts/download_data.sh"
        with tempfile.TemporaryDirectory() as temp:
            fake_bin = Path(temp) / "bin"
            fake_bin.mkdir()
            fake_gh = fake_bin / "gh"
            fake_gh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_gh.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(script), "--discipline", "pitching"],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing expected asset 'pitching_c3d.zip'", result.stderr)

    def test_downloader_explains_github_authentication(self) -> None:
        script = REPO / "scripts/download_data.sh"
        with tempfile.TemporaryDirectory() as temp:
            fake_bin = Path(temp) / "bin"
            fake_bin.mkdir()
            fake_gh = fake_bin / "gh"
            fake_gh.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            fake_gh.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(script), "--discipline", "pitching"],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gh auth login", result.stderr)

    def test_downloaded_artifacts_are_ignored(self) -> None:
        for path in (
            "baseball_pitching/data/c3d/example.c3d",
            "baseball_hitting/data/full_sig/joint_angles.csv",
            "computer_vision/demo.mp4",
            "binaries/tool.dmg",
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--quiet", path],
                    cwd=REPO,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
