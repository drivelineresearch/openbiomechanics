"""Build machine-readable data dictionaries from authoritative CSV headers.

Descriptions are best-effort matches parsed from each module README. Run with
``--check`` in CI to verify the committed dictionaries are current without
rewriting them.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Three README description patterns, tried per line:
#   quoted key:   "name": desc      or   'name' = desc      (YAML / python blocks)
#   backtick key: - `name`: desc                             (metadata bullet lists)
#   table row:    | name | desc | ...                        (Markdown tables)
QUOTED = re.compile(r'^\s*["\']([^"\']+)["\']\s*[:=]\s*(.+?)\s*$')
BACKTICK = re.compile(r"^\s*[-*]\s*`([^`]+)`\s*[:=]\s*(.+?)\s*$")
TABLE = re.compile(r"^\s*\|\s*`?([^|`]+?)`?\s*\|\s*([^|]+?)\s*\|")

# (dataset, CSV path, README path, output data_dictionary.csv path)
DATASETS = [
    (
        "pitching_poi",
        REPO / "baseball_pitching/data/poi/poi_metrics.csv",
        REPO / "baseball_pitching/README.md",
        REPO / "baseball_pitching/data/data_dictionary.csv",
    ),
    (
        "pitching_metadata",
        REPO / "baseball_pitching/data/metadata.csv",
        REPO / "baseball_pitching/README.md",
        REPO / "baseball_pitching/data/data_dictionary.csv",
    ),
    (
        "hitting_poi",
        REPO / "baseball_hitting/data/poi/poi_metrics.csv",
        REPO / "baseball_hitting/README.md",
        REPO / "baseball_hitting/data/data_dictionary.csv",
    ),
    (
        "hitting_metadata",
        REPO / "baseball_hitting/data/metadata.csv",
        REPO / "baseball_hitting/README.md",
        REPO / "baseball_hitting/data/data_dictionary.csv",
    ),
    (
        "hitting_hittrax",
        REPO / "baseball_hitting/data/poi/hittrax.csv",
        REPO / "baseball_hitting/README.md",
        REPO / "baseball_hitting/data/data_dictionary.csv",
    ),
    (
        "high_performance",
        REPO / "high_performance/data/hp_obp.csv",
        REPO / "high_performance/README.md",
        REPO / "high_performance/data/data_dictionary.csv",
    ),
]

FIELDS = ["dataset", "column", "dtype", "example", "description", "documented"]


def parse_descriptions(readme_path: Path) -> dict[str, str]:
    """Return column descriptions parsed from a module README."""
    descriptions: dict[str, str] = {}
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        for pattern in (QUOTED, BACKTICK, TABLE):
            match = pattern.match(line)
            if match:
                descriptions[match.group(1).strip()] = match.group(2).strip()
                break
    return descriptions


def build_rows() -> tuple[dict[Path, list[dict[str, Any]]], dict[str, Any]]:
    """Build per-file and aggregate dictionary rows without writing files."""
    by_output: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    aggregate: dict[str, Any] = {}

    for dataset, csv_path, readme_path, out_path in DATASETS:
        frame = pd.read_csv(csv_path)
        descriptions = parse_descriptions(readme_path)

        rows: list[dict[str, Any]] = []
        for column in frame.columns:
            non_null = frame[column].dropna()
            example = "" if non_null.empty else str(non_null.iloc[0])
            description = descriptions.get(column, "")
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "example": example,
                    "description": description,
                    "documented": bool(description),
                }
            )

        by_output[out_path].extend(rows)
        aggregate[dataset] = rows
        documented = sum(row["documented"] for row in rows)
        print(
            f"{dataset}: {documented} documented / "
            f"{len(rows) - documented} undocumented ({len(rows)} total)"
        )

    return by_output, aggregate


def render_outputs() -> dict[Path, bytes]:
    """Render every generated artifact to deterministic bytes."""
    by_output, aggregate = build_rows()
    outputs: dict[Path, bytes] = {}

    for out_path, rows in by_output.items():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
        outputs[out_path] = buffer.getvalue().encode("utf-8")

    outputs[REPO / "data_dictionary.json"] = json.dumps(aggregate, indent=2).encode(
        "utf-8"
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed dictionaries differ; do not write files",
    )
    args = parser.parse_args(argv)
    outputs = render_outputs()

    if args.check:
        stale = [
            path
            for path, content in outputs.items()
            if not path.exists() or path.read_bytes() != content
        ]
        if stale:
            print("stale generated dictionaries:")
            for path in stale:
                print(f"  {path.relative_to(REPO)}")
            print("run: python3 scripts/build_data_dictionary.py")
            return 1
        print("all generated dictionaries are current")
        return 0

    for path, content in outputs.items():
        path.write_bytes(content)
        if path.suffix == ".csv":
            row_count = content.count(b"\n") - 1
            print(f"wrote {path.relative_to(REPO)} ({row_count} rows)")
        else:
            print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
