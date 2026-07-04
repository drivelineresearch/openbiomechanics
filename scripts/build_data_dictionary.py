#!/usr/bin/env python3
"""Build machine-readable data dictionaries from the actual CSV headers.

The CSV header/dtypes are authoritative. Descriptions are best-effort matched by
parsing the POI/metadata dictionary blocks in each module README.md.

Outputs:
  - baseball_pitching/data/data_dictionary.csv   (pitching POI + metadata)
  - baseball_hitting/data/data_dictionary.csv     (hitting POI + metadata + hittrax)
  - high_performance/data/data_dictionary.csv     (high performance)
  - data_dictionary.json                          (repo root, all datasets)
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Three README description patterns, tried per line:
#   quoted key:   "name": desc      or   'name' = desc      (YAML / python blocks)
#   backtick key: - `name`: desc                             (metadata bullet lists)
#   table row:    | name | desc | ...                        (markdown tables)
QUOTED = re.compile(r'^\s*["\']([^"\']+)["\']\s*[:=]\s*(.+?)\s*$')
BACKTICK = re.compile(r'^\s*[-*]\s*`([^`]+)`\s*[:=]\s*(.+?)\s*$')
TABLE = re.compile(r'^\s*\|\s*`?([^|`]+?)`?\s*\|\s*([^|]+?)\s*\|')


def parse_descriptions(readme_path):
    """Return {column_name: description} scraped from a README's dictionary blocks."""
    descriptions = {}
    for line in readme_path.read_text().splitlines():
        for pattern in (QUOTED, BACKTICK, TABLE):
            match = pattern.match(line)
            if match:
                descriptions[match.group(1).strip()] = match.group(2).strip()
                break
    return descriptions


# (dataset, csv path, README path, output data_dictionary.csv path)
DATASETS = [
    ("pitching_poi", REPO / "baseball_pitching/data/poi/poi_metrics.csv",
     REPO / "baseball_pitching/README.md", REPO / "baseball_pitching/data/data_dictionary.csv"),
    ("pitching_metadata", REPO / "baseball_pitching/data/metadata.csv",
     REPO / "baseball_pitching/README.md", REPO / "baseball_pitching/data/data_dictionary.csv"),
    ("hitting_poi", REPO / "baseball_hitting/data/poi/poi_metrics.csv",
     REPO / "baseball_hitting/README.md", REPO / "baseball_hitting/data/data_dictionary.csv"),
    ("hitting_metadata", REPO / "baseball_hitting/data/metadata.csv",
     REPO / "baseball_hitting/README.md", REPO / "baseball_hitting/data/data_dictionary.csv"),
    ("hitting_hittrax", REPO / "baseball_hitting/data/poi/hittrax.csv",
     REPO / "baseball_hitting/README.md", REPO / "baseball_hitting/data/data_dictionary.csv"),
    ("high_performance", REPO / "high_performance/data/hp_obp.csv",
     REPO / "high_performance/README.md", REPO / "high_performance/data/data_dictionary.csv"),
]

FIELDS = ["dataset", "column", "dtype", "example", "description", "documented"]

by_output = defaultdict(list)
aggregate = {}

for dataset, csv_path, readme_path, out_path in DATASETS:
    df = pd.read_csv(csv_path)
    descriptions = parse_descriptions(readme_path)

    rows = []
    for column in df.columns:
        non_null = df[column].dropna()
        example = "" if non_null.empty else str(non_null.iloc[0])
        description = descriptions.get(column, "")
        rows.append({
            "dataset": dataset,
            "column": column,
            "dtype": str(df[column].dtype),
            "example": example,
            "description": description,
            "documented": bool(description),
        })

    by_output[out_path].extend(rows)
    aggregate[dataset] = rows

    documented = sum(r["documented"] for r in rows)
    print(f"{dataset}: {documented} documented / {len(rows) - documented} undocumented "
          f"({len(rows)} total)")

for out_path, rows in by_output.items():
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path.relative_to(REPO)} ({len(rows)} rows)")

json_path = REPO / "data_dictionary.json"
json_path.write_text(json.dumps(aggregate, indent=2))
print(f"wrote {json_path.relative_to(REPO)} ({len(aggregate)} datasets)")
