#!/usr/bin/env python3
"""00_reorganize_raw_tracking_tables.py
=================================================
Reorganize raw FIM-Track measurement tables into a trajectory-wise wide format.

This script is the first step of the preprocessing pipeline. It is intended for
raw CSV exports in which each row corresponds to one measurement type at one
frame, and each subsequent column corresponds to one tracked larva.

What this script does
---------------------
1. Reads one or more raw per-recording tracking tables.
2. Keeps only the measurement channels used downstream in the analysis.
3. Reorganizes the data into a wide table indexed by trajectory identity and frame.
4. Adds a recording index to trajectory IDs so that labels remain unique after
   concatenating multiple recordings.
5. Writes both per-recording intermediate tables and a concatenated output table.

Important note
--------------
The original project contains helper functions for track relabeling, gap
interpolation, and removal of non-moving larvae. In the execution path used for
this repository, those optional correction steps are left disabled, so the final
output corresponds directly to the organized per-recording tables concatenated
across recordings.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


# Measurements retained from the raw FIM-Track exports.
MEASUREMENTS_TO_KEEP = [
    "mom_x", "mom_y",
    "head_x", "head_y",
    "spinepoint_1_x", "spinepoint_1_y",
    "spinepoint_2_x", "spinepoint_2_y",
    "spinepoint_3_x", "spinepoint_3_y",
    "tail_x", "tail_y",
    "perimeter", "area", "spine_length",
    "radius_1", "radius_2", "radius_3",
    "is_coiled", "is_well_oriented",
]


def raw_to_wide_format(input_csv: Path, output_csv: Path, max_frame: int) -> None:
    """Convert a raw per-measurement tracking table into a wide trajectory table.

    Parameters
    ----------
    input_csv:
        Raw FIM-Track export for a single recording.
    output_csv:
        Destination path for the reorganized table.
    max_frame:
        Maximum frame index to keep.
    """
    data_by_larva_and_time: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)

    with input_csv.open("r", newline="", encoding="utf-8") as infile, \
         output_csv.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        larva_names = header[1:]

        for row in reader:
            measurement_and_time = row[0]
            measurement_name = measurement_and_time.split("(")[0]
            frame = int(measurement_and_time.split("(")[1].rstrip(")"))

            # Ignore frames outside the requested analysis window.
            if frame > max_frame:
                continue

            # Ignore measurements that are not used in later scripts.
            if measurement_name not in MEASUREMENTS_TO_KEEP:
                continue

            for column_index, value in enumerate(row[1:], start=1):
                if value == "":
                    continue
                larva = larva_names[column_index - 1]
                data_by_larva_and_time[(larva, frame)][measurement_name] = value

        writer.writerow(["larva", "time", *MEASUREMENTS_TO_KEEP])
        for (larva, frame), measurements in data_by_larva_and_time.items():
            writer.writerow(
                [larva, frame] + [measurements.get(name, "") for name in MEASUREMENTS_TO_KEEP]
            )


def process_recordings(raw_tables: Iterable[Path], max_frame: int, output_dir: Path) -> Path:
    """Process a list of raw recordings and concatenate them into one table.

    Each per-recording trajectory label is prefixed with the recording index,
    producing labels of the form ``trajectory_<video_index>_<local_id>``.

    Returns
    -------
    Path
        Path to the concatenated output CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tables = []

    for video_index, raw_csv in enumerate(raw_tables):
        intermediate_csv = output_dir / f"recording_{video_index:02d}_wide.csv"
        raw_to_wide_format(raw_csv, intermediate_csv, max_frame=max_frame)

        table = pd.read_csv(intermediate_csv)
        table["larva"] = table["larva"].apply(
            lambda value: f"trajectory_{video_index}_{value.split('(')[1].split(')')[0]}"
        )
        per_recording_csv = output_dir / f"recording_{video_index:02d}_relabeled.csv"
        table.to_csv(per_recording_csv, index=False)
        all_tables.append(table)

    merged = pd.concat(all_tables, ignore_index=True)
    merged_csv = output_dir / "merged_tracking_table_unscaled.csv"
    merged.to_csv(merged_csv, index=False)
    return merged_csv


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw_recordings"
    output_dir = project_root / "outputs" / "00_reorganized_tables"

    # Default expectation: raw tables named table_1.csv ... table_10.csv.
    raw_tables = [raw_dir / f"table_{i}.csv" for i in range(1, 11)]
    max_frame = 36000  # 60 min at 10 fps

    missing = [path.name for path in raw_tables if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw recording tables were not found. Expected files in "
            f"{raw_dir}: {missing}"
        )

    merged_csv = process_recordings(raw_tables, max_frame=max_frame, output_dir=output_dir)
    print(f"Saved merged tracking table to: {merged_csv}")


if __name__ == "__main__":
    main()
