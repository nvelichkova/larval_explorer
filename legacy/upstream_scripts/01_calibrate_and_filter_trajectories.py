#!/usr/bin/env python3
"""01_calibrate_and_filter_trajectories.py
=================================================
Convert frame indices and pixel coordinates into physical units, then remove
short trajectory segments.

Input
-----
A merged, unlabeled per-frame trajectory table such as the output of
``00_reorganize_raw_tracking_tables.py``.

Output
------
A calibrated trajectory table in seconds and millimetres, filtered to keep only
trajectory segments lasting at least 60 s.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


POSITION_COLUMNS = [
    "mom_x", "mom_y",
    "head_x", "head_y",
    "spinepoint_1_x", "spinepoint_1_y",
    "spinepoint_2_x", "spinepoint_2_y",
    "spinepoint_3_x", "spinepoint_3_y",
    "tail_x", "tail_y",
]


def calibrate_and_filter(
    input_csv: Path,
    output_csv: Path,
    frame_rate_fps: float = 10.0,
    millimetres_per_pixel: float = 240.0 / 2048.0,
    min_duration_seconds: float = 60.0,
) -> None:
    """Calibrate timestamps/positions and filter out short tracks."""
    data = pd.read_csv(input_csv)

    # Convert integer frame indices into seconds.
    data["time"] = (data["time"] / frame_rate_fps).round(1)

    # Convert all tracked positions from pixels into millimetres.
    for column in POSITION_COLUMNS:
        data[column] = (data[column] * millimetres_per_pixel).round(3)

    # Compute trajectory duration and keep only long enough tracks.
    duration_by_trajectory = data.groupby("larva")["time"].max() - data.groupby("larva")["time"].min()
    valid_ids = duration_by_trajectory[duration_by_trajectory >= min_duration_seconds].index
    filtered = data[data["larva"].isin(valid_ids)].copy()

    # Sorting is important because later scripts assume chronological order.
    filtered = filtered.sort_values(["larva", "time"], kind="mergesort")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    print(f"Saved calibrated trajectory table to: {output_csv}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "outputs" / "00_reorganized_tables" / "merged_tracking_table_unscaled.csv"
    output_csv = project_root / "outputs" / "01_calibrated_tables" / "trajectory_timeseries_calibrated.csv"
    calibrate_and_filter(input_csv, output_csv)


if __name__ == "__main__":
    main()
