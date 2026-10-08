#!/usr/bin/env python3
"""04_compute_mean_body_length.py
=================================================
Compute the mean and standard deviation of body length for each trajectory.

Body length is defined as the piecewise-linear arc length of the tracked midline:
head -> spine point 1 -> spine point 2 -> spine point 3 -> tail.
The resulting file is used by the RDP step-extraction script to set the
simplification tolerance epsilon relative to the body scale of each larva.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def euclidean_distance(x1, y1, x2, y2):
    """Vectorized Euclidean distance between two 2D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_mean_body_length(input_csv: Path, output_csv: Path) -> None:
    """Compute per-trajectory mean and standard deviation of body length."""
    data = pd.read_csv(input_csv)

    data["body_length"] = (
        euclidean_distance(data["head_x"], data["head_y"], data["spinepoint_1_x"], data["spinepoint_1_y"]) +
        euclidean_distance(data["spinepoint_1_x"], data["spinepoint_1_y"], data["spinepoint_2_x"], data["spinepoint_2_y"]) +
        euclidean_distance(data["spinepoint_2_x"], data["spinepoint_2_y"], data["spinepoint_3_x"], data["spinepoint_3_y"]) +
        euclidean_distance(data["spinepoint_3_x"], data["spinepoint_3_y"], data["tail_x"], data["tail_y"])
    )

    stats = data.groupby("larva")["body_length"].agg(["mean", "std"]).reset_index()
    stats.columns = ["larva", "average_body_length", "std_dev_body_length"]
    stats["file_name"] = input_csv.stem

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_csv, index=False)
    print(f"Saved body-length summary to: {output_csv}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "outputs" / "03_smoothed_tables" / "trajectory_timeseries.csv"
    output_csv = project_root / "outputs" / "04_body_length" / "mean_body_length_by_trajectory.csv"
    compute_mean_body_length(input_csv, output_csv)


if __name__ == "__main__":
    main()
