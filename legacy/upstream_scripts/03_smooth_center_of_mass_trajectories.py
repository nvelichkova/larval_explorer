#!/usr/bin/env python3
"""03_smooth_center_of_mass_trajectories.py
=================================================
Smooth the centre-of-mass coordinates of each trajectory with a Savitzky–Golay
filter.

Only the centre-of-mass coordinates are smoothed here because they are the
coordinates later used for geometric trajectory analyses (RDP, step extraction,
MSD, and related quantities). Landmark coordinates are left unchanged in this
script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.signal import savgol_filter


def smooth_trajectory_timeseries(input_csv: Path, output_csv: Path, window_size: int = 41, polynomial_order: int = 3) -> None:
    """Apply per-trajectory Savitzky–Golay smoothing to x/y centre-of-mass coordinates."""
    if window_size % 2 == 0:
        window_size += 1  # Savitzky–Golay requires an odd window length.

    data = pd.read_csv(input_csv)
    data = data.sort_values(["larva", "time"], kind="mergesort")

    columns_to_smooth = ["mom_x", "mom_y"]
    data[columns_to_smooth] = data.groupby("larva")[columns_to_smooth].transform(
        lambda series: savgol_filter(series, window_length=window_size, polyorder=polynomial_order)
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)
    print(f"Saved smoothed trajectory table to: {output_csv}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "outputs" / "01_calibrated_tables" / "trajectory_timeseries_calibrated.csv"
    output_csv = project_root / "outputs" / "03_smoothed_tables" / "trajectory_timeseries.csv"
    smooth_trajectory_timeseries(input_csv, output_csv)


if __name__ == "__main__":
    main()
