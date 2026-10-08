#!/usr/bin/env python3
"""05_extract_rdp_steps.py
=================================================
Extract a step-based representation of each trajectory using the
Ramer–Douglas–Peucker (RDP) algorithm.

For each larva, epsilon is defined from its mean body length, allowing the
geometric coarse-graining scale to adapt to body size.

Output columns
--------------
- animal ID
- step start/end times
- step start/end positions
- dx, dy
- duration
- speed
- step length
- theta (turning angle between consecutive steps)
- phi (turning-direction persistence versus reversal)
- epsilon_value_mm
- epsilon_factor
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def perpendicular_distance(x: float, y: float, p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Distance from a point to the line defined by p0 and p1."""
    x0, y0 = p0
    x1, y1 = p1
    numerator = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominator = math.hypot(x1 - x0, y1 - y0)
    return numerator / denominator if denominator else math.hypot(x - x0, y - y0)


def rdp(x: list[float], y: list[float], t: list[float], epsilon: float):
    """Recursive Ramer–Douglas–Peucker simplification."""
    if len(x) < 3:
        return x, y, t

    dmax = 0.0
    split_index = 0
    start_point = (x[0], y[0])
    end_point = (x[-1], y[-1])

    for i in range(1, len(x) - 1):
        distance = perpendicular_distance(x[i], y[i], start_point, end_point)
        if distance > dmax:
            dmax = distance
            split_index = i

    if dmax > epsilon:
        left_x, left_y, left_t = rdp(x[: split_index + 1], y[: split_index + 1], t[: split_index + 1], epsilon)
        right_x, right_y, right_t = rdp(x[split_index:], y[split_index:], t[split_index:], epsilon)
        return left_x[:-1] + right_x, left_y[:-1] + right_y, left_t[:-1] + right_t

    return [x[0], x[-1]], [y[0], y[-1]], [t[0], t[-1]]


def epsilon_for_larva(larva: str, trajectory_table_stem: str, body_lengths: pd.DataFrame, epsilon_factor: float) -> float:
    """Return epsilon = epsilon_factor * average_body_length for one larva."""
    row = body_lengths.query("larva == @larva and file_name == @trajectory_table_stem")
    if row.empty:
        raise ValueError(f"Body length not found for larva '{larva}' in file '{trajectory_table_stem}'.")
    return epsilon_factor * float(row["average_body_length"].iloc[0])


def build_steps_and_angles(x: list[float], y: list[float], t: list[float]) -> list[dict]:
    """Convert simplified vertices into step rows with theta and phi."""
    steps = []
    for i in range(1, len(x)):
        x0, y0, t0 = x[i - 1], y[i - 1], t[i - 1]
        x1, y1, t1 = x[i], y[i], t[i]
        dx, dy = x1 - x0, y1 - y0
        dt = t1 - t0
        steps.append({
            "tempo_inicial (s)": t0,
            "tempo_final (s)": t1,
            "X0 (mm)": x0,
            "Y0 (mm)": y0,
            "X (mm)": x1,
            "Y (mm)": y1,
            "dX": dx,
            "dY": dy,
            "duracao (s)": dt,
            "step_length": math.hypot(dx, dy),
            "velocity": math.hypot(dx, dy) / dt if dt else np.nan,
        })

    headings = [math.atan2(step["dY"], step["dX"]) for step in steps]

    for i, step in enumerate(steps):
        if i == 0:
            step["theta"] = np.nan
            step["phi"] = np.nan
            continue

        theta = headings[i] - headings[i - 1]
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        step["theta"] = theta

        previous_sign = np.sign(steps[i - 1]["theta"]) if not np.isnan(steps[i - 1]["theta"]) else 0
        current_sign = np.sign(theta)
        if previous_sign == 0:
            step["phi"] = np.nan
        else:
            step["phi"] = abs(theta) if current_sign == previous_sign else -abs(theta)

    return steps


def extract_rdp_steps(trajectory_csv: Path, body_length_csv: Path, output_csv: Path, epsilon_factor: float = 0.5) -> pd.DataFrame:
    """Run RDP per trajectory and export the resulting step table."""
    trajectories = pd.read_csv(trajectory_csv)
    body_lengths = pd.read_csv(body_length_csv)
    output_rows = []
    table_stem = trajectory_csv.stem

    for larva in trajectories["larva"].unique():
        subset = trajectories[trajectories["larva"] == larva]
        x = subset["mom_x"].tolist()
        y = subset["mom_y"].tolist()
        t = subset["time"].tolist()
        epsilon = epsilon_for_larva(larva, table_stem, body_lengths, epsilon_factor)
        sx, sy, st = rdp(x, y, t, epsilon)
        for step in build_steps_and_angles(sx, sy, st):
            step["animal ID"] = larva
            step["epsilon_value_mm"] = epsilon
            step["epsilon_factor"] = epsilon_factor
            output_rows.append(step)

    columns = [
        "animal ID",
        "tempo_inicial (s)", "tempo_final (s)",
        "X0 (mm)", "Y0 (mm)", "X (mm)", "Y (mm)",
        "dX", "dY", "duracao (s)", "velocity", "step_length",
        "theta", "phi", "epsilon_value_mm", "epsilon_factor",
    ]
    output = pd.DataFrame(output_rows, columns=columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    print(f"Saved RDP step table to: {output_csv}")
    return output


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    trajectory_csv = project_root / "outputs" / "03_smoothed_tables" / "trajectory_timeseries.csv"
    body_length_csv = project_root / "outputs" / "04_body_length" / "mean_body_length_by_trajectory.csv"
    output_csv = project_root / "outputs" / "05_rdp_steps" / "rdp_steps.csv"
    extract_rdp_steps(trajectory_csv, body_length_csv, output_csv, epsilon_factor=0.5)


if __name__ == "__main__":
    main()
