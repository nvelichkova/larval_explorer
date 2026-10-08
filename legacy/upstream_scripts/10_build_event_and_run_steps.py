#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_build_event_and_run_steps.py
==========================

Construct TWO step-series from a trajectory table and an events table.

Inputs
------
Trajectory CSV (per-time positions). Must contain:
  - larva id column: one of  ["larva","larva_id","id","track","track_id"]
  - time column:     one of  ["time","t","timestamp","tempo","tempo_s"]
  - position (mm):   prefer ("mom_x","mom_y"), else ("com_x","com_y"), else ("x","y")

Events CSV (intervals). Must contain:
  - "ID"   (larva id; must match trajectory)
  - "type" (e.g., "crawl", "head_cast")
  - "start","end" (seconds)

Outputs (in current folder)
--------------------------
1) run_anchor_steps.csv
   Steps between anchors: start of track, each head-cast midpoint (spatial midpoint
   of positions at head-cast start and end), and end of track.

2) event_level_steps.csv
   One row per event ("crawl" or "head_cast"), with:
     - tipo: "crawl" | "head_cast"
     - cast_theta (radians) for head_cast rows: Δφ between the immediate
       crawl before and the immediate crawl after (sign indicates left/right).

Both outputs include columns compatible with your RDP steps:
  'animal ID','tempo_inicial (s)','tempo_final (s)',
  'X0 (mm)','Y0 (mm)','X (mm)','Y (mm)',
  'dX','dY','duracao (s)','velocity','step_length','theta','phi',
  'epsilon_value_mm','epsilon_factor'
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------- Helpers -----------------
def angle_wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def find_first_existing(candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = Path(name)
        if p.exists():
            return p
    return None


def detect_columns_traj(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    # larva id
    id_candidates = ["larva", "larva_id", "id", "track", "track_id"]
    id_col = next((c for c in df.columns if c.lower() in id_candidates), None)
    if id_col is None:
        raise ValueError(
            "Could not find larva id column in trajectory "
            f"(tried: {id_candidates})"
        )
    # time
    time_candidates = ["time", "t", "timestamp", "tempo", "tempo_s"]
    time_col = next((c for c in df.columns if c.lower() in time_candidates), None)
    if time_col is None:
        raise ValueError(
            "Could not find time column in trajectory "
            f"(tried: {time_candidates})"
        )
    # position (prefer mom_x,mom_y → com_x,com_y → x,y)
    pos_pairs = [("mom_x", "mom_y"), ("com_x", "com_y"), ("x", "y"), ("X", "Y")]
    cols_lower = {c.lower(): c for c in df.columns}
    x_col = y_col = None
    for a, b in pos_pairs:
        if a in cols_lower and b in cols_lower:
            x_col, y_col = cols_lower[a], cols_lower[b]
            break
    if x_col is None:
        raise ValueError(
            "Could not find position columns in trajectory "
            "(looked for mom/com/x,y)."
        )
    return id_col, time_col, x_col, y_col


def interp_pos(
    times: np.ndarray, xs: np.ndarray, ys: np.ndarray, tq: np.ndarray | float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation of position at query time(s) tq; clips to [min,max] to avoid NaNs.
    ALWAYS returns 1D numpy arrays (even for a single query).
    """
    tq_arr = np.atleast_1d(np.asarray(tq, float))
    tmin, tmax = float(times[0]), float(times[-1])
    tqc = np.clip(tq_arr, tmin, tmax)
    xq = np.interp(tqc, times, xs)
    yq = np.interp(tqc, times, ys)
    return xq, yq


def build_step_rows(rows: List[Dict], compute_theta: bool = True) -> pd.DataFrame:
    """
    Assemble DataFrame and compute dX,dY,duration,velocity,phi,theta.
    Input rows must include:
      'animal ID','tempo_inicial (s)','tempo_final (s)',
      'X0 (mm)','Y0 (mm)','X (mm)','Y (mm)'.
    """
    if not rows:
        return pd.DataFrame(
            columns=[
                "animal ID",
                "tempo_inicial (s)",
                "tempo_final (s)",
                "X0 (mm)",
                "Y0 (mm)",
                "X (mm)",
                "Y (mm)",
                "dX",
                "dY",
                "duracao (s)",
                "velocity",
                "step_length",
                "theta",
                "phi",
                "epsilon_value_mm",
                "epsilon_factor",
            ]
        )

    df = pd.DataFrame(rows)

    # core kinematics
    df["dX"] = df["X (mm)"] - df["X0 (mm)"]
    df["dY"] = df["Y (mm)"] - df["Y0 (mm)"]
    df["duracao (s)"] = df["tempo_final (s)"] - df["tempo_inicial (s)"]
    df["step_length"] = np.hypot(df["dX"], df["dY"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["velocity"] = df["step_length"] / df["duracao (s)"]
    df["phi"] = np.arctan2(df["dY"], df["dX"])

    # theta (Δφ) within larva id
    df["theta"] = np.nan
    if compute_theta and len(df):
        for lid, idx in df.groupby("animal ID").groups.items():
            idx = list(idx)
            ph = df.loc[idx, "phi"].to_numpy()
            dphi = angle_wrap_pi(np.diff(ph, prepend=np.nan))
            df.loc[idx, "theta"] = dphi

    # placeholders to match RDP schema
    df["epsilon_value_mm"] = pd.NA
    df["epsilon_factor"] = pd.NA

    # order columns like RDP
    base_cols = [
        "animal ID",
        "tempo_inicial (s)",
        "tempo_final (s)",
        "X0 (mm)",
        "Y0 (mm)",
        "X (mm)",
        "Y (mm)",
        "dX",
        "dY",
        "duracao (s)",
        "velocity",
        "step_length",
        "theta",
        "phi",
    ]
    ordered = base_cols + ["epsilon_value_mm", "epsilon_factor"] + [
        c for c in df.columns if c not in base_cols + ["epsilon_value_mm", "epsilon_factor"]
    ]
    return df[ordered].reset_index(drop=True)


# ----------------- Main -----------------
def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build two step-series from trajectory + events."
    )
    parser.add_argument(
        "--traj",
        type=str,
        default=None,
        help="Trajectory CSV (default: data/trajectory_timeseries.csv)",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Event CSV (default: data/event_intervals.csv)",
    )
    parser.add_argument(
        "--out1",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "10_event_and_run_steps" / "run_anchor_steps.csv"),
        help="Output CSV for model 1",
    )
    parser.add_argument(
        "--out2",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "10_event_and_run_steps" / "event_level_steps.csv"),
        help="Output CSV for model 2",
    )
    args = parser.parse_args()

    # locate inputs
    traj_path = Path(args.traj) if args.traj else find_first_existing([str(PROJECT_ROOT / "data" / "trajectory_timeseries.csv")])
    ev_path = Path(args.events) if args.events else find_first_existing([str(PROJECT_ROOT / "data" / "event_intervals.csv")])

    if traj_path is None or not traj_path.exists():
        raise SystemExit("Trajectory file not found. Provide --traj or place data/trajectory_timeseries.csv in the repository.")
    if ev_path is None or not ev_path.exists():
        raise SystemExit("Event file not found. Provide --events or place data/event_intervals.csv in the repository.")

    traj = pd.read_csv(traj_path)
    events = pd.read_csv(ev_path)

    # validate columns
    id_col, time_col, x_col, y_col = detect_columns_traj(traj)
    for col in ["ID", "type", "start", "end"]:
        if col not in events.columns:
            raise SystemExit(f"Events file missing required column: {col}")

    # sanitize and sort
    traj = traj.sort_values([id_col, time_col]).reset_index(drop=True)
    events = events.sort_values(["ID", "start", "end"]).reset_index(drop=True)

    # build per-larva arrays
    per_larva: Dict[object, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lid, g in traj.groupby(id_col, sort=False):
        t = g[time_col].to_numpy(float)
        x = g[x_col].to_numpy(float)
        y = g[y_col].to_numpy(float)
        order = np.argsort(t, kind="mergesort")
        per_larva[lid] = (t[order], x[order], y[order])

    # -------- Model 1: anchors (head_cast midpoints + start/end) --------
    rows1: List[Dict] = []
    for lid, (tarr, xarr, yarr) in per_larva.items():
        # Start anchor
        anchors_t: List[float] = [float(tarr[0])]
        anchors_x: List[float] = [float(xarr[0])]
        anchors_y: List[float] = [float(yarr[0])]

        # Head-cast anchors
        lev = events[events["ID"] == lid]
        hc = lev[lev["type"].astype(str).str.lower() == "head_cast"]
        if not hc.empty:
            starts = hc["start"].to_numpy(float)
            ends = hc["end"].to_numpy(float)

            xs0, ys0 = interp_pos(tarr, xarr, yarr, starts)  # arrays
            xs1, ys1 = interp_pos(tarr, xarr, yarr, ends)    # arrays

            xmid = (xs0 + xs1) / 2.0
            ymid = (ys0 + ys1) / 2.0
            tm = (starts + ends) / 2.0

            # Extend anchors
            anchors_t.extend(tm.tolist())
            anchors_x.extend(xmid.tolist())
            anchors_y.extend(ymid.tolist())

        # End anchor
        anchors_t.append(float(tarr[-1]))
        anchors_x.append(float(xarr[-1]))
        anchors_y.append(float(yarr[-1]))

        # Sort anchors by time and connect successive anchors
        order = np.argsort(anchors_t, kind="mergesort")
        at = [anchors_t[i] for i in order]
        ax = [anchors_x[i] for i in order]
        ay = [anchors_y[i] for i in order]

        for i in range(len(at) - 1):
            ti, tj = at[i], at[i + 1]
            xi, yi = ax[i], ay[i]
            xj, yj = ax[i + 1], ay[i + 1]
            if tj > ti:
                rows1.append(
                    {
                        "animal ID": lid,
                        "tempo_inicial (s)": ti,
                        "tempo_final (s)": tj,
                        "X0 (mm)": xi,
                        "Y0 (mm)": yi,
                        "X (mm)": xj,
                        "Y (mm)": yj,
                    }
                )

    df1 = build_step_rows(rows1, compute_theta=True)
    df1.to_csv(args.out1, index=False)

    # -------- Model 2: event-level steps with tipo + cast_theta --------
    rows2: List[Dict] = []
    for lid, (tarr, xarr, yarr) in per_larva.items():
        lev = events[events["ID"] == lid]
        if lev.empty:
            continue
        for _, r in lev.iterrows():
            t_start = float(r["start"])
            t_end = float(r["end"])
            x0_arr, y0_arr = interp_pos(tarr, xarr, yarr, np.array([t_start]))
            x1_arr, y1_arr = interp_pos(tarr, xarr, yarr, np.array([t_end]))
            rows2.append(
                {
                    "animal ID": lid,
                    "tempo_inicial (s)": t_start,
                    "tempo_final (s)": t_end,
                    "X0 (mm)": float(x0_arr[0]),
                    "Y0 (mm)": float(y0_arr[0]),
                    "X (mm)": float(x1_arr[0]),
                    "Y (mm)": float(y1_arr[0]),
                    "tipo": str(r["type"]).lower(),
                }
            )

    df2 = build_step_rows(rows2, compute_theta=True)

    # cast_theta on head_cast rows: Δφ between adjacent *crawl* steps
    df2["cast_theta"] = np.nan
    for lid, idx in df2.groupby("animal ID").groups.items():
        sub = df2.loc[idx].sort_values("tempo_inicial (s)").reset_index()
        # indices in the sub-DataFrame (0..n-1), map back to df2 via "index" column
        for i in range(len(sub)):
            if sub.loc[i, "tipo"] != "head_cast":
                continue
            # find previous crawl φ
            prev_phi = math.nan
            for j in range(i - 1, -1, -1):
                if sub.loc[j, "tipo"] == "crawl":
                    prev_phi = float(sub.loc[j, "phi"])
                    break
            # find next crawl φ
            next_phi = math.nan
            for j in range(i + 1, len(sub)):
                if sub.loc[j, "tipo"] == "crawl":
                    next_phi = float(sub.loc[j, "phi"])
                    break
            cast = math.nan
            if math.isfinite(prev_phi) and math.isfinite(next_phi):
                cast = float(angle_wrap_pi(next_phi - prev_phi))
            df2.loc[sub.loc[i, "index"], "cast_theta"] = cast

    df2.to_csv(args.out2, index=False)


if __name__ == "__main__":
    main()
