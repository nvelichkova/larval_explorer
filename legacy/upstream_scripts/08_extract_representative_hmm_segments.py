#!/usr/bin/env python3
"""08_extract_representative_hmm_segments.py
=================================================
Extract sustained HMM segments from the filtered RDP step table.

A segment is retained if it is a contiguous run of the same filtered state and
contains at least ``MIN_STEPS`` consecutive steps. These segments are later used
as the representative geometric episodes for visualization and state-wise
parameter estimation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ID_COL = "animal ID"
TIME_COL = "tempo_inicial (s)"
STATE_COL = "estado_filtrado"
MIN_STEPS = 9


def extract_segments_for_one_trajectory(group: pd.DataFrame) -> list[dict]:
    """Return contiguous filtered-state segments for one trajectory."""
    times = group[TIME_COL].to_numpy()
    states = group[STATE_COL].to_numpy()
    segments: list[dict] = []

    if len(group) == 0:
        return segments

    start_index = 0
    current_state = states[0]

    for i in range(1, len(group)):
        if states[i] != current_state:
            end_index = i - 1
            length = end_index - start_index + 1
            if length >= MIN_STEPS:
                segments.append({
                    ID_COL: group[ID_COL].iloc[0],
                    "state": current_state,
                    "start_index": int(group.index[start_index]),
                    "end_index": int(group.index[end_index]),
                    "start_time_s": float(times[start_index]),
                    "end_time_s": float(times[end_index]),
                    "n_steps": int(length),
                    "duration_s": float(times[end_index] - times[start_index]),
                })
            start_index = i
            current_state = states[i]

    end_index = len(group) - 1
    length = end_index - start_index + 1
    if length >= MIN_STEPS:
        segments.append({
            ID_COL: group[ID_COL].iloc[0],
            "state": current_state,
            "start_index": int(group.index[start_index]),
            "end_index": int(group.index[end_index]),
            "start_time_s": float(times[start_index]),
            "end_time_s": float(times[end_index]),
            "n_steps": int(length),
            "duration_s": float(times[end_index] - times[start_index]),
        })

    return segments


def extract_representative_segments(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Extract representative filtered-state segments from the RDP step table."""
    steps = pd.read_csv(input_csv)
    for column in (ID_COL, TIME_COL, STATE_COL):
        if column not in steps.columns:
            raise ValueError(f"Required column missing from input table: {column}")

    steps[TIME_COL] = pd.to_numeric(steps[TIME_COL], errors="coerce")
    steps = steps.dropna(subset=[TIME_COL, STATE_COL]).sort_values([ID_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    all_segments = []
    for _, group in steps.groupby(ID_COL, sort=False):
        all_segments.extend(extract_segments_for_one_trajectory(group))

    segments = pd.DataFrame(all_segments)
    if segments.empty:
        segments = pd.DataFrame(columns=[ID_COL, "state", "start_index", "end_index", "start_time_s", "end_time_s", "n_steps", "duration_s"])

    segments = segments.sort_values([ID_COL, "start_time_s"], kind="mergesort").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_csv, index=False)
    print(f"Saved representative HMM segments to: {output_csv}")
    return segments


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "outputs" / "07_hmm_filtered" / "rdp_steps_with_filtered_hmm_states.csv"
    output_csv = project_root / "outputs" / "08_representative_segments" / "representative_hmm_segments.csv"
    extract_representative_segments(input_csv, output_csv)


if __name__ == "__main__":
    main()
