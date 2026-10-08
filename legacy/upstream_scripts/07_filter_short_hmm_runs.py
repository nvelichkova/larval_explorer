#!/usr/bin/env python3
"""07_filter_short_hmm_runs.py
=================================================
Apply a one-pass temporal filter to remove short isolated HMM state islands.

Filtering rule
--------------
Consecutive runs shorter than ``MIN_RUN`` are replaced by the state of the
longer neighboring run, evaluated in the original (unfiltered) sequence. In a
tie, the left neighbor is chosen. This reproduces the exact logic used in the
analysis scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ID_COL = "animal ID"
TIME_COL = "tempo_inicial (s)"
STATE_COL = "estado"
MIN_RUN = 3


def run_length_encode(states: list[int]) -> list[dict]:
    """Return run-length encoding for a discrete state sequence."""
    runs = []
    if len(states) == 0:
        return runs

    start = 0
    current_state = states[0]
    for i in range(1, len(states)):
        if states[i] != current_state:
            runs.append({"state": current_state, "start": start, "end": i, "length": i - start})
            start = i
            current_state = states[i]
    runs.append({"state": current_state, "start": start, "end": len(states), "length": len(states) - start})
    return runs


def filter_state_sequence_one_pass(states: list[int], min_run: int = MIN_RUN) -> list[int]:
    """Replace short state islands using the longer neighboring run."""
    runs = run_length_encode(list(states))
    original_lengths = [run["length"] for run in runs]
    filtered = list(states)

    for i, run in enumerate(runs):
        if original_lengths[i] >= min_run:
            continue

        has_left = i > 0
        has_right = i < len(runs) - 1

        if not has_left and not has_right:
            continue
        if has_left and not has_right:
            chosen_state = runs[i - 1]["state"]
        elif has_right and not has_left:
            chosen_state = runs[i + 1]["state"]
        else:
            left_length = original_lengths[i - 1]
            right_length = original_lengths[i + 1]
            chosen_state = runs[i + 1]["state"] if right_length > left_length else runs[i - 1]["state"]

        filtered[run["start"]: run["end"]] = [chosen_state] * (run["end"] - run["start"])

    return filtered


def filter_hmm_runs(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Add a filtered state column to the HMM-labelled RDP step table."""
    steps = pd.read_csv(input_csv)
    for column in (ID_COL, TIME_COL, STATE_COL):
        if column not in steps.columns:
            raise ValueError(f"Required column missing from input table: {column}")

    steps = steps.sort_values([ID_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    filtered_states = []
    for _, group in steps.groupby(ID_COL, sort=False):
        filtered_states.extend(filter_state_sequence_one_pass(group[STATE_COL].tolist(), MIN_RUN))

    steps["estado_filtrado"] = filtered_states
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    steps.to_csv(output_csv, index=False)
    print(f"Saved filtered HMM state table to: {output_csv}")
    return steps


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "outputs" / "06_hmm" / "rdp_steps_with_hmm_states.csv"
    output_csv = project_root / "outputs" / "07_hmm_filtered" / "rdp_steps_with_filtered_hmm_states.csv"
    filter_hmm_runs(input_csv, output_csv)


if __name__ == "__main__":
    main()
