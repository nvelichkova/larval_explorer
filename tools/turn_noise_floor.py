"""Stage 12 phase T1: measure the spine-angle noise floor on real recordings.

    python tools/turn_noise_floor.py <output_root> <recording.csv> [<recording.csv> ...] [--windows 5 7 11]

Each recording is run through stage 09 with default parameters (resuming
whatever is already current in ``output_root``). Per-frame turn geometry is
then computed from stage 02's tracks at each spine smoothing window, and the
noise floor is measured on clean frames inside stage 09 crawls. Writes
``turn_noise_floor_by_track.csv`` and ``turn_noise_floor.csv`` (one row per
recording, window and quantity) into ``output_root`` and prints the summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larval_explorer.core import pipeline as PL  # noqa: E402
from larval_explorer.core import schema as S  # noqa: E402
from larval_explorer.core import turns as T  # noqa: E402
from larval_explorer.core.params import TurnKinematicsParams  # noqa: E402


def frame_quality(kinematics: pd.DataFrame, events: pd.DataFrame, fps: float) -> dict:
    """Tracking-failure counts and distance rates inside and outside stage 09 events."""
    clean = T.clean_frames(kinematics)
    in_crawl = np.zeros(len(kinematics), dtype=bool)
    in_event = np.zeros(len(kinematics), dtype=bool)
    for track_id, rows in kinematics.groupby(S.TRACK_ID).groups.items():
        times = kinematics.loc[rows, S.TIME_S].to_numpy()
        mine = events[events[S.TRACK_ID] == track_id]
        crawls = mine[mine[S.EVENT_TYPE] == S.EVENT_CRAWL][[S.START_S, S.END_S]].to_numpy()
        positions = kinematics.index.get_indexer(rows)
        in_crawl[positions] = T._in_events(times, crawls)
        in_event[positions] = T._in_events(times, mine[[S.START_S, S.END_S]].to_numpy())

    forward = np.maximum(kinematics[S.TAIL_FORWARD_SPEED_MM_S].to_numpy(), 0)
    path = kinematics.groupby(S.TRACK_ID)[S.TAIL_PATH_LENGTH_MM].diff().fillna(0).to_numpy() * fps

    def rate(values, mask):
        return float(np.mean(values[mask & clean.to_numpy()])) if (mask & clean.to_numpy()).any() else np.nan

    return {
        "frames": len(kinematics),
        "spine_flips": int(kinematics[S.SPINE_FLIP].sum()),
        "not_well_oriented_frames": int((kinematics[S.IS_WELL_ORIENTED] != 1).sum()),
        "interpolated_frames": int(kinematics[S.INTERPOLATED].astype(bool).sum()),
        "forward_mm_s_in_crawls": rate(forward, in_crawl),
        "path_mm_s_in_crawls": rate(path, in_crawl),
        "forward_mm_s_outside_events": rate(forward, ~in_event),
        "path_mm_s_outside_events": rate(path, ~in_event),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--windows", nargs="*", type=int, default=[5, 7, 11])
    args = parser.parse_args()

    params = TurnKinematicsParams()
    by_track, summary, quality = [], [], []
    for source in args.recordings:
        recording_id = source.stem
        pipeline = PL.RecordingPipeline(recording_id, source, args.output_root)
        pipeline.run(["09"])
        fps = pipeline.params.stage_01.frame_rate_fps
        trajectories = pipeline.table("02", "trajectories")
        events = pipeline.table("09", "events")
        for window in args.windows:
            kinematics = T.frame_kinematics(trajectories, params, fps, sg_window=window)
            floor = T.noise_floor(kinematics, events)
            floor.insert(0, S.RECORDING_ID, recording_id)
            floor.insert(1, "spine_sg_window", window)
            by_track.append(floor)
            for quantity, rows in floor.groupby(S.QUANTITY, sort=False):
                weights = rows[S.N_FRAMES].to_numpy()
                summary.append({
                    S.RECORDING_ID: recording_id,
                    "spine_sg_window": window,
                    S.QUANTITY: quantity,
                    "tracks": len(rows),
                    S.N_FRAMES: int(weights.sum()),
                    "sd_median_over_tracks": float(rows[S.NOISE_SD].median()),
                    "sd_min": float(rows[S.NOISE_SD].min()),
                    "sd_max": float(rows[S.NOISE_SD].max()),
                    "median_abs_median_over_tracks": float(rows[S.MEDIAN_ABS].median()),
                    "p95_abs_median_over_tracks": float(rows[S.P95_ABS].median()),
                })
            quality.append({S.RECORDING_ID: recording_id, "spine_sg_window": window, "tracks": kinematics[S.TRACK_ID].nunique(),
                            **frame_quality(kinematics, events, fps)})

    pd.concat(by_track, ignore_index=True).to_csv(args.output_root / "turn_noise_floor_by_track.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_root / "turn_noise_floor.csv", index=False)
    quality = pd.DataFrame(quality)
    quality.to_csv(args.output_root / "turn_frame_quality.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 2):
        print(summary.drop(columns=["sd_min", "sd_max"]).to_string(index=False))
        print()
        print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
