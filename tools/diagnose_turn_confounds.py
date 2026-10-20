"""Stage 12 pre-T2 diagnostics. Read-only: changes no stage and no parameter.

    python tools/diagnose_turn_confounds.py <output_root> <recording.csv> [<recording.csv> ...]

1. Noise floor per larva, compared between recordings. Is the crawling spread
   of the spine angles itself different between conditions? If so, every
   threshold-based measure differs for reasons unrelated to turning.
2. ``is_well_oriented`` exposure. How much does stage 11's head-cast rate move
   if casts containing a not-well-oriented frame are left out (D-028)?

Each recording is run through stage 11 with default parameters (resuming what
is current). Writes ``confound_floor_by_larva.csv``,
``confound_floor_comparison.csv``, ``confound_cast_rate_by_segment.csv`` and
``confound_cast_rate_summary.csv`` into ``output_root``.
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

FLOOR_MEASURES = {
    "bend_1_sd": (S.bend_column(1), np.std),
    "bend_2_sd": (S.bend_column(2), np.std),
    "bend_3_sd": (S.bend_column(3), np.std),
    "head_angle_sd": (S.HEAD_ANGLE_DEG, np.std),
    "head_angle_abs_p95": (S.HEAD_ANGLE_DEG, lambda v: np.percentile(np.abs(v), 95)),
    "total_bend_p25": (S.TOTAL_BEND_DEG, lambda v: np.percentile(v, 25)),
    "total_bend_median": (S.TOTAL_BEND_DEG, np.median),
    "total_bend_p95": (S.TOTAL_BEND_DEG, lambda v: np.percentile(v, 95)),
}


def interval_mask(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    return T._in_events(times, intervals)


def floor_by_larva(recording_id, kinematics, events, body_length, fps):
    kinematics = kinematics.copy()
    kinematics[S.LARVA_INDEX] = kinematics[S.TRACK_ID].map(S.larva_index)
    clean = T.clean_frames(kinematics).to_numpy()
    in_crawl = np.zeros(len(kinematics), dtype=bool)
    for track_id, rows in kinematics.groupby(S.TRACK_ID).groups.items():
        crawls = events[(events[S.TRACK_ID] == track_id) & (events[S.EVENT_TYPE] == S.EVENT_CRAWL)]
        positions = kinematics.index.get_indexer(rows)
        in_crawl[positions] = interval_mask(kinematics.loc[rows, S.TIME_S].to_numpy(), crawls[[S.START_S, S.END_S]].to_numpy())
    body = body_length.set_index(S.TRACK_ID)[S.BODY_LENGTH_MEAN]

    rows = []
    for larva, frames in kinematics.groupby(S.LARVA_INDEX):
        mask = (clean & in_crawl)[kinematics.index.get_indexer(frames.index)]
        crawl = frames[mask]
        row = {S.RECORDING_ID: recording_id, S.LARVA_INDEX: larva, "tracks": frames[S.TRACK_ID].nunique(),
               "minutes_tracked": len(frames) / fps / 60, "clean_crawl_frames": len(crawl)}
        for name, (column, stat) in FLOOR_MEASURES.items():
            row[name] = float(stat(crawl[column].to_numpy())) if len(crawl) else np.nan
        row["not_well_oriented_fraction"] = float((frames[S.IS_WELL_ORIENTED] != 1).mean())
        row["flips_per_min"] = float(frames[S.SPINE_FLIP].sum() / (len(frames) / fps / 60))
        row["crawl_forward_speed_mm_s"] = float(np.mean(crawl[S.TAIL_FORWARD_SPEED_MM_S].clip(lower=0))) if len(crawl) else np.nan
        # Frame-weighted mean of the segments' mean body lengths.
        lengths = frames.groupby(S.TRACK_ID).size()
        row["body_length_mm"] = float(np.average(body.loc[lengths.index], weights=lengths.to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_floor(by_larva: pd.DataFrame) -> pd.DataFrame:
    """Per measure: median and IQR over larvae per recording, and the gap in units of between-larva SD."""
    measures = [c for c in by_larva.columns if c not in (S.RECORDING_ID, S.LARVA_INDEX, "tracks", "clean_crawl_frames")]
    recordings = list(dict.fromkeys(by_larva[S.RECORDING_ID]))
    rows = []
    for measure in measures:
        row = {"measure": measure}
        medians = []
        for rec in recordings:
            values = by_larva.loc[by_larva[S.RECORDING_ID] == rec, measure].dropna()
            q1, q2, q3 = np.percentile(values, [25, 50, 75])
            row[f"{rec}_n_larvae"] = len(values)
            row[f"{rec}_median"] = q2
            row[f"{rec}_iqr"] = f"{q1:.3g}-{q3:.3g}"
            row[f"{rec}_sd"] = float(values.std(ddof=1))
            medians.append(q2)
        if len(recordings) == 2:
            sds = [row[f"{rec}_sd"] for rec in recordings]
            pooled = float(np.sqrt(np.mean(np.square(sds))))
            row["difference_of_medians"] = medians[1] - medians[0]
            row["difference_in_between_larva_sd"] = (medians[1] - medians[0]) / pooled if pooled > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cast_rates(recording_id, pipeline, fps):
    """Stage 11's per-segment head-cast rate, recounted with and without flagged casts."""
    smoothed = pipeline.table("03", "smoothed")
    steps = pipeline.table("10", "event_level_steps")
    metrics = pipeline.table("11", "headcast_segment_metrics")
    events = pipeline.table("09", "events")
    casts = steps[steps[S.EVENT_TYPE].astype(str).str.lower() == S.EVENT_HEAD_CAST].copy()

    flagged = np.zeros(len(casts), dtype=bool)
    for i, (track_id, start, end) in enumerate(casts[[S.TRACK_ID, S.T_START_S, S.T_END_S]].itertuples(index=False)):
        frames = smoothed[(smoothed[S.TRACK_ID] == track_id) & (smoothed[S.TIME_S] >= start) & (smoothed[S.TIME_S] <= end)]
        flagged[i] = bool((frames[S.IS_WELL_ORIENTED] != 1).any())
    casts["flagged"] = flagged

    rows = []
    for seg in metrics.itertuples(index=False):
        seg = seg._asdict()
        inside = casts[(casts[S.TRACK_ID] == seg[S.TRACK_ID]) & (casts[S.T_START_S] >= seg[S.T0_SEG]) & (casts[S.T_START_S] <= seg[S.TF_SEG])]
        n_clean = int((~inside["flagged"]).sum())
        rows.append({S.RECORDING_ID: recording_id, S.SEGMENT_ID: seg[S.SEGMENT_ID], S.TRACK_ID: seg[S.TRACK_ID], S.STATE: seg[S.STATE],
                     S.SEG_DURATION_S: seg[S.SEG_DURATION_S], "n_headcasts_stage11": seg[S.N_HEADCASTS],
                     "n_headcasts_recount": len(inside), "n_headcasts_flagged": len(inside) - n_clean,
                     "hc_rate_hz_stage11": seg[S.HC_RATE_HZ], "hc_rate_hz_without_flagged": n_clean / seg[S.SEG_DURATION_S]})
    by_segment = pd.DataFrame(rows)

    # Phase 5's measure: head casts per minute tracked, per larva, from stage 09's events.
    hc = events[events[S.EVENT_TYPE] == S.EVENT_HEAD_CAST].copy()
    hc_flag = np.zeros(len(hc), dtype=bool)
    for i, (track_id, start, end) in enumerate(hc[[S.TRACK_ID, S.START_S, S.END_S]].itertuples(index=False)):
        frames = smoothed[(smoothed[S.TRACK_ID] == track_id) & (smoothed[S.TIME_S] >= start) & (smoothed[S.TIME_S] <= end)]
        hc_flag[i] = bool((frames[S.IS_WELL_ORIENTED] != 1).any())
    hc["flagged"] = hc_flag
    hc[S.LARVA_INDEX] = hc[S.TRACK_ID].map(S.larva_index)
    minutes = smoothed.groupby(smoothed[S.TRACK_ID].map(S.larva_index)).size() / fps / 60
    per_larva = pd.DataFrame({
        "minutes": minutes,
        "casts": hc.groupby(S.LARVA_INDEX).size(),
        "casts_flagged": hc.groupby(S.LARVA_INDEX)["flagged"].sum(),
    }).fillna(0)
    per_larva["rate_per_min"] = per_larva["casts"] / per_larva["minutes"]
    per_larva["rate_per_min_without_flagged"] = (per_larva["casts"] - per_larva["casts_flagged"]) / per_larva["minutes"]
    per_larva.insert(0, S.RECORDING_ID, recording_id)
    return by_segment, per_larva.reset_index(names=S.LARVA_INDEX), int(hc_flag.sum()), len(hc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recordings", nargs="+", type=Path)
    args = parser.parse_args()

    params = TurnKinematicsParams()
    floors, segments, larvae, cast_lines = [], [], [], []
    for source in args.recordings:
        recording_id = source.stem
        pipeline = PL.RecordingPipeline(recording_id, source, args.output_root)
        pipeline.run(["11"])
        fps = pipeline.params.stage_01.frame_rate_fps
        kinematics = T.frame_kinematics(pipeline.table("02", "trajectories"), params, fps)
        floors.append(floor_by_larva(recording_id, kinematics, pipeline.table("09", "events"),
                                     pipeline.table("04", "body_length"), fps))
        by_segment, per_larva, n_flagged, n_casts = cast_rates(recording_id, pipeline, fps)
        segments.append(by_segment)
        larvae.append(per_larva)
        cast_lines.append(f"{recording_id}: {n_flagged} of {n_casts} stage 09 head casts ({100 * n_flagged / n_casts:.1f}%) contain a not-well-oriented frame")

    floor = pd.concat(floors, ignore_index=True)
    comparison = compare_floor(floor)
    segments = pd.concat(segments, ignore_index=True)
    larvae = pd.concat(larvae, ignore_index=True)
    floor.to_csv(args.output_root / "confound_floor_by_larva.csv", index=False)
    comparison.to_csv(args.output_root / "confound_floor_comparison.csv", index=False)
    segments.to_csv(args.output_root / "confound_cast_rate_by_segment.csv", index=False)
    larvae.to_csv(args.output_root / "confound_cast_rate_by_larva.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print("== 1. noise floor per larva ==")
        print(floor.to_string(index=False))
        print()
        print(comparison.to_string(index=False))
        print()
        print("== 2. is_well_oriented exposure ==")
        print("\n".join(cast_lines))
        assert (segments["n_headcasts_recount"] == segments["n_headcasts_stage11"]).all(), "recount does not reproduce stage 11"
        print("recount reproduces stage 11's n_headcasts in every HMM segment")
        seg_summary = segments.groupby(S.RECORDING_ID).apply(lambda g: pd.Series({
            "hmm_segments": len(g),
            "segments_with_flagged_cast": int((g["n_headcasts_flagged"] > 0).sum()),
            "casts": int(g["n_headcasts_recount"].sum()),
            "casts_flagged": int(g["n_headcasts_flagged"].sum()),
            "median_rate_hz": g["hc_rate_hz_stage11"].median(),
            "median_rate_hz_without": g["hc_rate_hz_without_flagged"].median(),
            "mean_rate_hz": g["hc_rate_hz_stage11"].mean(),
            "mean_rate_hz_without": g["hc_rate_hz_without_flagged"].mean(),
            "max_relative_drop": float(((g["hc_rate_hz_stage11"] - g["hc_rate_hz_without_flagged"]) / g["hc_rate_hz_stage11"]).max()),
        }), include_groups=False)
        print(seg_summary.to_string())
        seg_summary.to_csv(args.output_root / "confound_cast_rate_summary.csv")
        print()
        print(larvae.to_string(index=False))


if __name__ == "__main__":
    main()
