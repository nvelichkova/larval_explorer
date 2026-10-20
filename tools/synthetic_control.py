"""Stage 12 positive control: the synthetic recordings with a known answer.

    python tools/synthetic_control.py <synthetic_dir> <out_dir>

``<synthetic_dir>`` holds ``<condition>.csv``, ``<condition>_truth.csv`` and
``<condition>_meta.json`` per condition (tests/data/synthetic/SYNTHETIC.md).
The synthetic CSV is a long table in whole pixels; it lacks the perimeter,
area and radius channels, so it enters at stage 02 rather than stage 00. Here:
pixels -> mm with the meta scale, time rounded as stage 01 does, one contiguous
track per larva (``<condition>__larva<N>__seg0``). Then stages 03, 04, 09 and
12 run exactly as on real data, and the pooled end-of-turn threshold is taken
across the synthetic conditions, as it would be across recordings.

Two comparisons with the truth, per condition and per heading axis:

- **true boundaries** - stage 12's geometry evaluated at the truth file's onset,
  sweep-end and end frames, and at its own sweep end (max |head angle| in the
  first sweep run) between the true onset and end. This isolates geometry.
- **detected** - stage 09 casts -> stage 12 turns, each truth turn matched to the
  detected turn overlapping it most. This is the end-to-end number.
- **truth-seeded** - stage 12 with the truth file's onset-to-sweep-end intervals
  given as the head casts (stage 09's crawls kept). Stage 12 then finds onset,
  sweep end and end by its own rules: this tests those rules without stage 09.

The signed fraction is primary (D-034). Truth values: ``ideal`` (the model's
12-point geometry, its own axis) and ``clean5`` (5 points, un-quantised).
Writes ``synthetic_turns.csv`` (one row per truth turn with every stage's
values), ``synthetic_summary.csv`` and the stage 12 tables per condition.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larval_explorer.core import schema as S  # noqa: E402
from larval_explorer.core import stages  # noqa: E402
from larval_explorer.core import turns as T  # noqa: E402
from larval_explorer.core.params import (  # noqa: E402
    BodyLengthParams, CalibrationParams, EventParams, SmoothingParams, TurnKinematicsParams,
)

MIN_TOTAL_DEG = 15.0
LARVA_NUMBER = re.compile(r"(\d+)$")
# The synthetic export's own column names.
SYN_LARVA, SYN_FRAME, SYN_TIME = "larva", "frame", "time_s"


def to_trajectories(raw: pd.DataFrame, condition: str, mm_per_px: float) -> pd.DataFrame:
    calibration = CalibrationParams()
    table = pd.DataFrame({
        S.TRACK_ID: [S.segment_track_id(condition, int(LARVA_NUMBER.search(l).group(1)), 0) for l in raw[SYN_LARVA]],
        S.TIME_S: raw[SYN_TIME].round(calibration.time_decimals),
    })
    for channel in S.REQUIRED_CHANNELS:
        if channel in S.POSITION_CHANNELS:
            table[channel] = (raw[channel] * mm_per_px).round(calibration.position_decimals)
        elif channel in raw.columns:
            table[channel] = raw[channel]
        else:
            table[channel] = np.nan  # perimeter, area, radii: not used by stages 03, 04, 09, 12
    table[S.SPINE_LENGTH] = raw[S.SPINE_LENGTH] * mm_per_px
    table[S.INTERPOLATED] = False
    return table


def run_condition(directory: Path, condition: str):
    meta = json.loads((directory / f"{condition}_meta.json").read_text())
    source = directory / f"{condition}.csv"
    if not source.exists():
        source = directory / f"{condition}.csv.gz"  # the repository fixture is compressed
    raw = pd.read_csv(source).sort_values([SYN_LARVA, SYN_FRAME])
    trajectories = to_trajectories(raw, condition, meta["mm_per_px"])
    smoothed = stages.stage_03(trajectories, SmoothingParams()).tables["smoothed"]
    body = stages.stage_04(smoothed, BodyLengthParams()).tables["body_length"]
    events_result = stages.stage_09(smoothed, EventParams())
    truth = pd.read_csv(directory / f"{condition}_truth.csv")
    truth[S.TRACK_ID] = [S.segment_track_id(condition, int(LARVA_NUMBER.search(l).group(1)), 0) for l in truth[SYN_LARVA]]
    return {
        "meta": meta, "trajectories": trajectories, "body": body, "fps": float(meta["fps"]),
        "events": events_result.tables["events"],
        "counts": events_result.tables["thresholds"].set_index(S.TRACK_ID)[S.N_HEAD_CASTS],
        "truth": truth,
    }


def true_boundary_values(kinematics: pd.DataFrame, truth: pd.DataFrame, params: TurnKinematicsParams) -> pd.DataFrame:
    """Stage 12 geometry at the truth file's turn boundaries."""
    rows = []
    by_track = {tid: g.reset_index(drop=True) for tid, g in kinematics.groupby(S.TRACK_ID)}
    for turn in truth.itertuples(index=False):
        k = by_track[turn.track_id]
        i_on, i_sw, i_end = int(turn.true_onset_frame), int(turn.true_sweep_end_frame), int(turn.true_end_frame)
        if i_end >= len(k):
            continue
        head = k[S.HEAD_ANGLE_DEG].to_numpy()
        runs = T._sweep_runs(head[i_on:i_end + 1], params.sweep_head_angle_deg, params.sweep_release_fraction)
        own = i_on + runs[0][0] + int(np.argmax(np.abs(head[i_on + runs[0][0]:i_on + runs[0][1] + 1]))) if runs else None
        row = {S.TRACK_ID: turn.track_id, S.TURN_INDEX: turn.turn_index, "own_sweep_end_frame": own,
               "own_minus_true_sweep_frames": (own - i_sw) if own is not None else np.nan}
        for axis in params.sensitivity_axes:
            h = k[S.heading_column(axis)].to_numpy()
            total = h[i_end] - h[i_on]
            for label, i in (("truesweep", i_sw), ("ownsweep", own)):
                value = (h[i] - h[i_on]) / total if i is not None and abs(total) >= MIN_TOTAL_DEG else np.nan
                row[f"pipeline_{label}_signed_{axis}"] = value
            row[f"pipeline_dpsi_total_{axis}"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def match_detected(turns: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Each truth turn's best-overlapping detected turn (any, included or not)."""
    rows = []
    for turn in truth.itertuples(index=False):
        mine = turns[turns[S.TRACK_ID] == turn.track_id]
        start = mine[S.T_ONSET_S].fillna(mine[S.HEAD_CAST_START_S])
        end = mine[S.T_END_S].fillna(mine[S.HEAD_CAST_END_S])
        overlap = (np.minimum(end, turn.t_end_s) - np.maximum(start, turn.t_onset_s)).clip(lower=0)
        row = {S.TRACK_ID: turn.track_id, S.TURN_INDEX: turn.turn_index, "detected": bool((overlap > 0).any())}
        if row["detected"]:
            best = mine.loc[overlap.idxmax()]
            row.update({
                "detected_turn_index": best[S.TURN_INDEX], "detected_included": bool(best[S.INCLUDED]),
                "detected_exclusion": best[S.EXCLUSION_REASON],
                "detected_onset_err_s": best[S.T_ONSET_S] - turn.t_onset_s,
                "detected_sweep_end_err_s": best[S.T_SWEEP_END_S] - turn.t_sweep_end_s,
                "detected_end_err_s": best[S.T_END_S] - turn.t_end_s,
                "detected_signed_primary": best[S.FRAC_HEADING_IN_SWEEP_SIGNED] if best[S.INCLUDED] else np.nan,
                "detected_signed_model_axis": best[S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED)] if best[S.INCLUDED] else np.nan,
                "detected_dpsi_total_primary": best[S.DPSI_TOTAL_DEG],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def describe(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "mean": v.mean(), "median": v.median(), "q25": v.quantile(0.25), "q75": v.quantile(0.75),
            "sd": v.std(), "share_negative": (v < 0).mean()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("synthetic_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(p.name.removesuffix("_meta.json") for p in args.synthetic_dir.glob("*_meta.json"))
    data = {c: run_condition(args.synthetic_dir, c) for c in conditions}
    base = TurnKinematicsParams()
    bends = []
    for c, d in data.items():
        k, _ = T.prepare_recording(d["trajectories"], d["events"], base, d["fps"])
        bends.append(T.crawl_total_bends(k, d["events"], k[S.ORIENTATION].to_numpy()))
    pooled = float(np.percentile(np.concatenate(bends), base.end_total_bend_crawl_percentile))
    params = dataclasses.replace(base, end_total_bend_deg=pooled)
    print(f"pooled end_total_bend_deg {pooled:.2f}")

    all_rows, summary = [], []
    for c, d in data.items():
        result = T.stage_12(d["trajectories"], d["events"], d["body"], params, d["fps"], d["counts"])
        for name, table in result.tables.items():
            if name != "turn_frames":
                table.to_csv(args.out_dir / f"{c}__{name}.csv", index=False)
        seeded_events = pd.concat([
            d["events"][d["events"][S.EVENT_TYPE] == S.EVENT_CRAWL],
            pd.DataFrame({S.TRACK_ID: d["truth"][S.TRACK_ID], S.EVENT_TYPE: S.EVENT_HEAD_CAST,
                          S.START_S: d["truth"].true_onset_frame / d["fps"],
                          S.END_S: d["truth"].true_sweep_end_frame / d["fps"]}),
        ], ignore_index=True)
        seeded_result = T.stage_12(d["trajectories"], seeded_events, d["body"], params, d["fps"])
        for name in ("turns", "heading_curves", "censoring"):
            seeded_result.tables[name].to_csv(args.out_dir / f"{c}__seeded_{name}.csv", index=False)
        seeded_ok = seeded_result.tables["turns"]
        seeded_ok = seeded_ok[seeded_ok[S.INCLUDED] & seeded_ok[S.FRAC_COMPUTED].astype(bool)][[S.TRACK_ID, S.TURN_INDEX]]
        seeded_curves = seeded_result.tables["heading_curves"].merge(seeded_ok, on=[S.TRACK_ID, S.TURN_INDEX])
        T.curve_summary(seeded_curves, []).to_csv(args.out_dir / f"{c}__seeded_curve_summary.csv", index=False)
        seeded = seeded_result.tables["turns"]
        seeded = seeded.sort_values([S.TRACK_ID, S.HEAD_CAST_START_S]).reset_index(drop=True)
        truth_sorted = d["truth"].sort_values([S.TRACK_ID, "t_onset_s"]).reset_index(drop=True)
        seeded_cols = {
            S.INCLUDED: "seeded_included", S.EXCLUSION_REASON: "seeded_exclusion",
            S.FRAC_HEADING_IN_SWEEP_SIGNED: "seeded_signed_primary",
            S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED): "seeded_signed_model_axis",
        }
        seeded_view = seeded[list(seeded_cols)].rename(columns=seeded_cols)
        seeded_view["seeded_onset_err_s"] = seeded[S.T_ONSET_S] - truth_sorted.t_onset_s
        seeded_view["seeded_sweep_end_err_s"] = seeded[S.T_SWEEP_END_S] - truth_sorted.t_sweep_end_s
        seeded_view["seeded_end_err_s"] = seeded[S.T_END_S] - truth_sorted.t_end_s
        seeded_view[S.TRACK_ID], seeded_view[S.TURN_INDEX] = truth_sorted[S.TRACK_ID], truth_sorted[S.TURN_INDEX]
        seeded_view.loc[~seeded_view.seeded_included.astype(bool), ["seeded_signed_primary", "seeded_signed_model_axis"]] = np.nan
        kinematics, _ = T.prepare_recording(d["trajectories"], d["events"], params, d["fps"])
        truth = d["truth"]
        geometry = true_boundary_values(kinematics, truth, params)
        detected = match_detected(result.tables["turns"], truth)
        merged = truth.merge(geometry, on=[S.TRACK_ID, S.TURN_INDEX], how="left").merge(detected, on=[S.TRACK_ID, S.TURN_INDEX], how="left").merge(seeded_view, on=[S.TRACK_ID, S.TURN_INDEX], how="left")
        merged["ideal_signed"] = np.where(merged.ideal_dpsi_total_deg.abs() >= MIN_TOTAL_DEG,
                                          merged.ideal_dpsi_sweep_deg / merged.ideal_dpsi_total_deg, np.nan)
        for axis in ("tail_sp1", "tail_sp2"):
            merged[f"clean5_signed_{axis}"] = np.where(
                merged[f"clean5_dpsi_total_deg_{axis}"].abs() >= MIN_TOTAL_DEG,
                merged[f"clean5_dpsi_sweep_deg_{axis}"] / merged[f"clean5_dpsi_total_deg_{axis}"], np.nan)
        all_rows.append(merged)

        big = merged[merged.ideal_dpsi_total_deg.abs() >= MIN_TOTAL_DEG]
        stages_compared = {
            "ideal (truth, model axis)": big.ideal_signed,
            "ideal ABS (truth, model axis)": big.ideal_frac,
            "clean5 tail_sp1": big.clean5_signed_tail_sp1,
            "pipeline true boundaries, true sweep end, tail_sp1": big.pipeline_truesweep_signed_tail_sp1,
            "pipeline true boundaries, own sweep end, tail_sp1": big.pipeline_ownsweep_signed_tail_sp1,
            "pipeline truth-seeded turns, tail_sp1": big.seeded_signed_model_axis,
            "pipeline detected turns, tail_sp1": big.detected_signed_model_axis,
            "clean5 tail_sp2": big.clean5_signed_tail_sp2,
            "pipeline true boundaries, own sweep end, tail_sp2": big.pipeline_ownsweep_signed_tail_sp2,
            "pipeline truth-seeded turns, tail_sp2": big.seeded_signed_primary,
            "pipeline detected turns, tail_sp2": big.detected_signed_primary,
        }
        for name, values in stages_compared.items():
            summary.append({"condition": c, "stage": name, **describe(values)})
        det = big.detected.fillna(False).astype(bool)
        print(f"\n== {c}: truth turns {len(truth)}, >= {MIN_TOTAL_DEG:.0f} deg {len(big)}; stage 09 casts {int(d['counts'].sum())}; "
              f"truth turns >=15 deg overlapped by a detected turn {det.sum()} ({100 * det.mean():.0f}%), of which included "
              f"{int(big.detected_included.astype('boolean').fillna(False).astype(bool).sum())}")
        print("   exclusions of matched detected turns:", big.loc[det, "detected_exclusion"].fillna("included").value_counts().to_dict())
        print("   timing error of matched detected turns (s): onset median %+.2f, sweep end median %+.2f (abs median %.2f), end median %+.2f" % (
            big.detected_onset_err_s.median(), big.detected_sweep_end_err_s.median(), big.detected_sweep_end_err_s.abs().median(),
            big.detected_end_err_s.median()))
        print("   own sweep end - true sweep end at true boundaries (frames): median %+.1f, abs median %.1f" % (
            big.own_minus_true_sweep_frames.median(), big.own_minus_true_sweep_frames.abs().median()))
        print("   truth-seeded: exclusions", big.seeded_exclusion.fillna("included").value_counts().to_dict())
        print("   truth-seeded timing error (s): onset median %+.2f, sweep end median %+.2f (abs %.2f), end median %+.2f (abs %.2f)" % (
            big.seeded_onset_err_s.median(), big.seeded_sweep_end_err_s.median(), big.seeded_sweep_end_err_s.abs().median(),
            big.seeded_end_err_s.median(), big.seeded_end_err_s.abs().median()))
        print("   detected turns: %d candidates, %d included with a fraction" % (len(result.tables["turns"]), int((result.tables["turns"][S.INCLUDED] & result.tables["turns"][S.FRAC_COMPUTED].astype(bool)).sum())))

    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(args.out_dir / "synthetic_turns.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out_dir / "synthetic_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print()
        print(summary.to_string(index=False))
        wide = summary.pivot(index="stage", columns="condition", values="mean")
        if len(conditions) == 2:
            wide["difference (" + f"{conditions[0]} - {conditions[1]})"] = wide[conditions[0]] - wide[conditions[1]]
        print("\nmeans by condition (diagnostic only: one synthetic recording per condition):")
        print(wide.loc[list(dict.fromkeys(summary.stage))].to_string())


if __name__ == "__main__":
    main()
