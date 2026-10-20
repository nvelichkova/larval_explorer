"""Stage 12 phase T2: turns on real recordings, and the hand-check dumps.

    python tools/turns_t2.py <output_root> <recording.csv> [<recording.csv> ...] [--turns-per-track 5]

Not yet a pipeline stage (that is T3): this runs each recording through
stage 09, computes the pooled end-of-turn bend threshold across all of them
(D-031), runs ``turns.stage_12`` with the relative speed rule and with the
absolute variant, and writes into ``<output_root>/stage12_t2/``:

- ``<recording>__turns.csv``, ``__censoring.csv``, ``__axis_sensitivity.csv``,
  ``__label_pieces.csv``, ``__larva_reference.csv`` (relative speed rule)
- ``<recording>__turns_absolute_speed.csv`` (the sensitivity variant)
- ``handcheck/`` one track per recording chosen by the agreed criteria, with
  ``<track>__frames.csv`` (raw and smoothed spine points plus every derived
  quantity for each turn window) and ``<track>__turn_<k>.txt`` (the turn's
  numbers and how each is obtained from the frames file)
- ``t2_summary.json``
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larval_explorer.core import pipeline as PL  # noqa: E402
from larval_explorer.core import schema as S  # noqa: E402
from larval_explorer.core import turns as T  # noqa: E402
from larval_explorer.core.params import TurnKinematicsParams  # noqa: E402

MIN_HANDCHECK_TURNS = 5
HANDCHECK_DPSI_DEG = 30.0


def pooled_end_threshold(recordings, params, fps_by_recording):
    bends, per_recording = [], {}
    for recording_id, (trajectories, events) in recordings.items():
        kinematics, _ = T.prepare_recording(trajectories, events, params, fps_by_recording[recording_id])
        values = T.crawl_total_bends(kinematics, events, kinematics[S.ORIENTATION].to_numpy())
        per_recording[recording_id] = float(np.percentile(values, params.end_total_bend_crawl_percentile))
        bends.append(values)
    pooled = float(np.percentile(np.concatenate(bends), params.end_total_bend_crawl_percentile))
    return pooled, per_recording


def choose_handcheck_track(turns, trajectories, kinematics):
    """Longest track with no label jump, no bridged frame and >= 5 included turns with |dpsi_total| > 30 deg."""
    big = turns[turns[S.INCLUDED] & (turns[S.DPSI_TOTAL_DEG].abs() > HANDCHECK_DPSI_DEG)]
    counts = big.groupby(S.TRACK_ID).size()
    rows = []
    for track_id, track in kinematics.groupby(S.TRACK_ID):
        rows.append({S.TRACK_ID: track_id, "frames": len(track), "label_jumps": int(track[S.SPINE_FLIP].sum()),
                     "bridged": int(track[S.INTERPOLATED].astype(bool).sum()), "big_turns": int(counts.get(track_id, 0))})
    table = pd.DataFrame(rows).sort_values("frames", ascending=False)
    eligible = table[(table.label_jumps == 0) & (table.bridged == 0) & (table.big_turns >= MIN_HANDCHECK_TURNS)]
    return (eligible.iloc[0][S.TRACK_ID] if len(eligible) else None), table


def handcheck(out_dir, recording_id, track_id, trajectories, kinematics, turns, params, fps, n_turns):
    raw = trajectories[trajectories[S.TRACK_ID] == track_id].sort_values(S.TIME_S).reset_index(drop=True)
    kin = kinematics[kinematics[S.TRACK_ID] == track_id].reset_index(drop=True)
    smoothed = T.smooth(T.spine_points(raw), params.spine_sg_window, params.spine_sg_order)
    mom = T.smooth(T.centre_of_mass(raw), params.spine_sg_window, params.spine_sg_order)
    frames = kin.copy()
    for k, (x, y) in enumerate(T.SPINE_POINTS):
        frames.insert(2 + 4 * k, f"raw_{x}", raw[x].to_numpy())
        frames.insert(3 + 4 * k, f"raw_{y}", raw[y].to_numpy())
        frames.insert(4 + 4 * k, f"smooth_{x}", smoothed[:, k, 0])
        frames.insert(5 + 4 * k, f"smooth_{y}", smoothed[:, k, 1])
    frames["smooth_mom_x"], frames["smooth_mom_y"] = mom[:, 0], mom[:, 1]

    mine = turns[(turns[S.TRACK_ID] == track_id) & turns[S.INCLUDED] & (turns[S.DPSI_TOTAL_DEG].abs() > HANDCHECK_DPSI_DEG)]
    chosen = mine.head(n_turns)
    pad = params.frame_padding_s
    keep = np.zeros(len(frames), dtype=bool)
    for turn in chosen.itertuples(index=False):
        keep |= (frames[S.TIME_S] >= turn.t_onset_s - pad).to_numpy() & (frames[S.TIME_S] <= turn.t_end_s + pad).to_numpy()
    safe = track_id.replace(":", "_")
    frames[keep].to_csv(out_dir / f"{safe}__frames.csv", index=False)

    h, hm = S.heading_column(params.heading_axis), S.heading_column(params.model_heading_axis)
    for turn in chosen.itertuples(index=False):
        t = turn._asdict()
        at = {name: frames.loc[np.isclose(frames[S.TIME_S], t[col])].iloc[0]
              for name, col in (("onset", S.T_ONSET_S), ("sweep_end", S.T_SWEEP_END_S), ("max_bend", S.T_MAX_BEND_S), ("end", S.T_END_S))}
        lines = [
            f"recording {recording_id}  track {track_id}  turn {t[S.TURN_INDEX]}",
            f"stage 09 head cast {t[S.HEAD_CAST_START_S]:.1f}-{t[S.HEAD_CAST_END_S]:.1f} s",
            f"frames file: {safe}__frames.csv (one row per frame; raw_* = stage 02 mm, smooth_* = Savitzky-Golay {params.spine_sg_window}/{params.spine_sg_order})",
            "",
            "KEY FRAMES",
        ]
        for name, f in at.items():
            pts = "  ".join(f"({f[f'smooth_{x}']:.3f},{f[f'smooth_{y}']:.3f})" for x, y in T.SPINE_POINTS)
            lines.append(f"  {name:9s} t={f[S.TIME_S]:7.1f}  head_angle={f[S.HEAD_ANGLE_DEG]:+7.2f}  bends="
                         f"{f[S.bend_column(1)]:+6.1f} {f[S.bend_column(2)]:+6.1f} {f[S.bend_column(3)]:+6.1f}  total={f[S.TOTAL_BEND_DEG]:6.1f}"
                         f"  psi[{params.heading_axis}]={f[h]:+8.2f}  psi[{params.model_heading_axis}]={f[hm]:+8.2f}"
                         f"  dist={f[S.POSTERIOR_DISTANCE_MM]:.3f}  v={f[S.TAIL_FORWARD_SPEED_MM_S]:+.3f}")
            lines.append(f"            smoothed head,sp1,sp2,sp3,tail: {pts}")
        o, s_, m, e = at["onset"], at["sweep_end"], at["max_bend"], at["end"]
        lines += [
            "",
            "HOW EACH NUMBER IS OBTAINED (all angles from the smooth_* columns)",
            f"  onset     = last frame at/before cast start ({t[S.HEAD_CAST_START_S]:.1f}) with |head_angle| < {params.onset_head_angle_deg}"
            f"  (lookback {params.max_onset_lookback_s} s)  -> {o[S.TIME_S]:.1f}",
            f"  sweep_end = frame of max |head_angle| in the first run of |head_angle| >= {params.sweep_head_angle_deg}, one sign  -> {s_[S.TIME_S]:.1f}",
            f"  end       = first frame after sweep_end starting {params.end_hold_frames} frames with total_bend < {params.end_total_bend_deg:.2f}"
            f" and v > {t[S.END_SPEED_THRESHOLD_MM_S]:.3f} mm/s ({params.end_speed_fraction_of_crawl} x larva crawl speed {t[S.LARVA_CRAWL_SPEED_MM_S]:.3f})  -> {e[S.TIME_S]:.1f}",
            f"  max_bend  = frame of max total_bend in [onset, end]  -> {m[S.TIME_S]:.1f}",
            "",
            f"  head_angle = angle from (tail->sp2) to (sp2->head) = atan2(cross, dot)",
            f"  bend_i     = angle from (p[i-1]->p[i]) to (p[i]->p[i+1]), i = 1..3 at sp1..sp3",
            f"  psi[axis]  = atan2 of (front - tail), unwrapped along the track",
            "",
            f"  dpsi_sweep [{params.heading_axis}] = {s_[h]:+.2f} - {o[h]:+.2f} = {t[S.DPSI_SWEEP_DEG]:+.2f}",
            f"  dpsi_total [{params.heading_axis}] = {e[h]:+.2f} - {o[h]:+.2f} = {t[S.DPSI_TOTAL_DEG]:+.2f}",
            f"  frac_heading_in_sweep        = |{t[S.DPSI_SWEEP_DEG]:.2f}| / |{t[S.DPSI_TOTAL_DEG]:.2f}| = {t[S.FRAC_HEADING_IN_SWEEP]:.3f}"
            f"   (signed {t[S.FRAC_HEADING_IN_SWEEP_SIGNED]:+.3f})",
            f"  model axis [{params.model_heading_axis}]: dpsi_sweep {s_[hm] - o[hm]:+.2f}, dpsi_total {e[hm] - o[hm]:+.2f},"
            f" frac {t[S.model_axis_column(S.FRAC_HEADING_IN_SWEEP)]:.3f}",
            "",
            f"  at max bend: peak boundary {t[S.at_max_column(S.BEND_PEAK_BOUNDARY)]:.3f} (edge={t[S.at_max_column(S.BEND_PEAK_AT_EDGE)]}),"
            f" arc {t[S.at_max_column(S.BEND_PEAK_ARC_FRACTION)]:.3f}, model boundary {t[S.at_max_column(S.BEND_PEAK_MODEL_BOUNDARY)]:.2f}",
            f"               trail boundary {t[S.at_max_column(S.BEND_TRAIL_BOUNDARY)]:.3f} (censored={t[S.at_max_column(S.BEND_TRAIL_CENSORED)]}),"
            f" arc {t[S.at_max_column(S.BEND_TRAIL_ARC_FRACTION)]:.3f}, model boundary {t[S.at_max_column(S.BEND_TRAIL_MODEL_BOUNDARY)]:.2f}",
            f"  peak: parabola vertex through |bend_1..3| when the largest is bend_2; trail: posterior-most crossing of"
            f" {params.trail_fraction_of_peak} x max|bend|",
            f"  arc fraction: from boundary_*_arc_fraction columns; model boundary = {params.n_body_segments} x arc",
            "",
            f"  bend travel: frames in [max_bend, end] with total_bend >= {params.travel_min_total_bend_deg}: {t[S.BEND_TRAVEL_N_FRAMES]},"
            f" distance spanned {t[S.BEND_TRAVEL_DISTANCE_SEGLEN]:.3f} seglen (seglen {t[S.SEGMENT_LENGTH_MM]:.4f} mm)",
            f"     slope of bend_peak_model_boundary vs (dist - dist[max_bend]) / seglen = {t[S.BEND_TRAVEL_PER_SEGLEN]}"
            f"  (r2 {t[S.BEND_TRAVEL_R2]}; per wave {t[S.BEND_TRAVEL_PER_WAVE]}, stride {t[S.STRIDE_SEGLEN]:.2f} seglen)",
            f"  n_sweeps {t[S.N_SWEEPS]}, duration {t[S.TURN_DURATION_S]:.1f} s, peak head angle {t[S.PEAK_HEAD_ANGLE_DEG]:+.1f}",
        ]
        (out_dir / f"{safe}__turn_{t[S.TURN_INDEX]}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--turns-per-track", type=int, default=5)
    args = parser.parse_args()
    out = args.output_root / "stage12_t2"
    (out / "handcheck").mkdir(parents=True, exist_ok=True)

    base = TurnKinematicsParams()
    data, fps, body, counts = {}, {}, {}, {}
    for source in args.recordings:
        pipeline = PL.RecordingPipeline(source.stem, source, args.output_root)
        pipeline.run(["09"])
        fps[source.stem] = pipeline.params.stage_01.frame_rate_fps
        data[source.stem] = (pipeline.table("02", "trajectories"), pipeline.table("09", "events"))
        body[source.stem] = pipeline.table("04", "body_length")
        counts[source.stem] = pipeline.table("09", "thresholds").set_index(S.TRACK_ID)[S.N_HEAD_CASTS]

    pooled, per_recording = pooled_end_threshold(data, base, fps)
    params = dataclasses.replace(base, end_total_bend_deg=pooled)
    summary = {"end_total_bend_deg_pooled": pooled, "end_total_bend_deg_per_recording": per_recording,
               "percentile": base.end_total_bend_crawl_percentile, "recordings": {}}
    print(f"end_total_bend_deg: pooled {pooled:.2f}; per recording " + ", ".join(f"{k} {v:.2f}" for k, v in per_recording.items()))

    for recording_id, (trajectories, events) in data.items():
        result = T.stage_12(trajectories, events, body[recording_id], params, fps[recording_id], counts[recording_id])
        absolute = T.stage_12(trajectories, events, body[recording_id],
                              dataclasses.replace(params, end_speed_mode="absolute"), fps[recording_id], counts[recording_id])
        for name, table in result.tables.items():
            if name != "turn_frames":
                table.to_csv(out / f"{recording_id}__{name}.csv", index=False)
        absolute.tables["turns"].to_csv(out / f"{recording_id}__turns_absolute_speed.csv", index=False)
        uncorrected = T.stage_12(trajectories, events, body[recording_id],
                                 dataclasses.replace(params, correct_swapped_labels=False), fps[recording_id], counts[recording_id])
        uncorrected.tables["turns"].to_csv(out / f"{recording_id}__turns_without_label_correction.csv", index=False)
        uncorrected.tables["censoring"].to_csv(out / f"{recording_id}__censoring_without_label_correction.csv", index=False)
        kept = result.tables["turns"]
        kept = kept[kept[S.INCLUDED] & kept[S.FRAC_COMPUTED].astype(bool)][[S.TRACK_ID, S.TURN_INDEX]]
        T.curve_summary(result.tables["heading_curves"].merge(kept, on=[S.TRACK_ID, S.TURN_INDEX]), []).to_csv(
            out / f"{recording_id}__curve_summary.csv", index=False)

        turns = result.tables["turns"]
        kinematics, _ = T.prepare_recording(trajectories, events, params, fps[recording_id])
        track_id, eligibility = choose_handcheck_track(turns, trajectories, kinematics)
        eligibility.to_csv(out / "handcheck" / f"{recording_id}__track_eligibility.csv", index=False)
        chosen = None
        if track_id is not None:
            chosen = handcheck(out / "handcheck", recording_id, track_id, trajectories, kinematics, turns, params,
                               fps[recording_id], args.turns_per_track)

        def distribution(table, column):
            kept = table[table[S.INCLUDED] & table[S.FRAC_COMPUTED].astype(bool)]
            values = kept[column]
            if not len(values):
                return None
            q = values.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            return {"n_turns": len(values), "n_larvae": int(kept[S.LARVA_INDEX].nunique()),
                    "q10": q[0.1], "q25": q[0.25], "median": q[0.5], "q75": q[0.75], "q90": q[0.9],
                    "mean": float(values.mean()), "sd": float(values.std()), "share_negative": float((values < 0).mean()),
                    "median_abs_dpsi_total_deg": float(kept[S.DPSI_TOTAL_DEG].abs().median())}

        signed, signed_model = S.FRAC_HEADING_IN_SWEEP_SIGNED, S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED)
        summary["recordings"][recording_id] = {
            **result.diagnostics, "warnings": list(result.warnings),
            "handcheck_track": track_id,
            "handcheck_turns": [] if chosen is None else chosen[S.TURN_INDEX].tolist(),
            "signed_frac_primary_axis": distribution(turns, signed),
            "signed_frac_model_axis": distribution(turns, signed_model),
            "abs_frac_primary_axis": distribution(turns, S.FRAC_HEADING_IN_SWEEP),
            "signed_frac_primary_axis_absolute_speed": distribution(absolute.tables["turns"], signed),
            "signed_frac_primary_axis_without_label_correction": distribution(uncorrected.tables["turns"], signed),
            "signed_frac_model_axis_without_label_correction": distribution(uncorrected.tables["turns"], signed_model),
        }
        with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
            print(f"\n== {recording_id}")
            print(result.tables["censoring"].to_string(index=False))
            print("larva reference:")
            print(result.tables["larva_reference"].to_string(index=False))
            print("hand-check eligibility (longest first):")
            print(eligibility.head(8).to_string(index=False))
            print("chosen:", track_id)
    (out / "t2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
