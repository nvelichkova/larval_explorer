"""Stage 12 across recordings: pooled threshold, per-condition summary, sanity checks (T3).

Stage 12's end-of-turn threshold is the 25th percentile of crawling total bend
pooled across every recording in the analysis (D-031), so a stage 12 result
depends on which recordings are analysed together, like the pooled HMM fit
(D-002). It is therefore run here, as an explicit analysis over chosen
recordings, rather than as a per-recording pipeline stage (D-037).

Turns nest in larvae, larvae in recordings, recordings in conditions. Every
summary reports n at all three levels and gives larva-level statistics beside
turn-level ones; only the larva level is a basis for comparing conditions. A
contrast between two conditions is refused when either has a single recording
(D-031): with one recording per condition, condition is confounded with
plate and session.
"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from larval_explorer.core import manifest as M
from larval_explorer.core import pipeline as PL
from larval_explorer.core import schema as S
from larval_explorer.core import turns as T
from larval_explorer.core.aggregate import ALL_RECORDINGS, AnalysisError
from larval_explorer.core.params import TurnKinematicsParams, params_to_dict

TURN_ANALYSIS_FOLDER = "turn_analyses"
TURN_ANALYSIS_MANIFEST = "turn_analysis_manifest.json"
INPUT_STAGES = ("02", "04", "09")

SUBSET_ALL, SUBSET_SINGLE, SUBSET_MULTI = "all", "single_sweep", "multi_sweep"

# Summary measures, primary first (D-034, D-036). Signed measures get a share negative.
SIGNED_MEASURES = (
    S.FRAC_HEADING_IN_SWEEP_SIGNED,
    S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED),
    S.FRAC_SIGNED_AT_HEAD_ANGLE_PEAK,
    S.model_axis_column(S.FRAC_SIGNED_AT_HEAD_ANGLE_PEAK),
)
OTHER_MEASURES = (
    S.BEND_LAG_12_S, S.BEND_LAG_12_SEGLEN, S.BEND_LAG_23_S,
    S.FRAC_HEADING_IN_SWEEP, S.model_axis_column(S.FRAC_HEADING_IN_SWEEP),
    S.ABS_DPSI_TOTAL_DEG, S.ABS_PEAK_HEAD_ANGLE_DEG,
    S.TURN_DURATION_S, S.SWEEP_DURATION_S, S.N_SWEEPS,
)
# Diagnostic only (D-038): real larvae crawl almost nothing while the bend
# migrates, so a rate per segment-length crawled is ill-posed for this animal.
# It is summarised so that its being undefined stays visible.
DIAGNOSTIC_MEASURES = (S.BEND_TRAVEL_PER_SEGLEN, S.BEND_TRAVEL_PER_WAVE)
SUMMARY_MEASURES = SIGNED_MEASURES + OTHER_MEASURES + DIAGNOSTIC_MEASURES


@dataclass(frozen=True)
class RecordingTurnInputs:
    """What stage 12 needs from one recording's pipeline."""

    recording_id: str
    trajectories: pd.DataFrame
    events: pd.DataFrame
    body_length: pd.DataFrame
    head_cast_counts: pd.Series
    frame_rate_fps: float


def load_turn_inputs(pipeline: PL.RecordingPipeline) -> RecordingTurnInputs:
    """Stage 02, 04 and 09 tables of a recording; each must be current."""
    stale = [stage for stage in INPUT_STAGES if pipeline.status(stage) != PL.STATUS_OK]
    if stale:
        raise AnalysisError(f"'{pipeline.recording_id}' needs stages {', '.join(INPUT_STAGES)} current; "
                            f"not current: {', '.join(stale)}.")
    if not pipeline.params.stage_02.enabled:
        raise AnalysisError(f"'{pipeline.recording_id}': turn kinematics needs stage 02 segmentation enabled.")
    thresholds = pipeline.table("09", "thresholds")
    return RecordingTurnInputs(
        recording_id=pipeline.recording_id,
        trajectories=pipeline.table("02", "trajectories"),
        events=pipeline.table("09", "events"),
        body_length=pipeline.table("04", "body_length"),
        head_cast_counts=thresholds.set_index(S.TRACK_ID)[S.N_HEAD_CASTS],
        frame_rate_fps=pipeline.params.stage_01.frame_rate_fps,
    )


@dataclass(frozen=True)
class TurnAnalysis:
    """Everything one turn analysis produced. Tables carry ``recording_id``."""

    params: TurnKinematicsParams          # resolved: end_total_bend_deg is the pooled value
    recording_ids: tuple[str, ...]
    conditions: dict[str, str]            # recording -> condition label
    group_keys: tuple[str, ...]
    end_threshold_per_recording: dict[str, float]
    tables: dict[str, pd.DataFrame]
    diagnostics: dict[str, dict]
    warnings: tuple[str, ...]
    recording_groups: dict = field(default_factory=dict)


def pooled_end_threshold(inputs: Sequence[RecordingTurnInputs], params: TurnKinematicsParams) -> tuple[float, dict[str, float]]:
    """The crawl total-bend percentile pooled over all recordings, and each recording's own value."""
    pooled, own = [], {}
    for item in inputs:
        kinematics, _ = T.prepare_recording(item.trajectories, item.events, params, item.frame_rate_fps)
        bends = T.crawl_total_bends(kinematics, item.events, kinematics[S.ORIENTATION].to_numpy())
        if not len(bends):
            raise AnalysisError(f"'{item.recording_id}' has no clean crawl frames to set the end-of-turn threshold from.")
        own[item.recording_id] = float(np.percentile(bends, params.end_total_bend_crawl_percentile))
        pooled.append(bends)
    return float(np.percentile(np.concatenate(pooled), params.end_total_bend_crawl_percentile)), own


def _keyed(table: pd.DataFrame, recording_id: str, condition: str) -> pd.DataFrame:
    out = table.copy()
    out.insert(0, S.RECORDING_ID, recording_id)
    out.insert(1, S.CONDITION, condition)
    if S.LARVA_INDEX in out.columns:
        out.insert(2, S.LARVA_ID, recording_id + "::larva" + out[S.LARVA_INDEX].astype(int).astype(str))
    return out


def _headline_set(turns: pd.DataFrame) -> pd.DataFrame:
    """Included turns with a fraction: the set every summary is computed on."""
    kept = turns[turns[S.INCLUDED].astype(bool) & turns[S.FRAC_COMPUTED].astype(bool)].copy()
    kept[S.ABS_DPSI_TOTAL_DEG] = kept[S.DPSI_TOTAL_DEG].abs()
    kept[S.ABS_PEAK_HEAD_ANGLE_DEG] = kept[S.PEAK_HEAD_ANGLE_DEG].abs()
    return kept


def _stats(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    n = len(v)
    if not n:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "se": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    return {"n": n, "mean": float(v.mean()), "sd": sd, "se": sd / np.sqrt(n) if n > 1 else np.nan,
            "median": float(v.median()), "q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75))}


def summarise(turns: pd.DataFrame) -> pd.DataFrame:
    """Per condition, subset and measure: n at three levels, turn- and larva-level statistics.

    ``turns`` is the headline set with ``condition``, ``recording_id`` and
    ``larva_id``. Larva-level values are each larva's mean over its turns; their
    mean, SD and SE are across larvae. The share negative is given for signed
    measures, at the turn level and as the mean of larvae's shares.
    """
    subsets = {
        SUBSET_ALL: turns,
        SUBSET_SINGLE: turns[~turns[S.MULTI_SWEEP].astype(bool)],
        SUBSET_MULTI: turns[turns[S.MULTI_SWEEP].astype(bool)],
    }
    rows = []
    for condition in sorted(turns[S.CONDITION].unique()):
        for subset, table in subsets.items():
            mine = table[table[S.CONDITION] == condition]
            for measure in SUMMARY_MEASURES:
                values = mine[[S.RECORDING_ID, S.LARVA_ID, measure]].dropna()
                turn_level = _stats(values[measure])
                larva_means = values.groupby(S.LARVA_ID)[measure].mean()
                larva_level = _stats(larva_means)
                signed = measure in SIGNED_MEASURES
                rows.append({
                    S.CONDITION: condition, S.SUBSET: subset, S.MEASURE: measure,
                    S.N_TURNS: turn_level["n"], S.N_LARVAE: larva_level["n"],
                    S.N_RECORDINGS: int(values[S.RECORDING_ID].nunique()),
                    S.TURN_MEAN: turn_level["mean"], S.TURN_SD: turn_level["sd"], S.TURN_SE: turn_level["se"],
                    S.TURN_MEDIAN: turn_level["median"], S.TURN_Q25: turn_level["q25"], S.TURN_Q75: turn_level["q75"],
                    S.SHARE_NEGATIVE: float((values[measure] < 0).mean()) if signed and len(values) else np.nan,
                    S.LARVA_SHARE_NEGATIVE: float(values.groupby(S.LARVA_ID)[measure].apply(lambda x: (x < 0).mean()).mean())
                    if signed and len(values) else np.nan,
                    S.LARVA_MEAN: larva_level["mean"], S.LARVA_SD: larva_level["sd"], S.LARVA_SE: larva_level["se"],
                })
    return pd.DataFrame(rows)


def check_conditions_differ(summary: pd.DataFrame) -> None:
    """Fail loudly if two conditions have identical summary statistics on every measure.

    Identical n, mean and SD on every measure means the same turns were
    assigned to both conditions, not that two groups happen to agree. A single
    discrete measure (``n_sweeps``) can legitimately coincide, so one match
    alone is not an error.
    """
    whole = summary[summary[S.SUBSET] == SUBSET_ALL]
    profiles = {
        condition: tuple(tuple(np.round(rows[[S.N_TURNS, S.TURN_MEAN, S.TURN_SD]].to_numpy(dtype=float), 12).ravel()))
        for condition, rows in whole.sort_values(S.MEASURE).groupby(S.CONDITION)
    }
    for a, b in itertools.combinations(sorted(profiles), 2):
        if whole.loc[whole[S.CONDITION] == a, S.N_TURNS].max() >= 2 and np.array_equal(
                np.array(profiles[a]), np.array(profiles[b]), equal_nan=True):
            raise AnalysisError(f"Conditions '{a}' and '{b}' have identical statistics on every measure. "
                                "Check that the recordings are assigned to the right conditions.")


def contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    """Pairwise condition contrasts of larva means, or a refusal (D-031).

    A pair is refused, with no numbers, when either condition has fewer than
    two recordings. Otherwise the difference of larva means is given with the
    pooled between-larva SD, and ``within_larva_spread`` marks a difference
    smaller than that SD. No statistical test is computed (D-025).
    """
    whole = summary[summary[S.SUBSET] == SUBSET_ALL].set_index([S.CONDITION, S.MEASURE])
    conditions = sorted(summary[S.CONDITION].unique())
    rows = []
    for a, b in itertools.combinations(conditions, 2):
        for measure in SUMMARY_MEASURES:
            ra, rb = whole.loc[(a, measure)], whole.loc[(b, measure)]
            row = {S.CONDITION_A: a, S.CONDITION_B: b, S.MEASURE: measure,
                   S.N_RECORDINGS_A: int(ra[S.N_RECORDINGS]), S.N_RECORDINGS_B: int(rb[S.N_RECORDINGS])}
            if min(ra[S.N_RECORDINGS], rb[S.N_RECORDINGS]) < 2:
                row[S.CONTRAST_STATUS] = S.CONTRAST_REFUSED_SINGLE_RECORDING
            else:
                pooled = float(np.sqrt((ra[S.LARVA_SD] ** 2 + rb[S.LARVA_SD] ** 2) / 2))
                difference = float(rb[S.LARVA_MEAN] - ra[S.LARVA_MEAN])
                row.update({S.CONTRAST_STATUS: S.CONTRAST_REPORTED, S.DIFFERENCE_OF_LARVA_MEANS: difference,
                            S.POOLED_BETWEEN_LARVA_SD: pooled,
                            S.WITHIN_LARVA_SPREAD: bool(abs(difference) < pooled) if np.isfinite(pooled) else np.nan})
            rows.append(row)
    columns = [S.CONDITION_A, S.CONDITION_B, S.MEASURE, S.N_RECORDINGS_A, S.N_RECORDINGS_B, S.CONTRAST_STATUS,
               S.DIFFERENCE_OF_LARVA_MEANS, S.POOLED_BETWEEN_LARVA_SD, S.WITHIN_LARVA_SPREAD]
    return pd.DataFrame(rows, columns=columns)


def grouping_report(conditions: dict[str, str], recording_groups: dict, turns: pd.DataFrame, group_keys: Sequence[str]) -> pd.DataFrame:
    """Distinct values of every grouping variable, with recordings, larvae and turns behind each."""
    rows = []
    variables = list(group_keys) + [S.CONDITION]
    for variable in variables:
        values = {rid: (conditions[rid] if variable == S.CONDITION else recording_groups.get(rid, {}).get(variable, ""))
                  for rid in conditions}
        for value in sorted(set(values.values())):
            rids = [rid for rid, v in values.items() if v == value]
            mine = turns[turns[S.RECORDING_ID].isin(rids)]
            rows.append({S.VARIABLE: variable, S.VALUE: value, S.N_RECORDINGS: len(rids),
                         S.N_LARVAE: int(mine[S.LARVA_ID].nunique()), S.N_TURNS: len(mine),
                         S.RECORDINGS: ", ".join(sorted(rids))})
    return pd.DataFrame(rows)


def turn_analysis(
    inputs: Sequence[RecordingTurnInputs],
    conditions: dict[str, str] | None = None,
    params: TurnKinematicsParams | None = None,
    group_keys: Sequence[str] = (),
    recording_groups: dict | None = None,
    sensitivity_windows: Sequence[int] = (5, 11),
) -> TurnAnalysis:
    """Run stage 12 on every recording with the pooled threshold, and summarise.

    ``conditions`` maps recording to condition label (``aggregate.condition_labels``);
    without it every recording is one condition. Smoothing sensitivity reruns the
    whole analysis, threshold included, at each window in ``sensitivity_windows``
    and reports the primary signed fraction for each.
    """
    if not inputs:
        raise AnalysisError("No recordings to analyse.")
    recording_ids = tuple(item.recording_id for item in inputs)
    if len(set(recording_ids)) != len(recording_ids):
        raise AnalysisError("A recording appears twice in the analysis.")
    conditions = dict(conditions) if conditions else {rid: ALL_RECORDINGS for rid in recording_ids}
    missing = [rid for rid in recording_ids if rid not in conditions]
    if missing:
        raise AnalysisError(f"No condition for: {', '.join(missing)}")
    base = params or TurnKinematicsParams()
    pooled, own = pooled_end_threshold(inputs, base)
    resolved = dataclasses.replace(base, end_total_bend_deg=pooled)

    collected: dict[str, list[pd.DataFrame]] = {}
    diagnostics, warnings = {}, []
    for item in inputs:
        result = T.stage_12(item.trajectories, item.events, item.body_length, resolved, item.frame_rate_fps,
                            item.head_cast_counts)
        condition = conditions[item.recording_id]
        for name, table in result.tables.items():
            if name == "turn_frames":
                continue
            collected.setdefault(name, []).append(_keyed(table, item.recording_id, condition))
        diagnostics[item.recording_id] = result.diagnostics
        warnings += [f"{item.recording_id}: {message}" for message in result.warnings]
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}

    headline = _headline_set(tables["turns"])
    summary = summarise(headline)
    check_conditions_differ(summary)
    tables["summary"] = summary
    tables["contrasts"] = contrasts(summary)
    groups = recording_groups or {}
    tables["grouping"] = grouping_report(conditions, groups, headline, group_keys)
    curves = tables["heading_curves"].merge(headline[[S.RECORDING_ID, S.TRACK_ID, S.TURN_INDEX]],
                                            on=[S.RECORDING_ID, S.TRACK_ID, S.TURN_INDEX])
    tables["curve_summary"] = T.curve_summary(curves, [S.CONDITION])

    by_condition = pd.Series(conditions).value_counts()
    single = sorted(by_condition[by_condition < 2].index)
    if len(by_condition) > 1 and single:
        warnings.append("Condition contrasts refused: one recording only in " + ", ".join(single)
                        + ". Condition is confounded with plate and session; report per-condition values only.")

    rows = []
    for window in sensitivity_windows:
        alt = dataclasses.replace(base, spine_sg_window=window)
        alt_pooled, _ = pooled_end_threshold(inputs, alt)
        alt = dataclasses.replace(alt, end_total_bend_deg=alt_pooled)
        parts = []
        for item in inputs:
            turns = T.stage_12(item.trajectories, item.events, item.body_length, alt, item.frame_rate_fps).tables["turns"]
            parts.append(_keyed(turns, item.recording_id, conditions[item.recording_id]))
        alt_summary = summarise(_headline_set(pd.concat(parts, ignore_index=True)))
        alt_summary.insert(0, S.SPINE_SG_WINDOW, window)
        alt_summary.insert(1, S.END_TOTAL_BEND_DEG, alt_pooled)
        rows.append(alt_summary)
    primary = summary.copy()
    primary.insert(0, S.SPINE_SG_WINDOW, resolved.spine_sg_window)
    primary.insert(1, S.END_TOTAL_BEND_DEG, pooled)
    sensitivity = pd.concat([primary] + rows, ignore_index=True)
    tables["smoothing_sensitivity"] = sensitivity[(sensitivity[S.SUBSET] == SUBSET_ALL)
                                                  & sensitivity[S.MEASURE].isin(SIGNED_MEASURES[:2])].reset_index(drop=True)

    return TurnAnalysis(resolved, recording_ids, conditions, tuple(group_keys), own, tables, diagnostics,
                        tuple(warnings), groups)


def export_turn_analysis(folder: Path, analysis: TurnAnalysis) -> Path:
    """Write every table as CSV, plus a manifest with the resolved parameters. Returns the manifest path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in analysis.tables.items():
        path = folder / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path.name)
    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "recordings": list(analysis.recording_ids),
        "conditions": analysis.conditions,
        "grouped_by": list(analysis.group_keys),
        "recording_groups": analysis.recording_groups,
        "params": params_to_dict(analysis.params),
        "end_total_bend_deg_pooled": analysis.params.end_total_bend_deg,
        "end_total_bend_deg_per_recording": analysis.end_threshold_per_recording,
        "primary": {
            "fraction": S.FRAC_HEADING_IN_SWEEP_SIGNED, "heading_axis": analysis.params.heading_axis,
            "model_matched_axis": analysis.params.model_heading_axis,
            "cast_end": "peak of the sp1 bend (D-036)", "migration": S.BEND_LAG_12_S,
            "diagnostic_only": list(DIAGNOSTIC_MEASURES),
            "summary_level_for_comparison": "larva",
        },
        "bend_sign_convention": "walking tail to head; same sign as the head angle; the model walks head to tail (D-032)",
        "diagnostics": analysis.diagnostics,
        "warnings": list(analysis.warnings),
        "files": files,
        "environment": M.environment(),
    }
    path = folder / TURN_ANALYSIS_MANIFEST
    path.write_text(json.dumps(M.to_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def new_turn_analysis_folder(output_root: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / TURN_ANALYSIS_FOLDER / f"turns_{stamp}"
