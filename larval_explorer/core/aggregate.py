"""Compare recordings across experimental conditions (Phase 5).

Two things happen here, both across recordings and both explicit:

1. **The pooled HMM fit** (D-002). The RDP steps of every recording in an
   analysis are fitted together, so that "state 0" means the same thing in
   every recording. Each recording then gets its share of that fit as its
   stage 06, and its own stages 07-11 are re-run on it.

2. **Aggregation.** Per-recording results are summarised per unit - a track
   segment, a larva or a recording - and the units are grouped by whichever
   registry metadata the user picks (D-003).

Observations are nested: track segment within larva within recording within
condition. Several segments of one larva are not independent, so the unit is
chosen explicitly (larva by default) and n is reported at all three levels.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from larval_explorer.core import manifest as M
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.core import schema as S
from larval_explorer.core import stages
from larval_explorer.core.params import HmmParams, PipelineParams, params_to_dict
from larval_explorer.core.project import Project
from larval_explorer.core.result import StageResult

UNIT_SEGMENT = "track segment"
UNIT_LARVA = "larva"
UNIT_RECORDING = "recording"
UNITS = (UNIT_SEGMENT, UNIT_LARVA, UNIT_RECORDING)
DEFAULT_UNIT = UNIT_LARVA

MEASURE_OCCUPANCY = "state_occupancy"
MEASURE_DWELL = "dwell_time_s"
MEASURE_HEAD_CAST_RATE = "head_cast_rate_per_min"
MEASURE_CRAWL_LENGTH = "crawl_length_mm"
STEERING_MEASURES = (S.RHO, S.KAPPA, S.MU, S.SIGMA)

N_STATES = 3
ALL_RECORDINGS = "all recordings"
POOLED_FIT = "pooled"
COMPARISON_FOLDER = "comparisons"
FIGURE_MANIFEST = "figure_manifest.json"


class AnalysisError(ValueError):
    """The analysis cannot be done as asked; the message says what to change."""


# ──────────────────────────────────────────────────────────────────────────────
# Pooled HMM fit
# ──────────────────────────────────────────────────────────────────────────────

def fit_pooled_hmm(steps_by_recording: dict[str, pd.DataFrame], params: HmmParams):
    """Fit one HMM on the RDP steps of several recordings.

    Sequences are keyed on (recording, track): track IDs repeat between
    recordings when segmentation is off, and keying on the track alone would
    silently join two animals into one sequence (D-015).

    Returns ``({recording_id: steps with states}, diagnostics)``.
    """
    ordered = sorted(steps_by_recording)
    pooled = pd.concat(
        [steps_by_recording[recording_id].assign(**{S.RECORDING_ID: recording_id}) for recording_id in ordered],
        ignore_index=True,
    )
    pooled = pooled.sort_values([S.RECORDING_ID, S.TRACK_ID], kind="mergesort").reset_index(drop=True)
    result = stages.stage_06(pooled, params, sequence_keys=(S.RECORDING_ID, S.TRACK_ID))
    steps = result.tables["steps"]
    shares = {
        recording_id: steps[steps[S.RECORDING_ID] == recording_id].drop(columns=S.RECORDING_ID).reset_index(drop=True)
        for recording_id in ordered
    }
    diagnostics = dict(result.diagnostics)
    diagnostics["pooled_over_recordings"] = len(ordered)
    return shares, diagnostics, result.warnings


def pool_fingerprint(pipelines: dict[str, PL.RecordingPipeline], params: HmmParams) -> str:
    """Hash of everything a pooled fit depends on: who is in it, their steps, the HMM settings."""
    payload = {
        "recordings": {recording_id: pipelines[recording_id].fingerprint("05") for recording_id in sorted(pipelines)},
        "params": params_to_dict(params),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class PooledOutcome:
    pool: PL.HmmPool
    refitted: bool
    state_mean_phi: dict
    failures: dict[str, str] = field(default_factory=dict)   # recording -> message, for stages after the fit
    warnings: list[str] = field(default_factory=list)


def run_pooled_analysis(
    project: Project,
    recording_ids: Sequence[str],
    params: PipelineParams | None = None,
    progress: Callable[[str, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> PooledOutcome:
    """Fit the HMM pooled over the given recordings and bring each one up to date on it.

    Runs stages 00-05 where needed, fits once, installs each recording's share
    as its stage 06, then runs 07-11. If the pool and its inputs are unchanged
    since the last pooled fit, nothing is refitted. ``progress(recording, text)``
    reports what is happening.
    """
    recording_ids = list(recording_ids)
    if len(recording_ids) < 1:
        raise AnalysisError("Tick at least one recording.")

    def tell(recording_id: str, text: str) -> None:
        if progress is not None:
            progress(recording_id, text)

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise PL.PipelineCancelled("The pooled analysis was cancelled.")

    pipelines: dict[str, PL.RecordingPipeline] = {}
    for recording_id in recording_ids:
        check_cancel()
        tell(recording_id, "stages 00-05")
        pipeline = project.pipeline(recording_id, params)
        try:
            pipeline.run(["05"], should_cancel=should_cancel)
        except PL.PipelineError as error:
            raise AnalysisError(
                f"'{recording_id}' failed before the HMM ({error}). Fix it or untick it; a pooled fit needs every recording."
            ) from error
        pipelines[recording_id] = pipeline

    hmm_settings = {recording_id: pipeline.params.stage_06 for recording_id, pipeline in pipelines.items()}
    hmm_params = hmm_settings[recording_ids[0]]
    different = [recording_id for recording_id, value in hmm_settings.items() if value != hmm_params]
    if different:
        raise AnalysisError(f"HMM parameters differ between recordings ({', '.join(different)}); a pooled fit needs one setting.")

    pool = PL.HmmPool(tuple(sorted(recording_ids)), pool_fingerprint(pipelines, hmm_params))
    current = all(p.pool == pool and p.status("06") == PL.STATUS_OK for p in pipelines.values())
    warnings: list[str] = []
    if not current:
        check_cancel()
        tell(POOLED_FIT, f"fitting the HMM on {len(recording_ids)} recordings")
        steps = {recording_id: pipeline.table("05", "steps") for recording_id, pipeline in pipelines.items()}
        try:
            shares, diagnostics, fit_warnings = fit_pooled_hmm(steps, hmm_params)
        except ValueError as error:
            raise AnalysisError(f"The pooled HMM could not be fitted: {error}") from error
        warnings.extend(fit_warnings)
        for recording_id, pipeline in pipelines.items():
            pipeline.install_pooled_hmm(pool, StageResult({"steps": shares[recording_id]}, diagnostics, fit_warnings))

    failures: dict[str, str] = {}
    for recording_id, pipeline in pipelines.items():
        check_cancel()
        tell(recording_id, "stages 07-11")
        try:
            pipeline.run(["11"], should_cancel=should_cancel)
        except PL.PipelineError as error:
            failures[recording_id] = str(error)
    first = pipelines[recording_ids[0]]
    return PooledOutcome(pool, not current, dict(first.manifest.get("hmm_state_mean_phi", {})), failures, warnings)


def per_condition_hmm(
    steps_by_recording: dict[str, pd.DataFrame], conditions: dict[str, str], params: HmmParams
) -> pd.DataFrame:
    """Diagnostic: fit the HMM separately in each condition, and pooled, and compare the states.

    One row per fit and state: the fitted mean phi, the number of steps
    decoded into the state, and whether the fit converged. If a condition's
    states sit far from the pooled ones, the pooled description fits it badly.
    Nothing is written to any recording.
    """
    groups = {POOLED_FIT: sorted(steps_by_recording)}
    for condition in sorted(set(conditions.values())):
        groups[condition] = sorted(r for r in steps_by_recording if conditions[r] == condition)
    rows = []
    for name, members in groups.items():
        try:
            _, diagnostics, _ = fit_pooled_hmm({r: steps_by_recording[r] for r in members}, params)
        except ValueError:
            for state in range(N_STATES):
                rows.append({S.FIT: name, S.STATE: state, S.MEAN_PHI: np.nan, S.N_STEPS: 0,
                             S.CONVERGED: False, S.N_RECORDINGS: len(members)})
            continue
        for state in range(N_STATES):
            rows.append({
                S.FIT: name, S.STATE: state,
                S.MEAN_PHI: diagnostics["state_mean_phi"][str(state)],
                S.N_STEPS: diagnostics["state_counts"][str(state)],
                S.CONVERGED: diagnostics["converged"],
                S.N_RECORDINGS: len(members),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-recording tables
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RecordingData:
    """What aggregation reads from one processed recording."""

    recording_id: str
    tracks: pd.DataFrame          # track_id, larva_index, tracked_s
    steps: pd.DataFrame           # stage 07: filtered HMM states per RDP step
    events: pd.DataFrame          # stage 09
    event_steps: pd.DataFrame     # stage 10, event level
    segment_fits: pd.DataFrame    # stage 11
    hmm_pool: tuple[str, ...] = ()
    hmm_pool_fingerprint: str | None = None


def load_recording_data(pipeline: PL.RecordingPipeline) -> RecordingData:
    """Read one recording's results. Every stage must be current."""
    not_current = [stage for stage, status in pipeline.statuses().items() if status != PL.STATUS_OK]
    if not_current:
        raise AnalysisError(
            f"'{pipeline.recording_id}' is not fully processed (stages {', '.join(not_current)}). "
            "Run the pooled fit, or a batch, first."
        )
    if "track_segments.csv" in pipeline.manifest["stages"]["02"]["outputs"]:
        segments = pipeline.table("02", "track_segments")
        kept = segments[segments[S.KEPT].astype(bool)]
        tracks = kept[[S.TRACK_ID, S.LARVA_INDEX]].assign(**{S.TRACKED_S: kept[S.DURATION_S]})
    else:
        # Segmentation off: a track is a larva, tracked from its first to its last frame.
        spans = pipeline.table("03", "smoothed").groupby(S.TRACK_ID)[S.TIME_S].agg(["min", "max"])
        tracks = pd.DataFrame({
            S.TRACK_ID: spans.index,
            S.LARVA_INDEX: [S.larva_index(track_id) for track_id in spans.index],
            S.TRACKED_S: (spans["max"] - spans["min"]).to_numpy(),
        })
    return RecordingData(
        recording_id=pipeline.recording_id,
        tracks=tracks.reset_index(drop=True),
        steps=pipeline.table("07", "steps"),
        events=pipeline.table("09", "events"),
        event_steps=pipeline.table("10", "event_level_steps"),
        segment_fits=pipeline.table("11", "segment_fits"),
        hmm_pool=tuple(pipeline.manifest.get("hmm_pool", [])),
        hmm_pool_fingerprint=pipeline.manifest.get("hmm_pool_fingerprint"),
    )


def _keyed(table: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    return table.assign(**{S.RECORDING_ID: recording_id})


def track_table(data: Sequence[RecordingData]) -> pd.DataFrame:
    """Every track of every recording, with the larva it belongs to and how long it was tracked."""
    frames = []
    for recording in data:
        tracks = _keyed(recording.tracks, recording.recording_id)
        tracks[S.LARVA_ID] = recording.recording_id + "::larva" + tracks[S.LARVA_INDEX].astype(int).astype(str)
        frames.append(tracks)
    return pd.concat(frames, ignore_index=True)


def state_seconds(data: Sequence[RecordingData]) -> pd.DataFrame:
    """Time spent in each filtered state, per track: the summed duration of its RDP steps."""
    frames = []
    for recording in data:
        steps = recording.steps
        if not len(steps):
            continue
        seconds = steps.groupby([S.TRACK_ID, S.STATE_FILTERED])[S.DURATION_S].sum().reset_index()
        seconds.columns = [S.TRACK_ID, S.STATE, S.SECONDS]
        frames.append(_keyed(seconds, recording.recording_id))
    columns = [S.TRACK_ID, S.STATE, S.SECONDS, S.RECORDING_ID]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def dwell_times(data: Sequence[RecordingData]) -> pd.DataFrame:
    """Every uninterrupted stay in one filtered state: its state and how long it lasted.

    A stay never spans two tracks, so with segmentation on it never spans
    lost frames either. Stays cut off by the start or end of a track are
    included, which biases dwell times downwards for short tracks.
    """
    rows = []
    for recording in data:
        ordered = recording.steps.sort_values([S.TRACK_ID, S.T_START_S], kind="mergesort")
        for track_id, track in ordered.groupby(S.TRACK_ID, sort=False):
            states = track[S.STATE_FILTERED].to_numpy()
            durations = track[S.DURATION_S].to_numpy(float)
            starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
            ends = np.r_[starts[1:], len(states)]
            for start, end in zip(starts, ends):
                rows.append((recording.recording_id, track_id, int(states[start]), float(durations[start:end].sum())))
    return pd.DataFrame(rows, columns=[S.RECORDING_ID, S.TRACK_ID, S.STATE, S.DWELL_S])


def steering_fits(data: Sequence[RecordingData]) -> pd.DataFrame:
    """Steering parameters of every fitted HMM segment, long, with its weight (points fitted)."""
    frames = []
    for recording in data:
        fits = recording.segment_fits
        if not len(fits):
            continue
        long = fits.melt(id_vars=[S.TRACK_ID, S.STATE, S.N_POINTS_TOTAL], value_vars=list(STEERING_MEASURES),
                         var_name=S.PARAMETER, value_name=S.VALUE).rename(columns={S.N_POINTS_TOTAL: S.WEIGHT})
        frames.append(_keyed(long, recording.recording_id))
    columns = [S.TRACK_ID, S.STATE, S.WEIGHT, S.PARAMETER, S.VALUE, S.RECORDING_ID]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def head_cast_counts(data: Sequence[RecordingData]) -> pd.DataFrame:
    """Number of head casts per track."""
    frames = []
    for recording in data:
        casts = recording.events[recording.events[S.EVENT_TYPE] == S.EVENT_HEAD_CAST] if len(recording.events) else recording.events
        counts = casts.groupby(S.TRACK_ID).size().rename(S.N_HEAD_CASTS).reset_index() if len(casts) else pd.DataFrame(columns=[S.TRACK_ID, S.N_HEAD_CASTS])
        frames.append(_keyed(counts, recording.recording_id))
    return pd.concat(frames, ignore_index=True)


def crawl_lengths(data: Sequence[RecordingData]) -> pd.DataFrame:
    """Length of every crawl step, per track."""
    frames = []
    for recording in data:
        steps = recording.event_steps
        if not len(steps):
            continue
        crawls = steps.loc[steps[S.EVENT_TYPE] == S.EVENT_CRAWL, [S.TRACK_ID, S.STEP_LENGTH]]
        frames.append(_keyed(crawls.rename(columns={S.STEP_LENGTH: S.CRAWL_LENGTH}), recording.recording_id))
    columns = [S.TRACK_ID, S.CRAWL_LENGTH, S.RECORDING_ID]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Comparison:
    """One value per unit, measure and state, labelled with its condition; and the n's."""

    values: pd.DataFrame     # condition, recording_id, unit_id, measure, state (NaN if none), value
    counts: pd.DataFrame     # condition, n_recordings, n_larvae, n_track_segments, n_units
    unit: str
    group_keys: tuple[str, ...]
    conditions: tuple[str, ...]
    recording_ids: tuple[str, ...]
    hmm_pool: tuple[str, ...]
    warnings: tuple[str, ...]
    # recording -> {grouping column: value}; lets exported tables carry genotype etc. as columns.
    recording_groups: dict = field(default_factory=dict)


def condition_labels(registry: pd.DataFrame, recording_ids: Sequence[str], group_keys: Sequence[str]) -> dict[str, str]:
    """Condition of each recording: its values of the grouping columns, joined.

    A recording with a blank in a grouping column is refused: left alone it
    would quietly form a group of its own.
    """
    group_keys = list(group_keys)
    unknown = [key for key in group_keys if key not in R.metadata_columns(registry)]
    if unknown:
        raise AnalysisError(f"Not a metadata column: {', '.join(unknown)}")
    labels, blank = {}, []
    for recording_id in recording_ids:
        metadata = R.metadata_for(registry, recording_id)
        values = [metadata[key].strip() for key in group_keys]
        if any(value == "" for value in values):
            blank.append(recording_id)
        labels[recording_id] = " | ".join(values) if group_keys else ALL_RECORDINGS
    if blank:
        raise AnalysisError(
            f"No value for {', '.join(group_keys)} in: {', '.join(blank)}. Fill it in on the Project tab or untick them."
        )
    return labels


def _unit_column(unit: str) -> str:
    if unit not in UNITS:
        raise AnalysisError(f"Unknown unit '{unit}'. Choose one of: {', '.join(UNITS)}")
    return {UNIT_SEGMENT: S.TRACK_ID, UNIT_LARVA: S.LARVA_ID, UNIT_RECORDING: S.RECORDING_ID}[unit]


def compare(
    data: Sequence[RecordingData],
    conditions: dict[str, str],
    unit: str = DEFAULT_UNIT,
    group_keys: Sequence[str] = (),
    recording_groups: dict | None = None,
) -> Comparison:
    """Summarise every measure per unit and label each unit with its condition.

    Within a unit, raw quantities are pooled before a value is formed: state
    occupancy is time in the state over total time, dwell time is the mean
    over the unit's stays, steering parameters are means weighted by points
    fitted, head-cast rate is casts over tracked minutes, crawl length is the
    mean over the unit's crawls. No value is an average of averages.
    """
    unit_column = _unit_column(unit)
    tracks = track_table(data)
    tracks[S.TRACK_ID] = tracks[S.RECORDING_ID] + "::" + tracks[S.TRACK_ID].astype(str)

    def with_unit(table: pd.DataFrame) -> pd.DataFrame:
        table = table.copy()
        table[S.TRACK_ID] = table[S.RECORDING_ID] + "::" + table[S.TRACK_ID].astype(str)
        merged = table.merge(tracks[[S.TRACK_ID, S.LARVA_ID]], on=S.TRACK_ID, how="inner")
        merged[S.UNIT_ID] = merged[unit_column]
        return merged

    by_unit = [S.RECORDING_ID, S.UNIT_ID]
    parts: list[pd.DataFrame] = []

    def add(measure: str, table: pd.DataFrame) -> None:
        table = table.copy()
        table[S.MEASURE] = measure
        if S.STATE not in table.columns:
            table[S.STATE] = np.nan
        parts.append(table[[S.RECORDING_ID, S.UNIT_ID, S.MEASURE, S.STATE, S.VALUE]])

    seconds = with_unit(state_seconds(data))
    if len(seconds):
        per_state = seconds.groupby(by_unit + [S.STATE])[S.SECONDS].sum().unstack(S.STATE, fill_value=0.0)
        per_state = per_state.reindex(columns=range(N_STATES), fill_value=0.0)
        per_state.columns.name = S.STATE
        fractions = per_state.div(per_state.sum(axis=1), axis=0)
        add(MEASURE_OCCUPANCY, fractions.stack(future_stack=True).dropna().rename(S.VALUE).reset_index())

    dwell = with_unit(dwell_times(data))
    if len(dwell):
        add(MEASURE_DWELL, dwell.groupby(by_unit + [S.STATE])[S.DWELL_S].mean().rename(S.VALUE).reset_index())

    fits = with_unit(steering_fits(data))
    fits = fits[np.isfinite(fits[S.VALUE].astype(float)) & (fits[S.WEIGHT].astype(float) > 0)] if len(fits) else fits
    for parameter in STEERING_MEASURES:
        part = fits[fits[S.PARAMETER] == parameter] if len(fits) else fits
        if not len(part):
            continue
        weighted = part.assign(_product=part[S.VALUE].astype(float) * part[S.WEIGHT].astype(float))
        sums = weighted.groupby(by_unit + [S.STATE])[["_product", S.WEIGHT]].sum()
        add(parameter, (sums["_product"] / sums[S.WEIGHT].astype(float)).rename(S.VALUE).reset_index())

    tracked = tracks.assign(**{S.UNIT_ID: tracks[unit_column]})
    casts = with_unit(head_cast_counts(data))
    minutes = tracked.groupby(by_unit)[S.TRACKED_S].sum() / 60.0
    cast_totals = casts.groupby(by_unit)[S.N_HEAD_CASTS].sum() if len(casts) else pd.Series(dtype=float)
    rate = (cast_totals.reindex(minutes.index, fill_value=0).astype(float) / minutes).rename(S.VALUE)
    add(MEASURE_HEAD_CAST_RATE, rate[minutes > 0].reset_index())

    crawls = with_unit(crawl_lengths(data))
    if len(crawls):
        add(MEASURE_CRAWL_LENGTH, crawls.groupby(by_unit)[S.CRAWL_LENGTH].mean().rename(S.VALUE).reset_index())

    values = pd.concat(parts, ignore_index=True)
    values.insert(0, S.CONDITION, values[S.RECORDING_ID].map(conditions))

    tracks[S.CONDITION] = tracks[S.RECORDING_ID].map(conditions)
    ordered_conditions = tuple(sorted(set(conditions.values())))
    counts = tracks.groupby(S.CONDITION).agg(**{
        S.N_RECORDINGS: (S.RECORDING_ID, "nunique"),
        S.N_LARVAE: (S.LARVA_ID, "nunique"),
        S.N_TRACK_SEGMENTS: (S.TRACK_ID, "nunique"),
    }).reindex(ordered_conditions, fill_value=0).reset_index()
    counts[S.N_UNITS] = counts[{UNIT_SEGMENT: S.N_TRACK_SEGMENTS, UNIT_LARVA: S.N_LARVAE, UNIT_RECORDING: S.N_RECORDINGS}[unit]]

    recording_ids = tuple(recording.recording_id for recording in data)
    pools = {(recording.hmm_pool, recording.hmm_pool_fingerprint) for recording in data}
    pool_ids, pool_fingerprint_ = next(iter(pools))
    warnings = []
    if len(recording_ids) > 1 and (len(pools) != 1 or pool_fingerprint_ is None or set(pool_ids) != set(recording_ids)):
        warnings.append(
            "The HMM states do not come from one fit pooled over exactly these recordings, so a state may not "
            "mean the same thing in each. Run the pooled fit on this selection before comparing states."
        )
    one_recording = [condition for condition, n in zip(counts[S.CONDITION], counts[S.N_RECORDINGS]) if n == 1]
    if one_recording and len(ordered_conditions) > 1:
        warnings.append(
            f"Only one recording in: {', '.join(one_recording)}. Differences between conditions cannot be told "
            "apart from differences between recordings."
        )
    return Comparison(
        values=values, counts=counts, unit=unit, group_keys=tuple(group_keys), conditions=ordered_conditions,
        recording_ids=recording_ids, hmm_pool=tuple(pool_ids) if len(pools) == 1 else (), warnings=tuple(warnings),
        recording_groups=dict(recording_groups or {}),
    )


def compare_recordings(
    project: Project, recording_ids: Sequence[str], group_keys: Sequence[str], unit: str = DEFAULT_UNIT
) -> Comparison:
    """Load the recordings' results and compare them, grouped by registry metadata."""
    recording_ids = list(recording_ids)
    if not recording_ids:
        raise AnalysisError("Tick at least one recording.")
    conditions = condition_labels(project.registry, recording_ids, group_keys)
    data = [load_recording_data(project.pipeline(recording_id)) for recording_id in recording_ids]
    groups = {}
    for recording_id in recording_ids:
        metadata = R.metadata_for(project.registry, recording_id)
        groups[recording_id] = {key: metadata[key].strip() for key in group_keys}
    return compare(data, conditions, unit, group_keys, groups)


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────

MEASURE_DEFINITIONS = {
    MEASURE_OCCUPANCY: "Fraction of time in each filtered HMM state: summed RDP step duration in the state / total.",
    MEASURE_DWELL: "Mean duration (s) of uninterrupted stays in a filtered HMM state, including stays cut by a track end.",
    S.RHO: "Steering parameter rho: mean over the unit's fitted HMM segments, weighted by points fitted.",
    S.KAPPA: "Steering parameter kappa: weighted as rho.",
    S.MU: "Steering parameter mu: weighted as rho.",
    S.SIGMA: "Steering parameter sigma: weighted as rho.",
    MEASURE_HEAD_CAST_RATE: "Head casts per minute tracked.",
    MEASURE_CRAWL_LENGTH: "Mean crawl length (mm).",
}

EXCEL_FILE = "comparison_data.xlsx"
SHEET_ABOUT, SHEET_N, SHEET_LONG = "about", "n", "all_values_long"
SHEET_STEERING = "steering_parameters"
_ITEM, _DETAIL = "item", "value"


def state_column(state: int, prefix: str = "") -> str:
    """Header of a per-state column in the exported tables, e.g. ``state_0`` or ``rho_state_0``."""
    return f"{prefix}_{S.STATE}_{state}" if prefix else f"{S.STATE}_{state}"


def comparison_sheets(comparison: Comparison) -> dict[str, pd.DataFrame]:
    """The comparison as tables laid out for statistics software.

    ``all_values_long`` has one row per unit, measure and state. The other
    data sheets hold one figure each, one row per unit, with a column per
    state. Every data sheet carries the condition, the grouping columns
    (genotype, ...), the recording and the unit, so rows can be nested or
    used as a random effect.
    """
    keys = list(comparison.group_keys)
    id_columns = [S.CONDITION, *keys, S.RECORDING_ID, S.UNIT_ID]

    values = comparison.values.copy()
    for key in keys:
        values[key] = values[S.RECORDING_ID].map(lambda r, key=key: comparison.recording_groups.get(r, {}).get(key, ""))
    long = values[[S.CONDITION, *keys, S.RECORDING_ID, S.UNIT_ID, S.MEASURE, S.STATE, S.VALUE]]
    long = long.sort_values([S.MEASURE, S.CONDITION, S.RECORDING_ID, S.UNIT_ID, S.STATE], kind="mergesort")

    def one_measure(measure: str) -> pd.DataFrame:
        return values[values[S.MEASURE] == measure]

    def by_state(measure: str, prefix: str = "") -> pd.DataFrame:
        part = one_measure(measure)
        columns = [state_column(state, prefix) for state in range(N_STATES)]
        if not len(part):
            return pd.DataFrame(columns=id_columns + columns)
        wide = part.pivot_table(index=id_columns, columns=S.STATE, values=S.VALUE, aggfunc="first")
        wide = wide.reindex(columns=range(N_STATES))
        wide.columns = columns
        return wide.reset_index()

    def plain(measure: str) -> pd.DataFrame:
        part = one_measure(measure)
        return part[id_columns + [S.VALUE]].rename(columns={S.VALUE: measure}).sort_values(id_columns, kind="mergesort")

    steering = None
    for parameter in STEERING_MEASURES:
        table = by_state(parameter, parameter)
        steering = table if steering is None else steering.merge(table, on=id_columns, how="outer")
    steering = steering.sort_values(id_columns, kind="mergesort")

    about = [
        ("unit (one row per)", comparison.unit),
        ("grouped by", ", ".join(keys) if keys else "nothing: all recordings together"),
        ("conditions", ", ".join(comparison.conditions)),
        ("recordings", ", ".join(comparison.recording_ids)),
        ("HMM fitted over", ", ".join(comparison.hmm_pool) if comparison.hmm_pool else "not one pooled fit"),
        ("created", datetime.datetime.now().astimezone().isoformat(timespec="seconds")),
        ("independence", "Units within a recording share a plate and a session; several track segments of one "
                         "larva are not independent. Use recording_id (and the larva) as a grouping factor."),
        ("blank cells", "No value: the unit never had a fitted HMM segment or a stay in that state. "
                        "State occupancy and head-cast rate use 0 where the unit was tracked."),
        ("statistics", "None computed here."),
    ]
    about += [(f"warning {i}", warning) for i, warning in enumerate(comparison.warnings, start=1)]
    about += [(f"measure: {measure}", text) for measure, text in MEASURE_DEFINITIONS.items()]
    about += [(f"library: {name}", version) for name, version in M.environment().items()]

    return {
        SHEET_ABOUT: pd.DataFrame(about, columns=[_ITEM, _DETAIL]),
        SHEET_N: comparison.counts,
        SHEET_LONG: long,
        MEASURE_OCCUPANCY: by_state(MEASURE_OCCUPANCY),
        MEASURE_DWELL: by_state(MEASURE_DWELL),
        SHEET_STEERING: steering,
        MEASURE_HEAD_CAST_RATE: plain(MEASURE_HEAD_CAST_RATE),
        MEASURE_CRAWL_LENGTH: plain(MEASURE_CRAWL_LENGTH),
    }


def export_excel(path: Path, comparison: Comparison) -> Path:
    """Write the numbers behind the comparison figures as one Excel workbook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in comparison_sheets(comparison).items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            sheet.freeze_panes = "A2"
            for index, column in enumerate(table.columns, start=1):
                longest = max([len(str(column))] + [len(str(value)) for value in table[column].head(200)])
                sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = min(60, longest + 2)
    return path


def export_comparison(
    folder: Path,
    comparison: Comparison,
    figures: dict,
    figure_titles: dict[str, str] | None = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> Path:
    """Write a comparison as one self-describing folder.

    The figures in vector and raster form, the values and counts they were
    drawn from as CSV and as an Excel workbook, and ``figure_manifest.json`` saying what each file is,
    which recordings went in, how they were grouped, the unit, and n at each
    level. Returns the manifest path.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    comparison.values.to_csv(folder / "values.csv", index=False)
    comparison.counts.to_csv(folder / "counts.csv", index=False)
    export_excel(folder / EXCEL_FILE, comparison)
    entries = {}
    for name, figure in figures.items():
        FigureCanvasAgg(figure)
        files = []
        for extension in formats:
            path = folder / f"{name}.{extension}"
            figure.savefig(path, dpi=dpi)
            files.append(path.name)
        entries[name] = {"title": (figure_titles or {}).get(name, name), "files": files}
    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "unit": comparison.unit,
        "grouped_by": list(comparison.group_keys),
        "conditions": list(comparison.conditions),
        "recordings": list(comparison.recording_ids),
        "hmm_pool": list(comparison.hmm_pool),
        "n": comparison.counts.to_dict(orient="records"),
        "warnings": list(comparison.warnings),
        "figures": entries,
        "data_files": ["values.csv", "counts.csv", EXCEL_FILE],
        "dpi": dpi,
        "environment": M.environment(),
    }
    path = folder / FIGURE_MANIFEST
    path.write_text(json.dumps(M.to_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def new_comparison_folder(output_root: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / COMPARISON_FOLDER / f"comparison_{stamp}"
