"""Pipeline stages as pure functions: (input tables, params) -> StageResult.

No disk I/O, no figures, no Qt (D-007). Tables are in canonical column names
with a fresh RangeIndex, which is what the next upstream script would see
after reading the previous one's CSV.

Stages 09 and 11 are defined in ``events.py`` and ``steering.py`` and
re-exported here. Each body is a port of the corresponding upstream function in
``legacy/upstream_scripts/`` with file access removed and column names taken
from ``schema`` (D-013). The logic is deliberately left as upstream wrote it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from larval_explorer.core import schema as S
from larval_explorer.core.legacy import upstream
from larval_explorer.core.params import (
    BodyLengthParams,
    CalibrationParams,
    HmmFilterParams,
    HmmParams,
    HmmSegmentParams,
    RdpParams,
    ReorganizeParams,
    SegmentationParams,
    SmoothingParams,
    StepModelParams,
)
from larval_explorer.core.result import StageResult
from larval_explorer.core.segments import segment_tracks

# Stages 09 and 11 are large enough to live in their own modules.
from larval_explorer.core.events import stage_09  # noqa: E402,F401
from larval_explorer.core.steering import stage_11  # noqa: E402,F401


# ──────────────────────────────────────────────────────────────────────────────
# 00 - reorganise a raw FIM-Track export
# ──────────────────────────────────────────────────────────────────────────────

_CHANNEL = "channel"
_VALUE = "value"


def stage_00(raw: pd.DataFrame, params: ReorganizeParams) -> StageResult:
    """Raw export (rows ``<channel>(<frame>)``, one column per larva) -> wide table.

    One row per tracked (larva, frame). Frames where a larva has no value in
    any kept channel produce no row, as upstream. Track IDs follow upstream
    ``process_recordings`` for a single recording: ``trajectory_0_N`` (D-015).
    Recordings are never concatenated here (D-010).
    """
    report = S.validate_fimtrack(raw)
    report.raise_if_invalid()
    measurements = list(params.measurements_to_keep)

    labels = S.parse_fimtrack_labels(raw.index)
    keep = labels[_CHANNEL].isin(measurements).to_numpy()
    if params.max_frame is not None:
        keep &= (labels[S.FRAME] <= params.max_frame).to_numpy(bool, na_value=False)
    labels = labels[keep]
    values = raw.to_numpy(float)[keep]

    # Row-major order is file order then larva order, which is the order in
    # which upstream first meets each (larva, frame) key.
    row, column = np.nonzero(~np.isnan(values))
    long = pd.DataFrame({
        S.TRACK_ID: np.asarray(raw.columns, dtype=object)[column],
        S.FRAME: labels[S.FRAME].to_numpy(np.int64)[row],
        _CHANNEL: labels[_CHANNEL].to_numpy(object)[row],
        _VALUE: values[row, column],
    })
    keys = long[[S.TRACK_ID, S.FRAME]].drop_duplicates()
    wide = long.pivot(index=[S.TRACK_ID, S.FRAME], columns=_CHANNEL, values=_VALUE)
    wide = wide.reindex(pd.MultiIndex.from_frame(keys)).reindex(columns=measurements)
    wide.columns.name = None
    wide = wide.reset_index()

    wide[S.TRACK_ID] = wide[S.TRACK_ID].map(S.upstream_track_id)
    # Upstream writes the cells as text and reads them back, so a channel
    # holding only whole numbers arrives downstream as integers.
    for name in measurements:
        column_values = wide[name]
        if column_values.notna().all() and (column_values % 1 == 0).all():
            wide[name] = column_values.astype(np.int64)

    diagnostics = {
        "larvae": len(report.larvae),
        "frames_in_file": report.n_frames,
        "rows_out": len(wide),
        "tracks_out": int(wide[S.TRACK_ID].nunique()),
        "ignored_channels": list(report.extra_channels),
    }
    return StageResult({"wide": wide}, diagnostics, report.warnings)


# ──────────────────────────────────────────────────────────────────────────────
# 01 - calibrate
# ──────────────────────────────────────────────────────────────────────────────

def stage_01(wide: pd.DataFrame, params: CalibrationParams) -> StageResult:
    """Frames -> seconds, pixels -> millimetres.

    Upstream script 01 also filters short tracks and sorts; that half lives in
    stage 02 (D-008).
    """
    S.RAW_WIDE.validate(wide)
    data = wide.rename(columns={S.FRAME: S.TIME_S})

    # Convert integer frame indices into seconds.
    data[S.TIME_S] = (data[S.TIME_S] / params.frame_rate_fps).round(params.time_decimals)

    # Convert all tracked positions from pixels into millimetres.
    for column in S.POSITION_CHANNELS:
        data[column] = (data[column] * params.millimetres_per_pixel).round(params.position_decimals)

    return StageResult({"calibrated": data.reset_index(drop=True)}, {"rows": len(data)})


# ──────────────────────────────────────────────────────────────────────────────
# 02 - track segmentation (new), or upstream's span filter
# ──────────────────────────────────────────────────────────────────────────────

def stage_02(
    calibrated: pd.DataFrame,
    params: SegmentationParams,
    calibration: CalibrationParams,
    *,
    recording_id: str,
) -> StageResult:
    """Split tracks into contiguous segments, or reproduce upstream when disabled.

    Disabled, this is the second half of upstream script 01: a track is kept
    whole if its first-to-last span reaches the minimum, whatever was lost in
    between.
    """
    if params.enabled:
        result = segment_tracks(calibrated, params, calibration, recording_id=recording_id)
        tables = {
            "trajectories": result.table,
            "track_segments": result.segments,
            "larva_summary": result.larva_summary,
        }
        return StageResult(tables, result.diagnostics, result.warnings)

    S.TRAJECTORY.validate(calibrated)
    data = calibrated

    # Compute trajectory duration and keep only long enough tracks.
    duration_by_trajectory = data.groupby(S.TRACK_ID)[S.TIME_S].max() - data.groupby(S.TRACK_ID)[S.TIME_S].min()
    valid_ids = duration_by_trajectory[duration_by_trajectory >= params.min_segment_seconds].index
    filtered = data[data[S.TRACK_ID].isin(valid_ids)].copy()

    # Sorting is important because later scripts assume chronological order.
    filtered = filtered.sort_values([S.TRACK_ID, S.TIME_S], kind="mergesort")

    diagnostics = {
        "tracks_in": len(duration_by_trajectory),
        "tracks_out": len(valid_ids),
        "tracks_dropped": len(duration_by_trajectory) - len(valid_ids),
        "segmentation": "disabled",
    }
    return StageResult({"trajectories": filtered.reset_index(drop=True)}, diagnostics)


# ──────────────────────────────────────────────────────────────────────────────
# 03 - smooth the centre of mass
# ──────────────────────────────────────────────────────────────────────────────

def stage_03(trajectories: pd.DataFrame, params: SmoothingParams) -> StageResult:
    """Per-track Savitzky-Golay smoothing of the centre-of-mass coordinates."""
    S.TRAJECTORY.validate(trajectories)
    window_size = params.window_size
    if window_size % 2 == 0:
        window_size += 1  # Savitzky–Golay requires an odd window length.

    track_lengths = trajectories.groupby(S.TRACK_ID).size()
    too_short = track_lengths[track_lengths < window_size]
    if len(too_short):
        raise ValueError(
            f"Track(s) shorter than the smoothing window of {window_size} frames: "
            + ", ".join(f"{track} ({n} frames)" for track, n in too_short.items())
        )

    data = trajectories.sort_values([S.TRACK_ID, S.TIME_S], kind="mergesort")

    columns_to_smooth = [S.MOM_X, S.MOM_Y]
    data[columns_to_smooth] = data.groupby(S.TRACK_ID)[columns_to_smooth].transform(
        lambda series: savgol_filter(series, window_length=window_size, polyorder=params.polynomial_order)
    )

    diagnostics = {"window_size_used": window_size, "tracks": len(track_lengths)}
    return StageResult({"smoothed": data.reset_index(drop=True)}, diagnostics)


# ──────────────────────────────────────────────────────────────────────────────
# 04 - mean body length
# ──────────────────────────────────────────────────────────────────────────────

_BODY_LENGTH = "body_length"
# Upstream records the stem of the trajectory file each row was computed from.
UPSTREAM_TRAJECTORY_STEM = "trajectory_timeseries"


def stage_04(
    trajectories: pd.DataFrame,
    params: BodyLengthParams,
    source_table: str = UPSTREAM_TRAJECTORY_STEM,
) -> StageResult:
    """Mean and standard deviation of midline arc length per track."""
    S.TRAJECTORY.validate(trajectories)
    euclidean_distance = upstream("04").euclidean_distance
    data = trajectories

    body_length = (
        euclidean_distance(data[S.HEAD_X], data[S.HEAD_Y], data[S.SPINEPOINT_1_X], data[S.SPINEPOINT_1_Y]) +
        euclidean_distance(data[S.SPINEPOINT_1_X], data[S.SPINEPOINT_1_Y], data[S.SPINEPOINT_2_X], data[S.SPINEPOINT_2_Y]) +
        euclidean_distance(data[S.SPINEPOINT_2_X], data[S.SPINEPOINT_2_Y], data[S.SPINEPOINT_3_X], data[S.SPINEPOINT_3_Y]) +
        euclidean_distance(data[S.SPINEPOINT_3_X], data[S.SPINEPOINT_3_Y], data[S.TAIL_X], data[S.TAIL_Y])
    )

    stats = body_length.rename(_BODY_LENGTH).groupby(data[S.TRACK_ID]).agg(["mean", "std"]).reset_index()
    stats.columns = [S.TRACK_ID, S.BODY_LENGTH_MEAN, S.BODY_LENGTH_STD]
    stats[S.SOURCE_TABLE] = source_table

    return StageResult({"body_length": stats}, {"tracks": len(stats)})


# ──────────────────────────────────────────────────────────────────────────────
# 05 - RDP steps
# ──────────────────────────────────────────────────────────────────────────────

def rdp_iterative(x: list[float], y: list[float], t: list[float], epsilon: float):
    """Ramer-Douglas-Peucker with an explicit stack (D-006).

    Returns exactly the vertices upstream's recursive ``rdp`` returns: the same
    distance arithmetic, and on ties the first farthest point.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be zero or positive.")
    n = len(x)
    if n < 3:
        return x, y, t

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.zeros(n, dtype=bool)
    stack = [(0, n - 1)]
    while stack:
        first, last = stack.pop()
        keep[first] = keep[last] = True
        if last - first < 2:
            continue
        split_index, dmax = _farthest_point(xs, ys, first, last)
        if dmax > epsilon:
            stack.append((first, split_index))
            stack.append((split_index, last))

    kept = np.flatnonzero(keep).tolist()
    return [x[i] for i in kept], [y[i] for i in kept], [t[i] for i in kept]


def _farthest_point(xs: np.ndarray, ys: np.ndarray, first: int, last: int) -> tuple[int, float]:
    """Index and distance of the interior point farthest from the chord first-last."""
    x0, y0 = float(xs[first]), float(ys[first])
    x1, y1 = float(xs[last]), float(ys[last])
    px, py = xs[first + 1:last], ys[first + 1:last]

    denominator = math.hypot(x1 - x0, y1 - y0)
    if denominator:
        distances = np.abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0) / denominator
    else:
        perpendicular_distance = upstream("05").perpendicular_distance
        distances = np.array([perpendicular_distance(a, b, (x0, y0), (x1, y1)) for a, b in zip(px.tolist(), py.tolist())])

    # Upstream keeps a running maximum that starts at 0.0 and only moves on a
    # strictly greater distance, so NaN never wins and ties go to the first.
    distances = np.where(np.isnan(distances), -np.inf, distances)
    best = int(np.argmax(distances))
    if distances[best] > 0.0:
        return first + 1 + best, float(distances[best])
    return first, 0.0


def stage_05(trajectories: pd.DataFrame, body_lengths: pd.DataFrame, params: RdpParams) -> StageResult:
    """RDP per track, with epsilon scaled by the track's mean body length."""
    S.TRAJECTORY.validate(trajectories)
    S.BODY_LENGTH.validate(body_lengths)
    build_steps_and_angles = upstream("05").build_steps_and_angles
    name = S.RDP_STEPS.upstream_name
    epsilon_factor = params.epsilon_factor
    output_rows = []

    for larva in trajectories[S.TRACK_ID].unique():
        subset = trajectories[trajectories[S.TRACK_ID] == larva]
        x = subset[S.MOM_X].tolist()
        y = subset[S.MOM_Y].tolist()
        t = subset[S.TIME_S].tolist()

        row = body_lengths[body_lengths[S.TRACK_ID] == larva]
        if row.empty:
            raise ValueError(f"Body length not found for track '{larva}'.")
        epsilon = epsilon_factor * float(row[S.BODY_LENGTH_MEAN].iloc[0])

        sx, sy, st = rdp_iterative(x, y, t, epsilon)
        for step in build_steps_and_angles(sx, sy, st):
            step[name(S.TRACK_ID)] = larva
            step[name(S.EPSILON_VALUE_MM)] = epsilon
            step[name(S.EPSILON_FACTOR)] = epsilon_factor
            output_rows.append(step)

    columns = [
        S.TRACK_ID,
        S.T_START_S, S.T_END_S,
        S.X0_MM, S.Y0_MM, S.X1_MM, S.Y1_MM,
        S.DX_MM, S.DY_MM, S.DURATION_S, S.VELOCITY, S.STEP_LENGTH,
        S.THETA, S.PHI, S.EPSILON_VALUE_MM, S.EPSILON_FACTOR,
    ]
    output = pd.DataFrame(output_rows, columns=[name(column) for column in columns])
    steps = S.RDP_STEPS.to_canonical(output)

    diagnostics = {
        "tracks": int(steps[S.TRACK_ID].nunique()),
        "steps": len(steps),
        "steps_with_phi": int(steps[S.PHI].notna().sum()),
    }
    return StageResult({"steps": steps}, diagnostics)


# ──────────────────────────────────────────────────────────────────────────────
# 06 - HMM on phi
# ──────────────────────────────────────────────────────────────────────────────

def stage_06(
    steps: pd.DataFrame,
    params: HmmParams,
    sequence_keys: Sequence[str] = (S.TRACK_ID,),
) -> StageResult:
    """Fit the 3-state Gaussian HMM on phi; decode states and posteriors.

    ``sequence_keys`` are the columns that identify one animal's sequence. A
    table pooled over recordings must include the recording column, because
    track IDs repeat between recordings (D-015).

    Stored state indices are upstream's and are never reordered (D-016): the
    manual initialisation makes 0 persistent-turning, 1 reversing, 2 mixed.
    """
    # hmmlearn pulls in scikit-learn; imported here so the other stages do not pay for it.
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler

    S.RDP_STEPS.validate(steps)
    n_states = params.n_states
    sequence_keys = list(sequence_keys)
    steps = steps.dropna(subset=[S.PHI, S.TRACK_ID]).copy()
    if steps.empty:
        raise ValueError(
            "No RDP step has a defined phi, so the HMM cannot be fitted. A track needs at least three "
            "steps; check the RDP tolerance and the track lengths."
        )

    features = [S.PHI]
    scaler = StandardScaler()
    standardized = steps.copy()
    standardized[features] = scaler.fit_transform(steps[features])

    # Upstream stacks the sequences in sorted-key order and then writes the
    # decoded states back in row order. The two only agree if the rows are
    # already grouped in that order, which upstream's own tables always are.
    # Refuse anything else instead of attaching states to the wrong steps.
    group_number = standardized.groupby(sequence_keys).ngroup().to_numpy()
    if len(group_number) > 1 and (np.diff(group_number) < 0).any():
        raise ValueError(
            "The step table must be sorted by " + ", ".join(sequence_keys) + " before the HMM is fitted."
        )

    sequences = []
    lengths = []
    for _, group in standardized.groupby(sequence_keys):
        sequences.append(group[features].values)
        lengths.append(len(group))
    X_all = np.vstack(sequences)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=params.covariance_type,
        n_iter=params.n_iter,
        random_state=params.random_state,
        init_params="",
    )

    # Manual initialization reproduces the original modeling choice.
    mu_phi = params.mu_phi
    sigma_phi_fast = params.sigma_phi_fast
    sigma_phi_mixed = params.sigma_phi_mixed
    mu = scaler.mean_[0]
    sigma = scaler.scale_[0]
    to_z = lambda value: (value - mu) / sigma
    to_z_variance = lambda variance: variance / (sigma ** 2)

    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = np.ones((n_states, n_states)) / n_states
    model.means_ = np.array([[to_z(+mu_phi)], [to_z(-mu_phi)], [to_z(0.0)]])
    model.covars_ = np.array([
        [to_z_variance(sigma_phi_fast)],
        [to_z_variance(sigma_phi_fast)],
        [to_z_variance(sigma_phi_mixed)],
    ])

    model.fit(X_all, lengths)
    states = model.predict(X_all, lengths)
    # Upstream does not pass ``lengths`` here, so the posteriors treat all
    # sequences as one. Reproduced as is.
    probabilities = model.predict_proba(X_all)

    steps[S.STATE] = states
    steps[[S.state_probability(k) for k in range(n_states)]] = probabilities

    fitted_mean_phi = (model.means_[:, 0] * sigma + mu).tolist()
    warnings = []
    if not model.monitor_.converged:
        warnings.append(f"HMM did not converge within {params.n_iter} iterations.")
    diagnostics = {
        "sequences": len(lengths),
        "steps": len(steps),
        "converged": bool(model.monitor_.converged),
        "iterations": int(model.monitor_.iter),
        "log_likelihood_history": [float(value) for value in model.monitor_.history],
        "state_mean_phi": {str(k): fitted_mean_phi[k] for k in range(n_states)},
        "state_counts": {str(k): int((states == k).sum()) for k in range(n_states)},
        "transition_matrix": model.transmat_.tolist(),
        "phi_mean": float(mu),
        "phi_scale": float(sigma),
    }
    return StageResult({"steps": steps.reset_index(drop=True)}, diagnostics, tuple(warnings))


# ──────────────────────────────────────────────────────────────────────────────
# 07 - filter short HMM runs
# ──────────────────────────────────────────────────────────────────────────────

def stage_07(steps: pd.DataFrame, params: HmmFilterParams) -> StageResult:
    """Add a filtered state column: short state islands take a neighbour's state."""
    S.HMM_STEPS.validate(steps)
    filter_state_sequence_one_pass = upstream("07").filter_state_sequence_one_pass

    steps = steps.sort_values([S.TRACK_ID, S.T_START_S], kind="mergesort").reset_index(drop=True)
    filtered_states = []
    for _, group in steps.groupby(S.TRACK_ID, sort=False):
        filtered_states.extend(filter_state_sequence_one_pass(group[S.STATE].tolist(), params.min_run))

    steps[S.STATE_FILTERED] = filtered_states
    diagnostics = {"steps": len(steps), "steps_relabelled": int((steps[S.STATE] != steps[S.STATE_FILTERED]).sum())}
    return StageResult({"steps": steps}, diagnostics)


# ──────────────────────────────────────────────────────────────────────────────
# 08 - representative HMM segments
# ──────────────────────────────────────────────────────────────────────────────

def _hmm_segments_for_one_track(group: pd.DataFrame, min_steps: int) -> list[dict]:
    """Return contiguous filtered-state segments for one trajectory."""
    times = group[S.T_START_S].to_numpy()
    states = group[S.STATE_FILTERED].to_numpy()
    segments: list[dict] = []

    if len(group) == 0:
        return segments

    start_index = 0
    current_state = states[0]

    for i in range(1, len(group)):
        if states[i] != current_state:
            end_index = i - 1
            length = end_index - start_index + 1
            if length >= min_steps:
                segments.append({
                    S.TRACK_ID: group[S.TRACK_ID].iloc[0],
                    S.STATE: current_state,
                    S.START_INDEX: int(group.index[start_index]),
                    S.END_INDEX: int(group.index[end_index]),
                    S.START_TIME_S: float(times[start_index]),
                    S.END_TIME_S: float(times[end_index]),
                    S.N_STEPS: int(length),
                    S.DURATION_S: float(times[end_index] - times[start_index]),
                })
            start_index = i
            current_state = states[i]

    end_index = len(group) - 1
    length = end_index - start_index + 1
    if length >= min_steps:
        segments.append({
            S.TRACK_ID: group[S.TRACK_ID].iloc[0],
            S.STATE: current_state,
            S.START_INDEX: int(group.index[start_index]),
            S.END_INDEX: int(group.index[end_index]),
            S.START_TIME_S: float(times[start_index]),
            S.END_TIME_S: float(times[end_index]),
            S.N_STEPS: int(length),
            S.DURATION_S: float(times[end_index] - times[start_index]),
        })

    return segments


def stage_08(steps: pd.DataFrame, params: HmmSegmentParams) -> StageResult:
    """Runs of one filtered state lasting at least ``min_steps`` steps (HMM segments)."""
    S.HMM_STEPS_FILTERED.validate(steps)
    steps = steps.copy()

    steps[S.T_START_S] = pd.to_numeric(steps[S.T_START_S], errors="coerce")
    steps = steps.dropna(subset=[S.T_START_S, S.STATE_FILTERED]).sort_values([S.TRACK_ID, S.T_START_S], kind="mergesort").reset_index(drop=True)

    all_segments = []
    for _, group in steps.groupby(S.TRACK_ID, sort=False):
        all_segments.extend(_hmm_segments_for_one_track(group, params.min_steps))

    segments = pd.DataFrame(all_segments)
    if segments.empty:
        segments = pd.DataFrame(columns=[S.TRACK_ID, S.STATE, S.START_INDEX, S.END_INDEX, S.START_TIME_S, S.END_TIME_S, S.N_STEPS, S.DURATION_S])

    segments = segments.sort_values([S.TRACK_ID, S.START_TIME_S], kind="mergesort").reset_index(drop=True)
    diagnostics = {
        "hmm_segments": len(segments),
        "hmm_segments_by_state": {str(state): int(count) for state, count in segments[S.STATE].value_counts().sort_index().items()},
    }
    return StageResult({"hmm_segments": segments}, diagnostics)


# ──────────────────────────────────────────────────────────────────────────────
# 10 - event-level and run-anchor steps
# ──────────────────────────────────────────────────────────────────────────────

def stage_10(trajectories: pd.DataFrame, events: pd.DataFrame, params: StepModelParams) -> StageResult:
    """Two step series from trajectories and events.

    ``run_anchor_steps`` joins track start, each head-cast midpoint and track
    end. ``event_level_steps`` has one step per crawl or head cast, with the
    heading change across each head cast.
    """
    S.TRAJECTORY.validate(trajectories)
    S.EVENTS.validate(events)
    script = upstream("10")
    interp_pos, build_step_rows, angle_wrap_pi = script.interp_pos, script.build_step_rows, script.angle_wrap_pi
    name = S.EVENT_LEVEL_STEPS.upstream_name

    # sanitize and sort
    traj = trajectories.sort_values([S.TRACK_ID, S.TIME_S]).reset_index(drop=True)
    events = events.sort_values([S.TRACK_ID, S.START_S, S.END_S]).reset_index(drop=True)

    # build per-larva arrays
    per_larva = {}
    for lid, g in traj.groupby(S.TRACK_ID, sort=False):
        t = g[S.TIME_S].to_numpy(float)
        x = g[S.MOM_X].to_numpy(float)
        y = g[S.MOM_Y].to_numpy(float)
        order = np.argsort(t, kind="mergesort")
        per_larva[lid] = (t[order], x[order], y[order])

    # -------- Model 1: anchors (head_cast midpoints + start/end) --------
    rows1 = []
    for lid, (tarr, xarr, yarr) in per_larva.items():
        # Start anchor
        anchors_t = [float(tarr[0])]
        anchors_x = [float(xarr[0])]
        anchors_y = [float(yarr[0])]

        # Head-cast anchors
        lev = events[events[S.TRACK_ID] == lid]
        hc = lev[lev[S.EVENT_TYPE].astype(str).str.lower() == S.EVENT_HEAD_CAST]
        if not hc.empty:
            starts = hc[S.START_S].to_numpy(float)
            ends = hc[S.END_S].to_numpy(float)

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
                        name(S.TRACK_ID): lid,
                        name(S.T_START_S): ti,
                        name(S.T_END_S): tj,
                        name(S.X0_MM): xi,
                        name(S.Y0_MM): yi,
                        name(S.X1_MM): xj,
                        name(S.Y1_MM): yj,
                    }
                )

    df1 = S.RUN_ANCHOR_STEPS.to_canonical(build_step_rows(rows1, compute_theta=True), validate=False)

    # -------- Model 2: event-level steps with tipo + cast_theta --------
    rows2 = []
    for lid, (tarr, xarr, yarr) in per_larva.items():
        lev = events[events[S.TRACK_ID] == lid]
        if lev.empty:
            continue
        for _, r in lev.iterrows():
            t_start = float(r[S.START_S])
            t_end = float(r[S.END_S])
            x0_arr, y0_arr = interp_pos(tarr, xarr, yarr, np.array([t_start]))
            x1_arr, y1_arr = interp_pos(tarr, xarr, yarr, np.array([t_end]))
            rows2.append(
                {
                    name(S.TRACK_ID): lid,
                    name(S.T_START_S): t_start,
                    name(S.T_END_S): t_end,
                    name(S.X0_MM): float(x0_arr[0]),
                    name(S.Y0_MM): float(y0_arr[0]),
                    name(S.X1_MM): float(x1_arr[0]),
                    name(S.Y1_MM): float(y1_arr[0]),
                    name(S.EVENT_TYPE): str(r[S.EVENT_TYPE]).lower(),
                }
            )

    df2 = S.EVENT_LEVEL_STEPS.to_canonical(build_step_rows(rows2, compute_theta=True), validate=False)

    # cast_theta on head_cast rows: Δφ between adjacent *crawl* steps
    df2[S.CAST_THETA] = np.nan
    for lid, idx in df2.groupby(S.TRACK_ID).groups.items():
        sub = df2.loc[idx].sort_values(S.T_START_S).reset_index()
        # indices in the sub-DataFrame (0..n-1), map back to df2 via "index" column
        for i in range(len(sub)):
            if sub.loc[i, S.EVENT_TYPE] != S.EVENT_HEAD_CAST:
                continue
            # find previous crawl φ
            prev_phi = math.nan
            for j in range(i - 1, -1, -1):
                if sub.loc[j, S.EVENT_TYPE] == S.EVENT_CRAWL:
                    prev_phi = float(sub.loc[j, S.HEADING])
                    break
            # find next crawl φ
            next_phi = math.nan
            for j in range(i + 1, len(sub)):
                if sub.loc[j, S.EVENT_TYPE] == S.EVENT_CRAWL:
                    next_phi = float(sub.loc[j, S.HEADING])
                    break
            cast = math.nan
            if math.isfinite(prev_phi) and math.isfinite(next_phi):
                cast = float(angle_wrap_pi(next_phi - prev_phi))
            df2.loc[sub.loc[i, "index"], S.CAST_THETA] = cast

    diagnostics = {"run_anchor_steps": len(df1), "event_level_steps": len(df2)}
    return StageResult({"run_anchor_steps": df1, "event_level_steps": df2}, diagnostics)
