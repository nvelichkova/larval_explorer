"""Stage 12 - turn kinematics: per-frame spine geometry (phase T1).

Design and definitions: docs/TURN_KINEMATICS.md (D-027). No upstream equivalent.

FIM-Track gives five spine points per frame - head, spinepoints 1-3, tail - so
there are three interior boundaries, numbered 1 (sp1, most anterior) to 3
(sp3). They are not resampled: interpolated boundaries would carry no
information. Bend positions are reported as a fractional boundary index on
that 1-3 scale, as an arc fraction from the head (0) to the tail (1), and as
the model-equivalent boundary ``n_body_segments × arc fraction``, which is
indicative only.

Angles are computed in radians and returned in degrees. Signs follow the
image: y points down, so a positive angle is clockwise as seen in the image.

Input is stage 02's trajectory table, not stage 03's: stage 03 has already
smoothed the centre of mass over 41 frames, and every point used here must be
smoothed alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from larval_explorer.core import schema as S
from larval_explorer.core.params import HEADING_AXES, TurnKinematicsParams

N_BOUNDARIES = 3
BOUNDARIES = (1, 2, 3)

# Spine points from head (index 0) to tail (index 4).
SPINE_POINTS = (
    (S.HEAD_X, S.HEAD_Y),
    (S.SPINEPOINT_1_X, S.SPINEPOINT_1_Y),
    (S.SPINEPOINT_2_X, S.SPINEPOINT_2_Y),
    (S.SPINEPOINT_3_X, S.SPINEPOINT_3_Y),
    (S.TAIL_X, S.TAIL_Y),
)
_TAIL, _SP2 = 4, 2
_MOM = "mom"
# Each heading axis runs from its rear point to its front point.
AXIS_POINTS = {
    "tail_sp3": (_TAIL, 3),
    "tail_sp2": (_TAIL, _SP2),
    "tail_sp1": (_TAIL, 1),
    "tail_mom": (_TAIL, _MOM),
    "tail_head": (_TAIL, 0),
}
assert tuple(AXIS_POINTS) == HEADING_AXES
# Forward progression is always measured along this axis, whatever heading is analysed.
PROGRESSION_AXIS = "tail_sp2"


# ──────────────────────────────────────────────────────────────────────────────
# Geometry on arrays: points has shape (frames, 5, 2)
# ──────────────────────────────────────────────────────────────────────────────

def spine_points(track: pd.DataFrame) -> np.ndarray:
    return np.stack([track[[x, y]].to_numpy(dtype=float) for x, y in SPINE_POINTS], axis=1)


def centre_of_mass(track: pd.DataFrame) -> np.ndarray:
    return track[[S.MOM_X, S.MOM_Y]].to_numpy(dtype=float)


def smooth(values: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay along time (axis 0), each coordinate separately."""
    return savgol_filter(values, window, order, axis=0)


def signed_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle from vector u to vector v, in (-pi, pi]; vectors on the last axis."""
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = (u * v).sum(axis=-1)
    return np.arctan2(cross, dot)


def bend_angles(points: np.ndarray) -> np.ndarray:
    """Signed bend at boundaries 1-3, walking from the tail toward the head. Shape (frames, 3).

    At boundary i: the angle from p[i+1]->p[i] to p[i]->p[i-1]. Walking this
    way gives a bend the same sign as the head angle, which also runs from the
    posterior to the anterior axis (D-032). The model walks head->tail, so its
    signed bends are the negatives of these.
    """
    toward_head = -np.diff(points, axis=1)  # link i points from p[i+1] to p[i]
    return signed_angle(toward_head[:, 1:], toward_head[:, :-1])


def head_angle(points: np.ndarray) -> np.ndarray:
    """Angle from the posterior axis (tail->sp2) to the anterior axis (sp2->head).

    The same quantity as stage 09's ``body_angle``.
    """
    return signed_angle(points[:, _SP2] - points[:, _TAIL], points[:, 0] - points[:, _SP2])


def boundary_arc_fractions(points: np.ndarray) -> np.ndarray:
    """Arc fraction (head 0, tail 1) of boundaries 1-3 in each frame. Shape (frames, 3)."""
    link_lengths = np.linalg.norm(np.diff(points, axis=1), axis=2)
    cumulative = np.cumsum(link_lengths, axis=1)
    return cumulative[:, :-1] / cumulative[:, -1:]


def axis_vector(points: np.ndarray, mom: np.ndarray, axis: str) -> np.ndarray:
    rear, front = AXIS_POINTS[axis]
    front_xy = mom if front == _MOM else points[:, front]
    return front_xy - points[:, rear]


def unwrapped_heading(vectors: np.ndarray) -> np.ndarray:
    """Direction of each vector, unwrapped over time so turns beyond ±180° stay continuous."""
    return np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))


def bend_peak(abs_bends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional boundary index (1-3) of the largest |bend|, and whether it is at an end.

    At the middle boundary the index is refined by the vertex of the parabola
    through it and its two neighbours, kept within half a boundary. At an end
    boundary there is no neighbour on one side and the integer is returned.
    A frame with no bend at all has no peak (NaN).
    """
    k = np.argmax(abs_bends, axis=1)
    index = (k + 1).astype(float)
    middle = k == 1
    y0, y1, y2 = abs_bends[middle, 0], abs_bends[middle, 1], abs_bends[middle, 2]
    curvature = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(curvature < 0, 0.5 * (y0 - y2) / curvature, 0.0)
    index[middle] += np.clip(offset, -0.5, 0.5)
    at_edge = ~middle
    flat = abs_bends.max(axis=1) <= 0
    index[flat] = np.nan
    at_edge[flat] = False
    return index, at_edge


def bend_trail(abs_bends: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-most position where |bend| is at least ``fraction`` of the largest.

    The crossing is placed by linear interpolation between the last boundary at
    or above the threshold and the next one below it. If boundary 3 is still
    above, the true edge lies behind sp3 and cannot be seen: 3 is returned and
    the frame is marked censored.
    """
    peak = abs_bends.max(axis=1)
    threshold = fraction * peak
    above = abs_bends >= threshold[:, None]
    last = N_BOUNDARIES - 1 - np.argmax(above[:, ::-1], axis=1)
    censored = last == N_BOUNDARIES - 1
    index = (last + 1).astype(float)
    rows = np.flatnonzero(~censored)
    y_last = abs_bends[rows, last[rows]]
    y_next = abs_bends[rows, last[rows] + 1]
    index[rows] += (y_last - threshold[rows]) / (y_last - y_next)
    flat = peak <= 0
    index[flat] = np.nan
    censored[flat] = False
    return index, censored


def index_to_arc(index: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """Arc fraction of a fractional boundary index, interpolating between boundaries."""
    out = np.full(len(index), np.nan)
    for row in np.flatnonzero(np.isfinite(index)):
        out[row] = np.interp(index[row], BOUNDARIES, arcs[row])
    return out


def forward_displacement(tail: np.ndarray, axis_vectors: np.ndarray) -> np.ndarray:
    """Tail displacement between consecutive frames projected on the progression axis.

    The axis direction is the mean of the two frames' unit vectors. Element 0
    is 0. Positive is forward (toward the front point of the axis).
    """
    unit = axis_vectors / np.linalg.norm(axis_vectors, axis=1, keepdims=True)
    mid = unit[1:] + unit[:-1]
    norm = np.linalg.norm(mid, axis=1, keepdims=True)
    # An exact reversal (a head/tail flip) has no mean direction: no progression.
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = np.where(norm > 0, mid / norm, 0.0)
    step = np.diff(tail, axis=0)
    return np.concatenate([[0.0], (step * mid).sum(axis=1)])


def detect_flips(points: np.ndarray) -> np.ndarray:
    """Frames whose head and tail labels are exchanged relative to the previous frame.

    Label continuity, with no threshold (D-030): the labels have swapped if
    matching each end to the *other* end's previous position costs less total
    distance than matching it to its own. Computed on unsmoothed points.
    Frame 0 is never a flip.
    """
    head, tail = points[:, 0], points[:, _TAIL]
    same = np.linalg.norm(head[1:] - head[:-1], axis=1) + np.linalg.norm(tail[1:] - tail[:-1], axis=1)
    crossed = np.linalg.norm(head[1:] - tail[:-1], axis=1) + np.linalg.norm(tail[1:] - head[:-1], axis=1)
    return np.concatenate([[False], crossed < same])


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────

def check_contiguous(track: pd.DataFrame, frame_rate_fps: float, window: int) -> None:
    """Stage 12 smooths over time, so a track must have every frame and fill the window."""
    track_id = track[S.TRACK_ID].iloc[0]
    if len(track) < window:
        raise ValueError(f"Track '{track_id}' has {len(track)} frames, fewer than the {window}-frame spine smoothing window.")
    steps = np.diff(track[S.TIME_S].to_numpy(dtype=float)) * frame_rate_fps
    if len(steps) and np.any(np.abs(steps - 1) > 0.5):
        raise ValueError(
            f"Track '{track_id}' has missing frames. Turn kinematics needs contiguous tracks: "
            "run stage 02 with segmentation enabled."
        )


def track_kinematics(
    track: pd.DataFrame,
    params: TurnKinematicsParams,
    frame_rate_fps: float,
    sg_window: int | None = None,
) -> pd.DataFrame:
    """Per-frame geometry of one contiguous track (stage 02 output, millimetres)."""
    window = params.spine_sg_window if sg_window is None else sg_window
    track = track.sort_values(S.TIME_S, kind="mergesort")
    check_contiguous(track, frame_rate_fps, window)

    raw = spine_points(track)
    points = smooth(raw, window, params.spine_sg_order)
    mom = smooth(centre_of_mass(track), window, params.spine_sg_order)

    bends = bend_angles(points)
    abs_bends = np.abs(bends)
    arcs = boundary_arc_fractions(points)
    peak, peak_edge = bend_peak(abs_bends)
    trail, trail_censored = bend_trail(abs_bends, params.trail_fraction_of_peak)
    peak_arc = index_to_arc(peak, arcs)
    trail_arc = index_to_arc(trail, arcs)

    tail = points[:, _TAIL]
    forward = forward_displacement(tail, axis_vector(points, mom, PROGRESSION_AXIS))
    path = np.concatenate([[0.0], np.linalg.norm(np.diff(tail, axis=0), axis=1)])

    out = pd.DataFrame({S.TRACK_ID: track[S.TRACK_ID].to_numpy(), S.TIME_S: track[S.TIME_S].to_numpy()})
    for b in BOUNDARIES:
        out[S.bend_column(b)] = np.rad2deg(bends[:, b - 1])
    out[S.HEAD_ANGLE_DEG] = np.rad2deg(head_angle(points))
    out[S.TOTAL_BEND_DEG] = np.rad2deg(abs_bends.sum(axis=1))
    for b in BOUNDARIES:
        out[S.boundary_arc_column(b)] = arcs[:, b - 1]
    for axis in HEADING_AXES:
        out[S.heading_column(axis)] = np.rad2deg(unwrapped_heading(axis_vector(points, mom, axis)))
    out[S.BEND_PEAK_BOUNDARY] = peak
    out[S.BEND_PEAK_ARC_FRACTION] = peak_arc
    out[S.BEND_PEAK_MODEL_BOUNDARY] = peak_arc * params.n_body_segments
    out[S.BEND_PEAK_AT_EDGE] = peak_edge
    out[S.BEND_TRAIL_BOUNDARY] = trail
    out[S.BEND_TRAIL_ARC_FRACTION] = trail_arc
    out[S.BEND_TRAIL_MODEL_BOUNDARY] = trail_arc * params.n_body_segments
    out[S.BEND_TRAIL_CENSORED] = trail_censored
    out[S.TAIL_FORWARD_SPEED_MM_S] = forward * frame_rate_fps
    out[S.POSTERIOR_DISTANCE_MM] = np.cumsum(np.maximum(forward, 0.0))
    out[S.TAIL_PATH_LENGTH_MM] = np.cumsum(path)
    out[S.SPINE_FLIP] = detect_flips(raw)
    for flag in (S.IS_WELL_ORIENTED, S.IS_COILED, S.INTERPOLATED):
        if flag in track.columns:
            out[flag] = track[flag].to_numpy()
    return out


def frame_kinematics(
    trajectories: pd.DataFrame,
    params: TurnKinematicsParams,
    frame_rate_fps: float,
    sg_window: int | None = None,
) -> pd.DataFrame:
    """``track_kinematics`` for every track of a stage 02 trajectory table."""
    S.TRAJECTORY.validate(trajectories)
    parts = [
        track_kinematics(track, params, frame_rate_fps, sg_window)
        for _, track in trajectories.groupby(S.TRACK_ID, sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def _in_events(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        inside |= (times >= start) & (times <= end)
    return inside


def clean_frames(kinematics: pd.DataFrame) -> pd.Series:
    """Frames with no flip, a well-oriented spine, and real (not bridged) data."""
    clean = ~kinematics[S.SPINE_FLIP].astype(bool)
    if S.IS_WELL_ORIENTED in kinematics.columns:
        clean &= kinematics[S.IS_WELL_ORIENTED] == 1
    if S.INTERPOLATED in kinematics.columns:
        clean &= ~kinematics[S.INTERPOLATED].astype(bool)
    return clean


def noise_floor(kinematics: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Spread of the bend and head angles while the larva crawls, per track.

    Measured on clean frames inside stage 09 crawls. Crawling larvae still bend
    a little, so this is an upper bound on measurement noise: noise plus small
    real bends. One row per (track, quantity) with the SD, median |value| and
    95th percentile |value|, in degrees.
    """
    quantities = [S.bend_column(b) for b in BOUNDARIES] + [S.HEAD_ANGLE_DEG, S.TOTAL_BEND_DEG]
    rows = []
    clean = clean_frames(kinematics)
    for track_id, frames in kinematics[clean].groupby(S.TRACK_ID, sort=True):
        crawls = events[(events[S.TRACK_ID] == track_id) & (events[S.EVENT_TYPE] == S.EVENT_CRAWL)]
        inside = _in_events(frames[S.TIME_S].to_numpy(), crawls[[S.START_S, S.END_S]].to_numpy())
        for quantity in quantities:
            values = frames.loc[inside, quantity].to_numpy()
            rows.append({
                S.TRACK_ID: track_id,
                S.QUANTITY: quantity,
                S.N_FRAMES: int(inside.sum()),
                S.NOISE_SD: float(np.std(values)) if len(values) else np.nan,
                S.MEDIAN_ABS: float(np.median(np.abs(values))) if len(values) else np.nan,
                S.P95_ABS: float(np.percentile(np.abs(values), 95)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) positions, inclusive, of each run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def label_pieces(kinematics: pd.DataFrame, params: TurnKinematicsParams) -> pd.DataFrame:
    """Split each track at head/tail label jumps and decide each piece's orientation (D-030).

    Within a piece the labels are constant, so it is either labelled correctly
    throughout or swapped throughout. Which one is decided by motion: the share
    of the tail's motion along the posterior axis that is head-first. Frames
    within half a smoothing window of a jump are left out, since smoothing
    straddles the jump. A share of at least ``piece_orientation_share`` is
    ``correct``, at most ``1 - piece_orientation_share`` is ``swapped``,
    anything between, or a piece with no motion, is ``unresolved``.

    Motion decides only which side of a geometric label jump is right. It is
    never a swap criterion on its own: real reversals are not bracketed by the
    ends exchanging positions. ``is_well_oriented`` is not used, because it
    fires mostly on strong bends (D-029, D-030).

    Motion counts only when the piece has at least ``piece_min_motion_frames``
    usable frames and a decisive share. A piece too short for that (tracker
    flicker, a swap of a frame or two that reverts) takes its orientation from
    the jumps that bracket it (D-033): each jump exchanges the labels, so the
    piece is the opposite of a motion-decided neighbour. If both neighbours are
    motion-decided they must agree, or the piece stays ``unresolved``. A long
    piece with indecisive motion stays ``unresolved``. ``orientation_basis``
    records which rule decided each piece.
    """
    half = params.spine_sg_window // 2
    opposite = {S.ORIENTATION_CORRECT: S.ORIENTATION_SWAPPED, S.ORIENTATION_SWAPPED: S.ORIENTATION_CORRECT}
    rows = []
    for track_id, track in kinematics.groupby(S.TRACK_ID, sort=True):
        jumps = np.flatnonzero(track[S.SPINE_FLIP].to_numpy().astype(bool))
        speed = track[S.TAIL_FORWARD_SPEED_MM_S].to_numpy()
        times = track[S.TIME_S].to_numpy()
        flagged = (track[S.IS_WELL_ORIENTED] != 1).to_numpy() if S.IS_WELL_ORIENTED in track.columns else np.zeros(len(track), bool)
        bounds = np.concatenate([[0], jumps, [len(track)]])
        track_rows = []
        for index, (first, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            # A frame's displacement also uses the frame before it, hence one more after a jump.
            lo = first + (half + 1 if first > 0 else 0)
            hi = stop - (half if stop < len(track) else 0)
            v = speed[lo:hi] if hi > lo else speed[0:0]
            forward, backward = np.maximum(v, 0).sum(), -np.minimum(v, 0).sum()
            share = forward / (forward + backward) if forward + backward > 0 else np.nan
            orientation, basis = S.ORIENTATION_UNRESOLVED, S.BASIS_UNRESOLVED
            too_short = len(v) < params.piece_min_motion_frames
            if not too_short and np.isfinite(share):
                if share >= params.piece_orientation_share:
                    orientation, basis = S.ORIENTATION_CORRECT, S.BASIS_MOTION
                elif share <= 1 - params.piece_orientation_share:
                    orientation, basis = S.ORIENTATION_SWAPPED, S.BASIS_MOTION
            track_rows.append({
                S.TRACK_ID: track_id,
                S.PIECE_INDEX: index,
                S.PIECE_START_S: times[first],
                S.PIECE_END_S: times[stop - 1],
                S.N_FRAMES: int(stop - first),
                S.HEAD_FIRST_SHARE: share,
                S.FLAGGED_FRACTION: float(flagged[first:stop].mean()),
                S.ORIENTATION: orientation,
                S.ORIENTATION_BASIS: basis,
                "_too_short": too_short,
            })
        for i, row in enumerate(track_rows):
            # Only pieces too short to carry motion evidence (flicker). A long piece
            # whose motion is indecisive stays unresolved: flipping it on parity
            # alone was shown to relabel stretches wrongly (D-033).
            if row[S.ORIENTATION_BASIS] != S.BASIS_UNRESOLVED or not row["_too_short"]:
                continue
            implied = {opposite[n[S.ORIENTATION]] for n in track_rows[max(0, i - 1):i] + track_rows[i + 1:i + 2]
                       if n[S.ORIENTATION_BASIS] == S.BASIS_MOTION}
            if len(implied) == 1:
                row[S.ORIENTATION], row[S.ORIENTATION_BASIS] = implied.pop(), S.BASIS_BRACKETING
        # A piece is only called swapped if a neighbour across a jump is known to be
        # correct. Tail-first motion with no such anchor may be a genuine reversal,
        # which must not be relabelled; it stays unresolved and is excluded instead.
        anchored = [
            any(n[S.ORIENTATION] == S.ORIENTATION_CORRECT for n in track_rows[max(0, i - 1):i] + track_rows[i + 1:i + 2])
            for i in range(len(track_rows))
        ]
        for row, has_anchor in zip(track_rows, anchored):
            if row[S.ORIENTATION] == S.ORIENTATION_SWAPPED and not has_anchor:
                row[S.ORIENTATION], row[S.ORIENTATION_BASIS] = S.ORIENTATION_UNRESOLVED, S.BASIS_UNANCHORED
        for row in track_rows:
            del row["_too_short"]
        rows += track_rows
    columns = [S.TRACK_ID, S.PIECE_INDEX, S.PIECE_START_S, S.PIECE_END_S, S.N_FRAMES,
               S.HEAD_FIRST_SHARE, S.FLAGGED_FRACTION, S.ORIENTATION, S.ORIENTATION_BASIS]
    return pd.DataFrame(rows, columns=columns)


def frame_orientation(kinematics: pd.DataFrame, pieces: pd.DataFrame) -> np.ndarray:
    """Orientation of the label piece each frame belongs to, aligned with ``kinematics``."""
    out = np.empty(len(kinematics), dtype=object)
    for piece in pieces.itertuples(index=False):
        rows = ((kinematics[S.TRACK_ID] == piece.track_id)
                & (kinematics[S.TIME_S] >= piece.piece_start_s) & (kinematics[S.TIME_S] <= piece.piece_end_s)).to_numpy()
        out[rows] = piece.orientation
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Phase T2: turns
# ──────────────────────────────────────────────────────────────────────────────

def crawl_mask(kinematics: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Frames of ``kinematics`` inside a stage 09 crawl of their own track."""
    inside = np.zeros(len(kinematics), dtype=bool)
    crawls = events[events[S.EVENT_TYPE] == S.EVENT_CRAWL]
    for track_id, rows in kinematics.groupby(S.TRACK_ID, sort=False).groups.items():
        mine = crawls[crawls[S.TRACK_ID] == track_id][[S.START_S, S.END_S]].to_numpy()
        positions = kinematics.index.get_indexer(rows)
        inside[positions] = _in_events(kinematics.loc[rows, S.TIME_S].to_numpy(), mine)
    return inside


def crawl_reference_frames(kinematics: pd.DataFrame, events: pd.DataFrame, orientation: np.ndarray) -> np.ndarray:
    """Clean, correctly labelled frames inside crawls: the reference for thresholds."""
    clean = (~kinematics[S.SPINE_FLIP].astype(bool)).to_numpy()
    if S.INTERPOLATED in kinematics.columns:
        clean &= ~kinematics[S.INTERPOLATED].astype(bool).to_numpy()
    return clean & (orientation == S.ORIENTATION_CORRECT) & crawl_mask(kinematics, events)


def crawl_total_bends(kinematics: pd.DataFrame, events: pd.DataFrame, orientation: np.ndarray) -> np.ndarray:
    """Total bend of every reference crawl frame. Pooled across recordings, its
    ``end_total_bend_crawl_percentile`` is ``end_total_bend_deg`` (D-031)."""
    return kinematics.loc[crawl_reference_frames(kinematics, events, orientation), S.TOTAL_BEND_DEG].to_numpy()


def larva_reference(kinematics, events, orientation, body_length) -> pd.DataFrame:
    """Per larva: mean body length, median crawl forward speed, median stride (mm).

    Body length is the frame-weighted mean of the larva's track segments'
    stage 04 means. A stride is the forward tail distance over one stage 09
    crawl (spine-length minimum to minimum), from crawls whose frames are all
    reference frames.
    """
    reference = crawl_reference_frames(kinematics, events, orientation)
    larva = kinematics[S.TRACK_ID].map(S.larva_index).to_numpy()
    body = body_length.set_index(S.TRACK_ID)[S.BODY_LENGTH_MEAN]
    crawls = events[events[S.EVENT_TYPE] == S.EVENT_CRAWL]
    strides: dict[int, list[float]] = {}
    for track_id, rows in kinematics.groupby(S.TRACK_ID, sort=False).groups.items():
        track = kinematics.loc[rows]
        positions = kinematics.index.get_indexer(rows)
        times = track[S.TIME_S].to_numpy()
        distance = track[S.POSTERIOR_DISTANCE_MM].to_numpy()
        for start, end in crawls[crawls[S.TRACK_ID] == track_id][[S.START_S, S.END_S]].to_numpy():
            i0, i1 = np.searchsorted(times, start), np.searchsorted(times, end, side="right") - 1
            if i1 > i0 and reference[positions[i0:i1 + 1]].all():
                strides.setdefault(S.larva_index(track_id), []).append(distance[i1] - distance[i0])
    rows = []
    for index in sorted(set(larva)):
        mine = larva == index
        tracks = kinematics.loc[mine, S.TRACK_ID].value_counts()
        speeds = np.maximum(kinematics.loc[mine & reference, S.TAIL_FORWARD_SPEED_MM_S].to_numpy(), 0)
        rows.append({
            S.LARVA_INDEX: index,
            S.BODY_LENGTH_MM: float(np.average(body.loc[tracks.index], weights=tracks.to_numpy())),
            S.LARVA_CRAWL_SPEED_MM_S: float(np.median(speeds)) if len(speeds) else np.nan,
            S.STRIDE_MM: float(np.median(strides[index])) if strides.get(index) else np.nan,
            S.N_STRIDES: len(strides.get(index, [])),
            S.N_REFERENCE_FRAMES: int((mine & reference).sum()),
        })
    return pd.DataFrame(rows)


def _sweep_runs(head_angle: np.ndarray, threshold: float, release: float) -> list[tuple[int, int]]:
    """Sweeps: runs of one sign, with hysteresis (D-034).

    A sweep starts when |head angle| reaches ``threshold`` and lasts while the
    sign holds and |head angle| stays at or above ``release`` times the
    sweep's running maximum. Brief dips below the threshold therefore do not
    end a sweep early; the synthetic control showed that a strict threshold
    ends slow sweeps up to 2 s before their peak.
    """
    runs, start, sign, best = [], None, 0.0, 0.0
    for i, h in enumerate(head_angle):
        if start is not None:
            if np.sign(h) == sign and abs(h) >= release * best:
                best = max(best, abs(h))
                continue
            runs.append((start, i - 1))
            start = None
        if abs(h) >= threshold:
            start, sign, best = i, np.sign(h), abs(h)
    if start is not None:
        runs.append((start, len(head_angle) - 1))
    return runs


def _fraction(sweep: float, total: float, minimum: float) -> tuple[float, float]:
    if not np.isfinite(total) or abs(total) < minimum:
        return np.nan, np.nan
    return abs(sweep) / abs(total), sweep / total


def _travel_fit(x: np.ndarray, y: np.ndarray, params: TurnKinematicsParams) -> tuple[float, float, int, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    span = float(x.max() - x.min()) if len(x) else np.nan
    if len(x) < params.travel_min_frames or not span >= params.travel_min_distance_seglen:
        return np.nan, np.nan, len(x), span
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (residual ** 2).sum() / total) if total > 0 else np.nan
    return float(slope), r2, len(x), span


_AT_MAX = (S.BEND_PEAK_BOUNDARY, S.BEND_PEAK_ARC_FRACTION, S.BEND_PEAK_MODEL_BOUNDARY, S.BEND_PEAK_AT_EDGE,
           S.BEND_TRAIL_BOUNDARY, S.BEND_TRAIL_ARC_FRACTION, S.BEND_TRAIL_MODEL_BOUNDARY, S.BEND_TRAIL_CENSORED)
_HEADING_CHANGE = (S.DPSI_SWEEP_DEG, S.DPSI_TOTAL_DEG, S.FRAC_HEADING_IN_SWEEP, S.FRAC_HEADING_IN_SWEEP_SIGNED)

# Every column of the turns table, in the order written. A turn excluded early
# leaves the later ones blank.
TURN_COLUMNS = (
    S.TRACK_ID, S.LARVA_INDEX, S.TURN_INDEX,
    S.HEAD_CAST_START_S, S.HEAD_CAST_END_S, S.T_ONSET_S, S.T_SWEEP_END_S, S.T_MAX_BEND_S, S.T_END_S,
    S.INCLUDED, S.EXCLUSION_REASON, S.FRAC_COMPUTED,
    S.HEADING_AXIS, *_HEADING_CHANGE, *(S.model_axis_column(c) for c in _HEADING_CHANGE),
    S.CAST_PEAK_HEAD_ANGLE_DEG, S.PEAK_HEAD_ANGLE_DEG, S.N_SWEEPS, S.MULTI_SWEEP,
    S.TURN_DURATION_S, S.SWEEP_DURATION_S, S.MAX_TOTAL_BEND_DEG,
    S.SWEEP_END_PROGRESS, S.SWEEP_END_DISTANCE_SEGLEN, S.TURN_DISTANCE_SEGLEN,
    *(S.at_max_column(c) for c in _AT_MAX),
    S.BEND_TRAVEL_PER_SEGLEN, S.BEND_TRAVEL_PER_WAVE, S.BEND_TRAVEL_R2, S.BEND_TRAVEL_N_FRAMES,
    S.BEND_TRAVEL_DISTANCE_SEGLEN,
    S.BODY_LENGTH_MM, S.SEGMENT_LENGTH_MM, S.STRIDE_SEGLEN, S.LARVA_CRAWL_SPEED_MM_S, S.END_SPEED_THRESHOLD_MM_S,
    S.N_INTERPOLATED_FRAMES, S.N_FLAGGED_FRAMES, S.N_CORRECTED_FRAMES, S.LABEL_JUMP_IN_WINDOW, S.WINDOW_ORIENTATION,
)


def heading_curves(times, distance, heading, i_on, i_end, seglen, params: TurnKinematicsParams) -> list[tuple]:
    """Cumulative heading change from onset as a fraction of the turn's own total (D-034).

    Returns ``(curve, x, fraction_of_total, dpsi_into_turn_deg)`` tuples on three
    abscissae: normalised progress (onset 0, end 1), forward tail distance since
    onset in segment lengths, and time since onset. ``dpsi_into_turn_deg`` is
    the change in degrees with the turn's final direction positive, so
    counter-rotation is negative on both scales. Empty when the turn is below
    ``min_abs_dpsi_total_deg``, where the normalisation is noise over noise.
    """
    span = slice(i_on, i_end + 1)
    change = heading[span] - heading[i_on]
    total = change[-1]
    if not np.isfinite(total) or abs(total) < params.min_abs_dpsi_total_deg:
        return []
    t = times[span] - times[i_on]
    d = (distance[span] - distance[i_on]) / seglen
    fraction = change / total
    into = change * np.sign(total)
    out = []
    progress = t / t[-1]
    for x in np.round(np.arange(0, 1 + 1e-9, params.curve_progress_step), 10):
        f = float(np.interp(x, progress, fraction))
        out.append((S.CURVE_PROGRESS, x, f, f * abs(total)))
    for x in np.round(np.arange(0, params.curve_max_distance_seglen + 1e-9, params.curve_distance_step_seglen), 10):
        if x > d[-1]:
            break
        k = int(np.searchsorted(d, x, side="left"))  # first frame having crawled x
        out.append((S.CURVE_DISTANCE, x, float(fraction[k]), float(into[k])))
    for k, x in enumerate(np.round(t, 10)):
        out.append((S.CURVE_TIME, x, float(fraction[k]), float(into[k])))
    return out


def curve_summary(curves: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median and quartiles of the curves at each x, with the number of turns contributing."""
    keys = by + [S.HEADING_AXIS, S.CURVE, S.CURVE_X]
    grouped = curves.groupby(keys, sort=True)[S.FRACTION_OF_TOTAL]
    return pd.DataFrame({
        S.N_TURNS: grouped.size(),
        S.MEDIAN: grouped.median(),
        S.Q25: grouped.quantile(0.25),
        S.Q75: grouped.quantile(0.75),
    }).reset_index()


def track_turns(track: pd.DataFrame, casts: pd.DataFrame, reference: dict, params: TurnKinematicsParams,
                frame_rate_fps: float) -> tuple[list[dict], list[pd.DataFrame], list[dict], list[dict]]:
    """Turn rows, per-frame windows and per-axis rows for one track's stage 09 head casts.

    ``track`` is the track's kinematics, positionally indexed, with an
    ``orientation`` column; ``reference`` holds the larva's body length, crawl
    speed and stride.
    """
    t = track[S.TIME_S].to_numpy()
    n = len(t)
    head = track[S.HEAD_ANGLE_DEG].to_numpy()
    total_bend = track[S.TOTAL_BEND_DEG].to_numpy()
    speed = track[S.TAIL_FORWARD_SPEED_MM_S].to_numpy()
    distance = track[S.POSTERIOR_DISTANCE_MM].to_numpy()
    jumps = track[S.SPINE_FLIP].to_numpy().astype(bool)
    orientation = track[S.ORIENTATION].to_numpy()
    zeros = np.zeros(n, dtype=bool)
    interpolated = track[S.INTERPOLATED].to_numpy().astype(bool) if S.INTERPOLATED in track else zeros
    coiled = (track[S.IS_COILED] == 1).to_numpy() if S.IS_COILED in track else zeros
    flagged = (track[S.IS_WELL_ORIENTED] != 1).to_numpy() if S.IS_WELL_ORIENTED in track else zeros
    relabelled = track[S.LABELS_CORRECTED].to_numpy().astype(bool) if S.LABELS_CORRECTED in track else zeros
    headings = {axis: track[S.heading_column(axis)].to_numpy() for axis in HEADING_AXES}
    track_id = track[S.TRACK_ID].iloc[0]

    body_mm = reference[S.BODY_LENGTH_MM]
    seglen = body_mm / params.n_body_segments
    stride_seglen = reference[S.STRIDE_MM] / seglen
    if params.end_speed_mode == "relative":
        speed_threshold = params.end_speed_fraction_of_crawl * reference[S.LARVA_CRAWL_SPEED_MM_S]
    else:
        speed_threshold = params.end_speed_absolute_bl_s * body_mm
    lookback = int(round(params.max_onset_lookback_s * frame_rate_fps))
    max_frames = int(round(params.max_turn_duration_s * frame_rate_fps))
    pad = int(round(params.frame_padding_s * frame_rate_fps))
    hold = params.end_hold_frames
    ends_ok = (total_bend < params.end_total_bend_deg) & (speed > speed_threshold)

    casts = casts.sort_values(S.START_S)
    starts = np.searchsorted(t, casts[S.START_S].to_numpy(), side="left")
    stops = np.searchsorted(t, casts[S.END_S].to_numpy(), side="right") - 1
    starts = np.minimum(starts, n - 1)
    stops = np.maximum(stops, starts)  # a cast between two frames still has its nearest frame
    rows, windows, axis_rows, curve_rows = [], [], [], []
    for j, (i_s, i_e) in enumerate(zip(starts, stops)):
        row = {
            S.TRACK_ID: track_id,
            S.LARVA_INDEX: S.larva_index(track_id),
            S.TURN_INDEX: j,
            S.HEAD_CAST_START_S: t[i_s], S.HEAD_CAST_END_S: t[i_e],
            S.CAST_PEAK_HEAD_ANGLE_DEG: float(np.abs(head[i_s:i_e + 1]).max()),
            S.HEADING_AXIS: params.heading_axis,
            S.BODY_LENGTH_MM: body_mm, S.SEGMENT_LENGTH_MM: seglen, S.STRIDE_SEGLEN: stride_seglen,
            S.LARVA_CRAWL_SPEED_MM_S: reference[S.LARVA_CRAWL_SPEED_MM_S],
            S.END_SPEED_THRESHOLD_MM_S: speed_threshold,
            S.INCLUDED: False, S.EXCLUSION_REASON: None, S.FRAC_COMPUTED: False,
        }
        rows.append(row)
        window = (max(0, i_s - pad), min(n - 1, i_e + pad))

        lo = max(0, i_s - lookback)
        straight = np.flatnonzero(np.abs(head[lo:i_s + 1]) < params.onset_head_angle_deg)
        if not len(straight):
            row[S.EXCLUSION_REASON] = S.EXCLUDE_ONSET_NOT_FOUND
            windows.append((j, window))
            continue
        i_on = lo + straight[-1]
        row[S.T_ONSET_S] = t[i_on]

        first = next(((a, b) for a, b in _sweep_runs(head[i_on:], params.sweep_head_angle_deg, params.sweep_release_fraction) if a <= i_e - i_on), None)
        if first is None:
            row[S.EXCLUSION_REASON] = S.EXCLUDE_NO_SWEEP
            windows.append((j, window))
            continue
        a, b = first[0] + i_on, first[1] + i_on
        i_sw = a + int(np.argmax(np.abs(head[a:b + 1])))
        row.update({S.T_SWEEP_END_S: t[i_sw], S.PEAK_HEAD_ANGLE_DEG: head[i_sw], S.SWEEP_DURATION_S: t[i_sw] - t[i_on]})

        limit = min(n - 1, i_on + max_frames)
        if j + 1 < len(starts):
            limit = min(limit, starts[j + 1] - 1)
        i_end = next((i for i in range(i_sw + 1, limit - hold + 2) if ends_ok[i:i + hold].all()), None)
        if i_end is None:
            row[S.EXCLUSION_REASON] = S.EXCLUDE_END_NOT_FOUND
            windows.append((j, (max(0, i_on - pad), limit)))
            continue

        span = slice(i_on, i_end + 1)
        i_mb = i_on + int(np.argmax(total_bend[span]))
        sweeps = _sweep_runs(head[span], params.sweep_head_angle_deg, params.sweep_release_fraction)
        for axis in HEADING_AXES:
            h = headings[axis]
            sweep, total = h[i_sw] - h[i_on], h[i_end] - h[i_on]
            values = (sweep, total, *_fraction(sweep, total, params.min_abs_dpsi_total_deg))
            axis_rows.append({S.TRACK_ID: track_id, S.TURN_INDEX: j, S.HEADING_AXIS: axis,
                              S.SPINE_SG_WINDOW: params.spine_sg_window, **dict(zip(_HEADING_CHANGE, values))})
            if axis == params.heading_axis:
                row.update(dict(zip(_HEADING_CHANGE, values)))
            if axis == params.model_heading_axis:
                row.update({S.model_axis_column(name): value for name, value in zip(_HEADING_CHANGE, values)})

        for axis in (params.heading_axis, params.model_heading_axis):
            for curve, x, fraction, into in heading_curves(t, distance, headings[axis], i_on, i_end, seglen, params):
                curve_rows.append({S.TRACK_ID: track_id, S.TURN_INDEX: j, S.HEADING_AXIS: axis,
                                   S.CURVE: curve, S.CURVE_X: x, S.FRACTION_OF_TOTAL: fraction, S.DPSI_INTO_TURN_DEG: into})
        row.update({
            S.SWEEP_END_PROGRESS: (t[i_sw] - t[i_on]) / (t[i_end] - t[i_on]),
            S.SWEEP_END_DISTANCE_SEGLEN: (distance[i_sw] - distance[i_on]) / seglen,
            S.TURN_DISTANCE_SEGLEN: (distance[i_end] - distance[i_on]) / seglen,
        })

        travel = np.arange(i_mb, i_end + 1)
        travel = travel[total_bend[travel] >= params.travel_min_total_bend_deg]
        x = (distance[travel] - distance[i_mb]) / seglen
        y = track[S.BEND_PEAK_MODEL_BOUNDARY].to_numpy()[travel]
        slope, r2, n_fit, fit_span = _travel_fit(x, y, params)
        row.update({
            S.T_MAX_BEND_S: t[i_mb], S.T_END_S: t[i_end],
            S.N_SWEEPS: len(sweeps), S.MULTI_SWEEP: len(sweeps) > 1,
            S.TURN_DURATION_S: t[i_end] - t[i_on],
            S.MAX_TOTAL_BEND_DEG: total_bend[i_mb],
            S.BEND_TRAVEL_PER_SEGLEN: slope, S.BEND_TRAVEL_PER_WAVE: slope * stride_seglen,
            S.BEND_TRAVEL_R2: r2, S.BEND_TRAVEL_N_FRAMES: n_fit, S.BEND_TRAVEL_DISTANCE_SEGLEN: fit_span,
            S.N_INTERPOLATED_FRAMES: int(interpolated[span].sum()),
            S.N_FLAGGED_FRAMES: int(flagged[span].sum()),
            S.N_CORRECTED_FRAMES: int(relabelled[span].sum()),
        })
        for column in _AT_MAX:
            row[S.at_max_column(column)] = track[column].iloc[i_mb]

        w0, w1 = i_on - pad, i_end + pad
        window = (max(0, w0), min(n - 1, w1))
        windows.append((j, window))
        jump = bool(jumps[window[0]:window[1] + 1].any())
        kinds = set(orientation[span])
        row[S.LABEL_JUMP_IN_WINDOW] = jump
        row[S.WINDOW_ORIENTATION] = "/".join(sorted(kinds))
        checks = (
            (S.EXCLUDE_NEAR_TRACK_EDGE, w0 < 0 or w1 > n - 1),
            (S.EXCLUDE_INTERPOLATED, interpolated[span].any()),
            (S.EXCLUDE_LABEL_JUMP, jump),
            (S.EXCLUDE_SWAPPED, S.ORIENTATION_SWAPPED in kinds),
            (S.EXCLUDE_UNRESOLVED, S.ORIENTATION_UNRESOLVED in kinds),
            (S.EXCLUDE_COILED, coiled[span].any()),
        )
        reason = next((name for name, failed in checks if failed), None)
        row[S.EXCLUSION_REASON] = reason
        row[S.INCLUDED] = reason is None
        row[S.FRAC_COMPUTED] = bool(np.isfinite(row[S.FRAC_HEADING_IN_SWEEP]))

    frames = []
    for j, (w0, w1) in windows:
        piece = track.iloc[w0:w1 + 1].copy()
        piece.insert(1, S.TURN_INDEX, j)
        frames.append(piece)
    return rows, frames, axis_rows, curve_rows


def censoring_report(turns: pd.DataFrame) -> pd.DataFrame:
    """Turns dropped by each filter in order, and whether the dropped ones were larger.

    Size is the cast's peak |head angle|, which exists for every candidate. A
    filter is flagged when its dropped turns have the higher median and the
    Mann-Whitney test gives p < 0.05. The last two rows are not exclusions: the
    small-turn rule blanks only the fraction, and the flag variant is the
    D-029 sensitivity check.
    """
    from scipy.stats import mannwhitneyu

    def median(values: pd.Series) -> float:
        values = values.dropna()
        return float(values.median()) if len(values) else np.nan

    def summary(name, kept, dropped, n_in):
        size_k, size_d = kept[S.CAST_PEAK_HEAD_ANGLE_DEG], dropped[S.CAST_PEAK_HEAD_ANGLE_DEG]
        p = float(mannwhitneyu(size_d, size_k).pvalue) if len(size_d) >= 3 and len(size_k) >= 3 else np.nan
        biased = bool(len(size_d) and len(size_k) and size_d.median() > size_k.median() and p < 0.05)
        return {S.FILTER: name, S.N_IN: n_in, S.N_DROPPED: len(dropped), S.N_OUT: len(kept),
                S.DROPPED_MEDIAN_CAST_PEAK_DEG: median(size_d),
                S.KEPT_MEDIAN_CAST_PEAK_DEG: median(size_k),
                S.DROPPED_MEDIAN_ABS_DPSI_TOTAL_DEG: median(dropped[S.DPSI_TOTAL_DEG].abs()) if S.DPSI_TOTAL_DEG in dropped else np.nan,
                S.KEPT_MEDIAN_ABS_DPSI_TOTAL_DEG: median(kept[S.DPSI_TOTAL_DEG].abs()) if S.DPSI_TOTAL_DEG in kept else np.nan,
                S.MANNWHITNEY_P: p, S.BIASED_TOWARD_LARGE_TURNS: biased}

    rows = [{S.FILTER: S.CANDIDATES, S.N_IN: len(turns), S.N_DROPPED: 0, S.N_OUT: len(turns)}]
    remaining = turns
    for reason in S.EXCLUSION_ORDER:
        dropped = remaining[remaining[S.EXCLUSION_REASON] == reason]
        kept = remaining[remaining[S.EXCLUSION_REASON] != reason]
        rows.append(summary(reason, kept, dropped, len(remaining)))
        remaining = kept
    small = ~remaining[S.FRAC_COMPUTED].astype(bool)
    rows.append(summary(S.SMALL_TURN, remaining[~small], remaining[small], len(remaining)))
    flagged = remaining[S.N_FLAGGED_FRAMES] > 0
    rows.append(summary(S.SENSITIVITY_NOT_WELL_ORIENTED, remaining[~flagged], remaining[flagged], len(remaining)))
    return pd.DataFrame(rows)


_LABEL_SWAP = (
    (S.HEAD_X, S.TAIL_X), (S.HEAD_Y, S.TAIL_Y),
    (S.SPINEPOINT_1_X, S.SPINEPOINT_3_X), (S.SPINEPOINT_1_Y, S.SPINEPOINT_3_Y),
)
MAX_CORRECTION_PASSES = 3


def swap_labels(trajectories: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Exchange head and tail, and sp1 and sp3, in the given rows. sp2 and the centre of mass stay."""
    out = trajectories.copy()
    for a, b in _LABEL_SWAP:
        out.loc[rows, a] = trajectories.loc[rows, b].to_numpy()
        out.loc[rows, b] = trajectories.loc[rows, a].to_numpy()
    return out


def prepare_recording(trajectories, events, params: TurnKinematicsParams, frame_rate_fps: float):
    """Per-frame kinematics with label orientation, and the final label pieces.

    With ``correct_swapped_labels`` (the default, D-033), frames in a piece
    oriented ``swapped`` are relabelled, which is an exact inversion, and the
    geometry is recomputed. This repeats until no piece is swapped, at most
    ``MAX_CORRECTION_PASSES`` times. A corrected swap leaves no label jump
    behind, so the turn around it is kept rather than excluded. The input table
    is not modified. The ``labels_corrected`` column marks relabelled frames.
    """
    data = trajectories.sort_values([S.TRACK_ID, S.TIME_S], kind="mergesort").reset_index(drop=True)
    corrected = np.zeros(len(data), dtype=bool)
    for _ in range(MAX_CORRECTION_PASSES + 1):
        kinematics = frame_kinematics(data, params, frame_rate_fps)
        pieces = label_pieces(kinematics, params)
        orientation = frame_orientation(kinematics, pieces)
        swapped = orientation == S.ORIENTATION_SWAPPED
        if not params.correct_swapped_labels or not swapped.any():
            break
        data = swap_labels(data, swapped)
        corrected ^= swapped
    kinematics[S.ORIENTATION] = orientation
    kinematics[S.LABELS_CORRECTED] = corrected
    return kinematics, pieces


def stage_12(
    trajectories: pd.DataFrame,
    events: pd.DataFrame,
    body_length: pd.DataFrame,
    params: TurnKinematicsParams,
    frame_rate_fps: float,
    stage_09_head_cast_counts: pd.Series | None = None,
):
    """Turn table, censoring report and per-frame windows for one recording.

    ``params.end_total_bend_deg`` must be set: it is the pooled crawl
    percentile across every recording in the analysis (D-031), computed with
    ``crawl_total_bends`` beforehand. If stage 09's per-track head-cast counts
    are given, the candidates must match them exactly (D-027).
    """
    from larval_explorer.core.result import StageResult

    if params.end_total_bend_deg is None:
        raise ValueError("end_total_bend_deg is not set: compute the pooled crawl percentile first (D-031).")
    kinematics, pieces = prepare_recording(trajectories, events, params, frame_rate_fps)
    reference = larva_reference(kinematics, events, kinematics[S.ORIENTATION].to_numpy(), body_length)
    by_larva = reference.set_index(S.LARVA_INDEX)

    casts = events[events[S.EVENT_TYPE] == S.EVENT_HEAD_CAST]
    rows, frames, axis_rows, curve_rows = [], [], [], []
    for track_id, track in kinematics.groupby(S.TRACK_ID, sort=True):
        mine = casts[casts[S.TRACK_ID] == track_id]
        if not len(mine):
            continue
        r, f, a, c = track_turns(track.reset_index(drop=True), mine, by_larva.loc[S.larva_index(track_id)].to_dict(),
                              params, frame_rate_fps)
        rows += r
        frames += f
        axis_rows += a
        curve_rows += c
    turns = pd.DataFrame(rows).reindex(columns=list(TURN_COLUMNS))

    if stage_09_head_cast_counts is not None:
        found = turns.groupby(S.TRACK_ID).size() if len(turns) else pd.Series(dtype=int)
        expected = stage_09_head_cast_counts[stage_09_head_cast_counts > 0]
        mismatch = found.reindex(expected.index.union(found.index), fill_value=0) != \
            expected.reindex(expected.index.union(found.index), fill_value=0)
        if mismatch.any():
            raise AssertionError(f"Stage 12 candidates differ from stage 09 head-cast counts for: {list(mismatch[mismatch].index)}")

    censoring = censoring_report(turns)
    biased = censoring[S.BIASED_TOWARD_LARGE_TURNS].astype("boolean").fillna(False).astype(bool)
    warnings = tuple(f"Filter '{name}' drops larger turns than it keeps." for name in censoring.loc[biased, S.FILTER])
    diagnostics = {
        "candidates": len(turns),
        "included": int(turns[S.INCLUDED].sum()) if len(turns) else 0,
        "fraction_computed": int(turns[S.FRAC_COMPUTED].sum()) if len(turns) else 0,
        "end_total_bend_deg": params.end_total_bend_deg,
        "end_speed_mode": params.end_speed_mode,
        "label_jumps_remaining": int(kinematics[S.SPINE_FLIP].sum()),
        "frames_relabelled": int(kinematics[S.LABELS_CORRECTED].sum()),
        "correct_swapped_labels": params.correct_swapped_labels,
        "frames_by_orientation": kinematics[S.ORIENTATION].value_counts().to_dict(),
        "biased_filters": list(censoring.loc[biased, S.FILTER]),
    }
    tables = {
        "turns": turns,
        "censoring": censoring,
        "axis_sensitivity": pd.DataFrame(axis_rows),
        "heading_curves": pd.DataFrame(curve_rows, columns=[S.TRACK_ID, S.TURN_INDEX, S.HEADING_AXIS, S.CURVE, S.CURVE_X,
                                                            S.FRACTION_OF_TOTAL, S.DPSI_INTO_TURN_DEG]),
        "label_pieces": pieces,
        "larva_reference": reference,
        "turn_frames": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
    }
    return StageResult(tables, diagnostics, warnings)
