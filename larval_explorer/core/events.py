"""Stage 09 - detect crawls and head casts from the per-frame trajectory table.

A port of upstream ``09_detect_crawls_and_head_casts.py`` (D-013). Upstream is
a module-level script that reads its input at import and draws seven figures
per track inside the detection loop. Here the per-track analysis is a
function, the inline constants are ``EventParams`` fields, and the signals the
figures were drawn from are returned as tables for ``plots/``.

The arithmetic is upstream's, including two things that look unintended and
are reproduced on purpose: the first and last links of the spine length mix
smoothed spine points with unsmoothed head and tail coordinates, and the
posterior angular-velocity threshold is computed but takes no part in
detection.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from larval_explorer.core import schema as S
from larval_explorer.core.params import EventParams
from larval_explorer.core.result import StageResult

# Sparse histogram tails make some candidate threshold fits rank deficient; upstream
# fits them anyway and the result is unchanged. The filter is installed once, for
# this module only: a per-call warnings.catch_warnings() is not safe when several
# recordings run in threads, and let the warnings through.
warnings.filterwarnings("ignore", category=np.exceptions.RankWarning, module=__name__.replace(".", r"\."))

_FIT_PARTS = ("split_bin", "low_slope", "low_intercept", "high_slope", "high_intercept")


def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def body_angle_series(A_posterior, A_anterior):
    """
    Body angle: angle from the tail->sp2 vector to the sp2->head vector.
    Positive values indicate rightward bending; negative values indicate leftward bending in the current coordinate system.
    """
    return normalize_angle(A_anterior - A_posterior)


def body_amp_minmax_zero(angles_rad: np.ndarray):
    """
    Amplitude computed from the minimum and maximum of the body-angle series.
    - if min <= 0 <= max: use the arc that passes through zero, i.e. |min| + |max|
    - otherwise: use max - min
    Returns (amplitude_rad >= 0, min, max).
    """
    a = normalize_angle(np.asarray(angles_rad))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, np.nan, np.nan
    a_min = float(np.min(a))
    a_max = float(np.max(a))
    if a_min <= 0.0 <= a_max:
        amp = abs(a_min) + abs(a_max)
    else:
        amp = a_max - a_min
    return float(max(0.0, amp)), a_min, a_max  # guarantee a non-negative amplitude


def two_line_threshold(abs_velocity: np.ndarray, params: EventParams):
    """Threshold from the |angular velocity| distribution.

    The histogram is split in two at every admissible bin, a weighted line is
    fitted to each side, and the split with the smallest weighted squared
    error is kept. The threshold is where the two lines cross, scaled by
    ``threshold_scale``. Returns ``(threshold, split_bin, low_fit, high_fit)``
    with each fit as ``(slope, intercept)``.
    """
    cnt, edges = np.histogram(abs_velocity, bins=params.threshold_histogram_bins)
    prob = cnt / cnt.sum()
    bc = (edges[:-1] + edges[1:]) * 0.5  # bin centers

    margin = params.threshold_split_margin_bins
    best_e, best_i = np.inf, None
    best_p0, best_p1 = None, None
    for i in range(margin, len(bc) - margin):
        x0, y0, w0 = bc[:i], prob[:i], cnt[:i]
        p0 = np.polyfit(x0, y0, 1, w=w0)
        e0 = np.sum(w0 * (y0 - np.polyval(p0, x0))**2)
        x1, y1, w1 = bc[i:], prob[i:], cnt[i:]
        p1 = np.polyfit(x1, y1, 1, w=w1)
        e1 = np.sum(w1 * (y1 - np.polyval(p1, x1))**2)
        if e0 + e1 < best_e:
            best_e, best_i = e0 + e1, i
            best_p0, best_p1 = p0, p1
    if best_i is None:
        raise ValueError("Could not fit the angular-velocity threshold: no admissible histogram split.")
    m0, b0 = best_p0
    m1, b1 = best_p1
    threshold = params.threshold_scale*((b1 - b0) / (m0 - m1))
    return threshold, best_i, (m0, b0), (m1, b1)


def detect_events_for_track(grp: pd.DataFrame, params: EventParams):
    """Crawls and head casts for one track.

    Returns ``(crawl_events, head_cast_events, signals, thresholds)``: two
    lists of ``(start_s, end_s)``, a per-frame table of the signals used, and
    a dict of per-track numbers.
    """
    sg_win, sg_poly = params.sg_window_position, params.sg_polynomial_order
    win_spine, win_angle, win_vel = params.sg_window_spine_length, params.sg_window_angle, params.sg_window_angular_velocity

    # time
    t = grp[S.TIME_S].values.astype(float)

    # 1) Smooth trajectories (21-pt SG, poly=3)
    hx = savgol_filter(grp[S.HEAD_X], sg_win, sg_poly); hy = savgol_filter(grp[S.HEAD_Y], sg_win, sg_poly)
    tx = savgol_filter(grp[S.TAIL_X], sg_win, sg_poly); ty = savgol_filter(grp[S.TAIL_Y], sg_win, sg_poly)
    mx = savgol_filter(grp[S.MOM_X],  sg_win, sg_poly); my = savgol_filter(grp[S.MOM_Y],  sg_win, sg_poly)
    sp1x= savgol_filter(grp[S.SPINEPOINT_1_X],sg_win,sg_poly); sp1y= savgol_filter(grp[S.SPINEPOINT_1_Y],sg_win,sg_poly)
    sp2x= savgol_filter(grp[S.SPINEPOINT_2_X],sg_win,sg_poly); sp2y= savgol_filter(grp[S.SPINEPOINT_2_Y],sg_win,sg_poly)
    sp3x= savgol_filter(grp[S.SPINEPOINT_3_X],sg_win,sg_poly); sp3y= savgol_filter(grp[S.SPINEPOINT_3_Y],sg_win,sg_poly)

    # 2) Spine length & peaks
    raw_sp_len = (
        np.hypot(sp1x - grp[S.HEAD_X], sp1y - grp[S.HEAD_Y]) +
        np.hypot(sp2x - sp1x,        sp2y - sp1y) +
        np.hypot(sp3x - sp2x,        sp3y - sp2y) +
        np.hypot(grp[S.TAIL_X] - sp3x, grp[S.TAIL_Y] - sp3y)
    )
    mean_spine = raw_sp_len.mean()
    sp_s = savgol_filter(raw_sp_len, win_spine, sg_poly)
    peaks_min, _ = find_peaks(-sp_s, prominence=params.spine_peak_prominence)

    # 3) Compute angles & smooth
    A_posterior  = np.arctan2(sp2y - ty, sp2x - tx)
    A_anterior   = np.arctan2(hy - sp2y, hx - sp2x)
    A_body = body_angle_series(A_posterior, A_anterior)
    for A in (A_posterior, A_anterior):
        A[:] = savgol_filter(A, win_angle, sg_poly)

    # 4) Angular velocities
    Apostv  = savgol_filter(np.gradient(A_posterior, t), win_vel, sg_poly)
    Aanttv  = savgol_filter(np.gradient(A_anterior,  t), win_vel, sg_poly)

    # 5) Linear speeds & amplitude (for later classification)
    hv = np.hypot(np.gradient(hx, t), np.gradient(hy, t)) # head speed
    tv = np.hypot(np.gradient(tx, t), np.gradient(ty, t)) # tail speed
    mv = np.hypot(np.gradient(mx, t), np.gradient(my, t)) # center of mass speed
    amp = hv - tv # amplitude between head speed and tail speed

    # 6) Dynamic thresholds
    THRESH_POSTERIOR_AV, split_post, low_post, high_post = two_line_threshold(np.abs(Apostv), params)
    THRESH_ANTERIOR_AV, split_ant, low_ant, high_ant = two_line_threshold(np.abs(Aanttv), params)

    # 7) Detect head casts
    peaks_vpos, _ = find_peaks(Aanttv,  prominence=params.angular_velocity_peak_prominence)
    peaks_vneg, _ = find_peaks(-Aanttv, prominence=params.angular_velocity_peak_prominence)
    peaks_all = np.sort(np.concatenate([peaks_vpos, peaks_vneg]))

    extent = params.event_extent_fraction
    head_cast_events = []
    for idx in peaks_all:
        v_now = Aanttv[idx]
        if np.abs(v_now) < THRESH_ANTERIOR_AV:
            continue
        # determine event start and end from the nearest opposite-sign peaks
        if v_now > 0:
            vales = peaks_vneg
            prev_vales = vales[vales < idx]
            next_vales = vales[vales > idx]
            if len(prev_vales) == 0 or len(next_vales) == 0:
                continue
            i0, i1 = prev_vales[-1], next_vales[0]
        else:
            picos = peaks_vpos
            prev_picos = picos[picos < idx]
            next_picos = picos[picos > idx]
            if len(prev_picos) == 0 or len(next_picos) == 0:
                continue
            i0, i1 = prev_picos[-1], next_picos[0]
        d0 = idx - i0
        d1 = i1 - idx
        i_start = int(i0 + extent * d0)
        i_end   = int(idx + extent * d1)
        i_start = max(0, i_start)
        i_end   = min(len(t) - 1, i_end)
        head_cast_events.append((t[i_start], t[i_end]))

    # head-vs-tail speed contrast filter: the head must move sufficiently faster than the tail
    validated_head_casts = []
    avg_tv = tv.mean()
    for s, e in head_cast_events:
        mask_evt  = (t >= s) & (t <= e)         # event range
        max_amp   = amp[mask_evt].max()         # max speed amplitude
        if (max_amp > avg_tv):
            validated_head_casts.append((s, e))

    head_cast_events = validated_head_casts

    # Merge overlapping head-cast intervals
    head_cast_events_sorted = sorted(head_cast_events)
    merged_head_casts = []
    for s, e in head_cast_events_sorted:
        if not merged_head_casts or s > merged_head_casts[-1][1]:
            merged_head_casts.append([s, e])
        else:
            merged_head_casts[-1][1] = max(merged_head_casts[-1][1], e)
    head_cast_events = [(s, e) for s, e in merged_head_casts]

    # Merge head-cast intervals separated by less than 1 s
    head_cast_events_orig = []
    for s, e in head_cast_events:
        if not head_cast_events_orig or s > head_cast_events_orig[-1][1] + params.merge_gap_seconds:
            head_cast_events_orig.append([s, e])
        else:
            head_cast_events_orig[-1][1] = max(head_cast_events_orig[-1][1], e)
    head_cast_events = [(s, e) for s, e in head_cast_events_orig]

    # Keep head casts whose body-angle amplitude reaches the minimum
    hc_bodyamp_min_rad = np.deg2rad(params.hc_bodyamp_min_deg)
    hc_kept = []
    for s, e in head_cast_events:
        mask = (t >= s) & (t <= e)
        amp_rad, a_min, a_max = body_amp_minmax_zero(A_body[mask])
        if amp_rad >= hc_bodyamp_min_rad:
            hc_kept.append((s, e))
    head_cast_events = hc_kept

    # 8) Detect crawls
    crawl_events = []
    for i in range(len(peaks_min) - 1):
        cs, ce = t[peaks_min[i]], t[peaks_min[i+1]]
        if any(s <= cs < e for s, e in head_cast_events):
            continue
        for s, e in head_cast_events:
            if cs < s < ce:
                ce = s
                break
        if ce - cs > 0:
            crawl_events.append((cs, ce))
    crawl_events = [(s, e) for (s, e) in crawl_events if (e - s) >= params.min_crawl_seconds]

    signals = pd.DataFrame({
        S.TIME_S: t,
        S.COM_X_SMOOTH: mx,
        S.COM_Y_SMOOTH: my,
        S.ANTERIOR_ANGULAR_VELOCITY: Aanttv,
        S.POSTERIOR_ANGULAR_VELOCITY: Apostv,
        S.HEAD_SPEED: hv,
        S.TAIL_SPEED: tv,
        S.COM_SPEED: mv,
        S.BODY_ANGLE: A_body,
        S.SPINE_LENGTH_SMOOTH: sp_s,
    })
    thresholds = {
        S.MEAN_SPINE_LENGTH: float(mean_spine),
        S.ANTERIOR_THRESHOLD: float(THRESH_ANTERIOR_AV),
        S.POSTERIOR_THRESHOLD: float(THRESH_POSTERIOR_AV),
        S.N_CRAWLS: len(crawl_events),
        S.N_HEAD_CASTS: len(head_cast_events),
    }
    for side, split, low, high in (("anterior", split_ant, low_ant, high_ant), ("posterior", split_post, low_post, high_post)):
        values = (int(split), float(low[0]), float(low[1]), float(high[0]), float(high[1]))
        for part, value in zip(_FIT_PARTS, values):
            thresholds[S.threshold_fit_column(side, part)] = value
    return crawl_events, head_cast_events, signals, thresholds


def stage_09(trajectories: pd.DataFrame, params: EventParams) -> StageResult:
    """Detect crawls and head casts in every track.

    Tables: ``events`` (one row per crawl or head cast), ``signals`` (per
    frame, for the diagnostic figures) and ``thresholds`` (per track).
    """
    S.TRAJECTORY.validate(trajectories)

    longest_window = max(params.sg_window_position, params.sg_window_spine_length,
                         params.sg_window_angle, params.sg_window_angular_velocity)
    track_lengths = trajectories.groupby(S.TRACK_ID).size()
    too_short = track_lengths[track_lengths < longest_window]
    if len(too_short):
        raise ValueError(
            f"Track(s) shorter than the {longest_window}-frame filter window: "
            + ", ".join(f"{track} ({n} frames)" for track, n in too_short.items())
        )

    all_events = []
    all_signals = []
    all_thresholds = []
    for aid, grp in trajectories.groupby(S.TRACK_ID):
        crawl_events, head_cast_events, signals, thresholds = detect_events_for_track(grp, params)

        for s, e in crawl_events:
            all_events.append({S.TRACK_ID: aid, S.EVENT_TYPE: S.EVENT_CRAWL, S.START_S: s, S.END_S: e})
        for s, e in head_cast_events:
            all_events.append({S.TRACK_ID: aid, S.EVENT_TYPE: S.EVENT_HEAD_CAST, S.START_S: s, S.END_S: e})

        signals.insert(0, S.TRACK_ID, aid)
        all_signals.append(signals)
        all_thresholds.append({S.TRACK_ID: aid, **thresholds})

    events = pd.DataFrame(all_events, columns=[S.TRACK_ID, S.EVENT_TYPE, S.START_S, S.END_S])
    signals = pd.concat(all_signals, ignore_index=True)
    thresholds = pd.DataFrame(all_thresholds)

    warnings_out = [
        f"Track '{row[S.TRACK_ID]}' has no head casts." for _, row in thresholds.iterrows() if row[S.N_HEAD_CASTS] == 0
    ]
    diagnostics = {
        "tracks": len(thresholds),
        "crawls": int(thresholds[S.N_CRAWLS].sum()),
        "head_casts": int(thresholds[S.N_HEAD_CASTS].sum()),
    }
    return StageResult({"events": events, "signals": signals, "thresholds": thresholds}, diagnostics, tuple(warnings_out))
