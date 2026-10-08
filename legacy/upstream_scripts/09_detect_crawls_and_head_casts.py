#!/usr/bin/env python3
"""09_detect_crawls_and_head_casts.py
=================================================
Detect crawl and head-cast intervals from the per-frame trajectory table.

This script reproduces the event-detection pipeline used in the manuscript. It
works trajectory by trajectory, smooths the tracked coordinates and body axes,
computes spine-length oscillations and angular velocities, estimates a
trajectory-specific angular-velocity threshold, validates candidate head casts
with kinematic filters, and finally exports an event table plus diagnostic
figures.

Outputs
-------
The main analytical output is ``outputs/09_event_detection/event_intervals.csv``.
Additional per-trajectory diagnostic plots are written to subfolders inside the
same output directory.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle  # ← NEW

from scipy.signal import savgol_filter, find_peaks

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# ===== Body angle and amplitude =====
HC_BODYAMP_MIN_DEG = 30.0
HC_BODYAMP_MIN_RAD = np.deg2rad(HC_BODYAMP_MIN_DEG)

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

def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def compute_angle(x1, y1, x2, y2, x3, y3):
    """
    Angle at (x2,y2) between vectors (x1,y1)->(x2,y2) and (x2,y2)->(x3,y3).
    """
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x3 - x2, y3 - y2])
    ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    return normalize_angle(ang)

# ──────────────────────────────────────────────────────────────────────────────
# Output folders
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "trajectory_timeseries.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "09_event_detection"

out = {k: OUTPUT_ROOT / v for k, v in {
    "events": "out_events",
    "spine":  "out_spine",
    "prob":   "out_prob",
    "traj_crawl":          "out_traj_crawl",
    "traj_headcast":       "out_traj_headcast",
    "traj_headcast_circ":  "out_traj_headcast_circled",  # ← NEW
    "ant_angle":           "out_ant_angle",    
    "post_angle":          "out_post_angle",   
    "lin_vel":             "out_lin_vel",
}.items()}

for p in out.values():
    p.mkdir(exist_ok=True)

SG_WIN, SG_POLY = 21, 3
PALETTE = [c for c in plt.cm.tab10.colors if c not in ((0,0,0),(1,0,0))]

# ──────────────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
all_events = []
for aid, grp in df.groupby("larva"):
    # time
    t = grp["time"].values.astype(float)

    # 1) Smooth trajectories (21-pt SG, poly=3)
    hx = savgol_filter(grp["head_x"], 21, 3); hy = savgol_filter(grp["head_y"], 21, 3)
    tx = savgol_filter(grp["tail_x"], 21, 3); ty = savgol_filter(grp["tail_y"], 21, 3)
    mx = savgol_filter(grp["mom_x"],  21, 3); my = savgol_filter(grp["mom_y"],  21, 3)
    sp1x= savgol_filter(grp["spinepoint_1_x"],21,3); sp1y= savgol_filter(grp["spinepoint_1_y"],21,3)
    sp2x= savgol_filter(grp["spinepoint_2_x"],21,3); sp2y= savgol_filter(grp["spinepoint_2_y"],21,3)
    sp3x= savgol_filter(grp["spinepoint_3_x"],21,3); sp3y= savgol_filter(grp["spinepoint_3_y"],21,3)

    # 2) Spine length & peaks
    raw_sp_len = (
        np.hypot(sp1x - grp["head_x"], sp1y - grp["head_y"]) +
        np.hypot(sp2x - sp1x,        sp2y - sp1y) +
        np.hypot(sp3x - sp2x,        sp3y - sp2y) +
        np.hypot(grp["tail_x"] - sp3x, grp["tail_y"] - sp3y)
    )
    mean_spine = raw_sp_len.mean()
    sp_s = savgol_filter(raw_sp_len, 7, 3)
    peaks_max, _ = find_peaks(sp_s, prominence=0.05)
    peaks_min, _ = find_peaks(-sp_s, prominence=0.05)

    # 3) Compute angles & smooth
    A_cauda = compute_angle(tx,ty, sp3x,sp3y, hx,hy)
    A_sp2   = compute_angle(tx,ty, sp2x,sp2y, hx,hy)
    A_head  = compute_angle(tx,ty, sp1x,sp1y, hx,hy)
    A_posterior  = np.arctan2(sp2y - ty, sp2x - tx)
    A_anterior   = np.arctan2(hy - sp2y, hx - sp2x)
    A_body = body_angle_series(A_posterior, A_anterior)
    for A in (A_cauda, A_sp2, A_head, A_posterior, A_anterior):
        A[:] = savgol_filter(A, 7, 3)

    # 4) Angular velocities
    Acv     = savgol_filter(np.gradient(A_cauda,     t), 11, 3)
    Asp2v   = savgol_filter(np.gradient(A_sp2,       t), 11, 3)
    Aheadv  = savgol_filter(np.gradient(A_head,      t), 11, 3)
    Apostv  = savgol_filter(np.gradient(A_posterior, t), 11, 3)
    Aanttv  = savgol_filter(np.gradient(A_anterior,  t), 11, 3)

    # 5) Linear speeds & amplitude (for later classification)
    hv = np.hypot(np.gradient(hx, t), np.gradient(hy, t)) # head speed
    tv = np.hypot(np.gradient(tx, t), np.gradient(ty, t)) # tail speed
    mv = np.hypot(np.gradient(mx, t), np.gradient(my, t)) # center of mass speed
    amp = hv - tv # amplitude between head speed and tail speed

    # 6) Dynamic thresholds
    # 6.1) posterior angular-velocity threshold
    abs_Apostv = np.abs(Apostv)
    cnt_post, edges_post = np.histogram(abs_Apostv, bins=100)
    prob_post = cnt_post / cnt_post.sum()
    bc_post   = (edges_post[:-1] + edges_post[1:]) * 0.5  # bin centers

    best_e_post, best_i_post = np.inf, None
    best_p0_post, best_p1_post = None, None
    for i in range(5, len(bc_post) - 5):
        x0, y0, w0 = bc_post[:i], prob_post[:i], cnt_post[:i]
        p0 = np.polyfit(x0, y0, 1, w=w0)
        e0 = np.sum(w0 * (y0 - np.polyval(p0, x0))**2)
        x1, y1, w1 = bc_post[i:], prob_post[i:], cnt_post[i:]
        p1 = np.polyfit(x1, y1, 1, w=w1)
        e1 = np.sum(w1 * (y1 - np.polyval(p1, x1))**2)
        if e0 + e1 < best_e_post:
            best_e_post, best_i_post = e0 + e1, i
            best_p0_post, best_p1_post = p0, p1
    m0_p, b0_p = best_p0_post
    m1_p, b1_p = best_p1_post
    THRESH_POSTERIOR_AV = 1.1*((b1_p - b0_p) / (m0_p - m1_p))

    # 6.2) anterior angular-velocity threshold
    abs_Aanttv = np.abs(Aanttv)
    cnt_ant, edges_ant = np.histogram(abs_Aanttv, bins=100)
    prob_ant = cnt_ant / cnt_ant.sum()
    bc_ant   = (edges_ant[:-1] + edges_ant[1:]) * 0.5  # bin centers

    best_e_ant, best_i_ant = np.inf, None
    best_p0_ant, best_p1_ant = None, None
    for i in range(5, len(bc_ant) - 5):
        x0, y0, w0 = bc_ant[:i], prob_ant[:i], cnt_ant[:i]
        p0 = np.polyfit(x0, y0, 1, w=w0)
        e0 = np.sum(w0 * (y0 - np.polyval(p0, x0))**2)
        x1, y1, w1 = bc_ant[i:], prob_ant[i:], cnt_ant[i:]
        p1 = np.polyfit(x1, y1, 1, w=w1)
        e1 = np.sum(w1 * (y1 - np.polyval(p1, x1))**2)
        if e0 + e1 < best_e_ant:
            best_e_ant, best_i_ant = e0 + e1, i
            best_p0_ant, best_p1_ant = p0, p1
    m0_a, b0_a = best_p0_ant
    m1_a, b1_a = best_p1_ant
    THRESH_ANTERIOR_AV = 1.1*((b1_a - b0_a) / (m0_a - m1_a))

    # 7) Detect head casts
    peaks_vpos, _ = find_peaks(Aanttv,  prominence=0.05)
    peaks_vneg, _ = find_peaks(-Aanttv, prominence=0.05)
    peaks_all = np.sort(np.concatenate([peaks_vpos, peaks_vneg]))

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
        i_start = int(i0 + 0.9 * d0)
        i_end   = int(idx + 0.9 * d1)
        i_start = max(0, i_start)
        i_end   = min(len(t) - 1, i_end)
        head_cast_events.append((t[i_start], t[i_end]))

    # head-vs-tail speed contrast filter: the head must move sufficiently faster than the tail
    validated_head_casts = []
    avg_tv = tv.mean() 
    for s, e in head_cast_events:
        mask_evt  = (t >= s) & (t <= e)         # event range
        mean_tv   = tv[mask_evt].mean()         # baseline of tail speed
        max_tv    = tv[mask_evt].max()
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
        if not head_cast_events_orig or s > head_cast_events_orig[-1][1] + 1.0:
            head_cast_events_orig.append([s, e])
        else:
            head_cast_events_orig[-1][1] = max(head_cast_events_orig[-1][1], e)
    head_cast_events = [(s, e) for s, e in head_cast_events_orig]
    hc_amp_rows = []   # (s, e, amp_rad, amp_deg, a_min, a_max) -> opcional p/ CSV
    hc_kept = []
    for s, e in head_cast_events:
        mask = (t >= s) & (t <= e)
        amp_rad, a_min, a_max = body_amp_minmax_zero(A_body[mask])
        hc_amp_rows.append((s, e, amp_rad, float(np.rad2deg(amp_rad)), a_min, a_max))
        if amp_rad >= HC_BODYAMP_MIN_RAD:
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
    crawl_events = [(s, e) for (s, e) in crawl_events if (e - s) >= 0.5]

    # 9) Save the per-trajectory event table
    ev = []
    for s, e in crawl_events:
        ev.append({"ID": aid, "type": "crawl", "start": s, "end": e})
    for s, e in head_cast_events:
        ev.append({"ID": aid, "type": "head_cast", "start": s, "end": e})
    pd.DataFrame(ev).to_csv(f"{out['events']}/events_{aid}.csv", index=False)
    all_events.extend(ev)

    # 10) Diagnostic plots

    # ── Anterior angular velocity plot (head) ──────────────────────────
    fig, ax_ant = plt.subplots(figsize=(6, 4))
    ax_ant.plot(t, Aanttv, 'b', label='dA_anterior/dt (rad/s)')
    peaks_ant_pos, _ = find_peaks(Aanttv,  prominence=0.05)
    peaks_ant_neg, _ = find_peaks(-Aanttv, prominence=0.05)
    ax_ant.axhline( THRESH_ANTERIOR_AV,  color='gray', ls='--')
    ax_ant.axhline(-THRESH_ANTERIOR_AV,  color='gray', ls='--')
    for s, e in head_cast_events:
        ax_ant.axvspan(s, e, color='red', alpha=0.3)
    ax_ant.set(title='Anterior (head) angular velocity',
               xlabel='Time (s)', ylabel='rad/s')
    ax_ant.legend()
    plt.tight_layout()
    plt.savefig(f"{out['ant_angle']}/ant_velocity_{aid}.pdf")
    plt.close(fig)

    # ── Head & tail linear speed plot ──────────────────────────────────
    fig, ax_v = plt.subplots(figsize=(6, 4))
    ax_v.plot(t, hv, label='Head speed (mm/s)')
    ax_v.plot(t, tv, label='Tail speed (mm/s)')
    for s, e in head_cast_events:
        ax_v.axvspan(s, e, color='red', alpha=0.3)
    ax_v.set(title='Linear speed – head vs tail',
             xlabel='Time (s)',
             ylabel='Speed (mm/s)')
    ax_v.legend()
    plt.tight_layout()
    plt.savefig(f"{out['lin_vel']}/lin_speed_{aid}.pdf")
    plt.close(fig)

    # ── |A_posterior_v| distribution ─────────────────────────────────
    fig, axp = plt.subplots(figsize=(6, 4))
    axp.plot(bc_post, prob_post, 'b-')
    axp.plot(bc_post[:best_i_post],  np.polyval(best_p0_post, bc_post[:best_i_post]), 'r--')
    axp.plot(bc_post[best_i_post:], np.polyval(best_p1_post, bc_post[best_i_post:]), 'g--')
    axp.axvline(THRESH_POSTERIOR_AV, color='k', ls=':', label=f'intersect = {THRESH_POSTERIOR_AV:.3f}')
    axp.set(title='|dA_posterior/dt| distribution',
            xlabel='|ω|  (rad/s)', ylabel='Probability')
    axp.legend()
    plt.tight_layout()
    plt.savefig(f"{out['prob']}/prob_post_{aid}.pdf")
    plt.close(fig)

    # ── |A_anterior_v| distribution ──────────────────────────────────
    fig, axa = plt.subplots(figsize=(6, 4))
    axa.plot(bc_ant, prob_ant, 'b-')
    axa.plot(bc_ant[:best_i_ant],  np.polyval(best_p0_ant, bc_ant[:best_i_ant]), 'r--')
    axa.plot(bc_ant[best_i_ant:], np.polyval(best_p1_ant, bc_ant[best_i_ant:]), 'g--')
    axa.axvline(THRESH_ANTERIOR_AV, color='k', ls=':', label=f'intersect = {THRESH_ANTERIOR_AV:.3f}')
    axa.set(title='|dA_anterior/dt| distribution',
            xlabel='|ω|  (rad/s)', ylabel='Probability')
    axa.legend()
    plt.tight_layout()
    plt.savefig(f"{out['prob']}/prob_ant_{aid}.pdf")
    plt.close(fig)

    # 11) PLOTTING TRAJECTORIES
    # ── Trajectories with head-cast highlighting and scale bar ───────────────────
    for name, X, Y in [
        ('mom',  mx, my),
        ('tail', tx, ty),
    ]:
        segs = np.stack([
            np.column_stack([mx[:-1], my[:-1]]),
            np.column_stack([mx[1:],  my[1:]])
        ], axis=1)

        cols = [
            'r' if any(s <= ti <= e or s <= ti1 <= e for s, e in head_cast_events)
            else 'k'
            for ti, ti1 in zip(t[:-1], t[1:])
        ]

        lc = LineCollection(segs, colors=cols, linewidth=2)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_collection(lc)
        ax.set_aspect('equal', 'datalim')
        ax.autoscale()

        # scalebar
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        mx0 = x_min + 0.05*(x_max - x_min)
        my0 = y_min + 0.05*(y_max - y_min)
        ax.hlines(my0, mx0, mx0 + mean_spine, colors='black', linewidth=3)
        ax.text(mx0 + mean_spine/2, my0 - 0.03*(y_max - y_min),
                f"{mean_spine:.2f}", ha='center', va='top', color='black')

        ax.set_title(f'{name.capitalize()} trajectory by head casts')
        ax.set_xlabel('X'); ax.set_ylabel('Y')

        plt.tight_layout()
        plt.savefig(f"{out['traj_headcast']}/{name}_{aid}.pdf")
        plt.close(fig)

    # ── Centre-of-mass trajectory (black) with red circles around each head cast ──
    if len(t) >= 2:
        segs_blk = np.stack([
            np.column_stack([mx[:-1], my[:-1]]),
            np.column_stack([mx[1:],  my[1:]])
        ], axis=1)
        lc_blk = LineCollection(segs_blk, colors=['black']*(len(t)-1), linewidth=2)

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.add_collection(lc_blk)
        ax2.set_aspect('equal', 'datalim')
        ax2.autoscale()

        for s, e in head_cast_events:
            mask = (t >= s) & (t <= e)
            xs = mx[mask]; ys = my[mask]
            if xs.size == 0:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            r = float(np.max(np.hypot(xs - cx, ys - cy)))
            if not np.isfinite(r) or r <= 0:
                r = float(max(1e-3, 0.5*mean_spine))
            circ = Circle((cx, cy), 1.1*r, fill=False, ec='red', lw=2)
            ax2.add_patch(circ)

        # scalebar
        x_min, x_max = ax2.get_xlim()
        y_min, y_max = ax2.get_ylim()
        mx0 = x_min + 0.05*(x_max - x_min)
        my0 = y_min + 0.05*(y_max - y_min)
        ax2.hlines(my0, mx0, mx0 + mean_spine, colors='black', linewidth=3)
        ax2.text(mx0 + mean_spine/2, my0 - 0.03*(y_max - y_min),
                 f"{mean_spine:.2f}", ha='center', va='top', color='black')

        ax2.set_title('COM trajectory: head-casts circled (black path, red circles)')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')

        plt.tight_layout()
        plt.savefig(f"{out['traj_headcast_circ']}/mom_circ_{aid}.pdf")
        plt.close(fig2)

pd.DataFrame(all_events).to_csv(out["events"] / "all_events.csv", index=False)


# Save the global event table across all trajectories.
all_events_df = pd.DataFrame(all_events)
all_events_df.to_csv(OUTPUT_ROOT / 'event_intervals.csv', index=False)
print(f"Saved event table to: {OUTPUT_ROOT / 'event_intervals.csv'}")
