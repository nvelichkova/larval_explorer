#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Steering fit (AR(2) on geometric Δheading from dX/dY) on representative segments
defined in representative_hmm_segments.csv.

Key points:
- Steering fit parameters (rho,kappa,mu,sigma) computed from crawl runs inside each segment.
- KDE is DENSITY and uses |mu| (abs(mu)).
- Segment trajectory plots color crawls vs head casts/other.
- Head cast analysis by state:
  * head-cast direction distribution: phi = atan2(dY,dX) (polar hist, density)
  * head-cast rate: n_headcasts / segment_duration (boxplot)
- Circular autocorrelation vs lag (1..MAX_LAG) by state for:
  * steering theta series = theta_geom of CRAWLS within segment (computed from dX/dY)
  * head-cast theta series = theta column from event_level_steps.csv for HEAD CAST events

*** FIX IMPORTANT (AUTOCORR) ***
Autocorr aggregation now matches your reference program:
- Unit sample = animal trajectory (per state, per kind).
- We compute autocorr INSIDE each segment, then aggregate within-animal (no cross-segment pairs),
  then aggregate across animals: mean/std over animals (std=0 if only 1 animal).
This prevents missing errorbars and reduces spurious oscillations.

ADDED FOR SIMULATION:
- Crawl length distribution per state: mean/std/PEAK of sqrt(dX^2 + dY^2) for CRAWLS.
- Headcast frequency relative to crawls per state: n_headcasts / n_crawls (event-based ratio).
- Headcast circular autocorr lag=1 (mean/std) per state (from df_state_ac).
- Von Mises circular fit for headcast theta per state:
  * save parameters (mu,kappa,R) into simulation_parameters.csv
  * save polar plot (hist density + von Mises pdf overlay)

ADDED NOW (YOUR REQUEST):
- Save PEAK (mode) of KDE distributions alongside mean/std:
  * In state_param_summary.csv: for rho/kappa/mu/abs_mu/sigma => *_peak_w (weighted KDE peak)
  * In simulation_parameters.csv:
      - carries the same *_peak_w columns from state summary
      - adds crawl_length_peak (KDE peak)
      - adds headcast_theta_peak (== headcast_theta_mu)

Outputs:
out_segmentos_steering/
- segment_fits.csv
- run_fits.csv
- state_param_summary.csv          <-- now includes peaks
- simulation_parameters.csv        <-- now includes peaks
- headcast_segment_metrics.csv
- autocorr_by_state.csv
- figures/
    - param_box_weighted.svg/pdf
    - param_kde_weighted.svg/pdf
    - autocorr_by_state.svg/pdf
    - headcast_rate_boxplot.svg/pdf
    - headcast_direction_polar.svg/pdf
    - headcast_theta_fit/<state>.svg/pdf
    - segments/<state>/segment_....svg/pdf
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

# =========================
# PATHS
# =========================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOW_PATH = str(PROJECT_ROOT / "data" / "event_level_steps.csv")
SEG_PATH = str(PROJECT_ROOT / "data" / "representative_hmm_segments.csv")
OUTDIR   = str(PROJECT_ROOT / "outputs" / "11_steering_fit_by_strategy")

# =========================
# event_level_steps columns
# =========================
ID_COL   = "animal ID"
T0_COL   = "tempo_inicial (s)"
TF_COL   = "tempo_final (s)"
TIPO_COL = "tipo"
THETA_COL = "theta"   # <-- IMPORTANT: head-cast autocorr uses THIS column

X0_COL = "X0 (mm)"
Y0_COL = "Y0 (mm)"
X_COL  = "X (mm)"
Y_COL  = "Y (mm)"
DX_COL = "dX"
DY_COL = "dY"

# =========================
# representative_hmm_segments columns
# =========================
SEG_ESTADO_COL = "state"
SEG_T0_COL     = "t_inicio (s)"
SEG_TF_COL     = "t_fim (s)"

# =========================
# Steering fit parameters
# =========================
MIN_POINTS_PER_RUN = 8

CLIP_OUTLIERS = True
CLIP_PCTS = (1, 99)

REQUIRE_STABLE = True
RHO_MIN, RHO_MAX = 0.0, 0.999

# IMPORTANT: mu stability control (mu = c/kappa becomes unstable for small kappa)
KAPPA_MIN = 0.05
MU_MAX    = 0.5    # set None to disable

# =========================
# Segment plot styling
# =========================
RECENTER_TRAJ = True
COLOR_CRAWL = "tab:blue"
COLOR_CAST  = "tab:orange"
ALPHA_LINE  = 0.95
LW_SEG      = 1.3

# =========================
# Distribution / figures
# =========================
SAVE_PDF_ALSO = True
WEIGHT_COL_SEG = "n_points_total"

# KDE bandwidth floors (prevents needle-like spikes)
BW_MIN_MAP = {
    "rho":     0.015,
    "kappa":   0.020,
    "mu":      0.020,
    "abs_mu":  0.020,
    "sigma":   0.010,
    # crawl_length_peak uses bw_min=1e-3 by default (fine)
}

# Head-cast direction polar hist bins
HC_DIR_BINS = 24

# =========================
# Circular autocorrelation
# =========================
MAX_LAG = 4
MIN_EVENTS_PER_SEG_AUTOCORR = 4  # recommended >= MAX_LAG+2
MIN_ANIMALS_PER_STATE_FOR_PLOT = 1  # plot even if 1 (std=0)


# =============================================================================
# Helpers
# =============================================================================

def angle_wrap_pi(x):
    """Wrap to (-pi, pi]."""
    return (x + np.pi) % (2*np.pi) - np.pi


def safe_slug(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^a-z0-9_\-]+", "", x)
    return x[:80] if len(x) > 80 else x


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    x = x[ok]; w = w[ok]
    return float(np.sum(w * x) / np.sum(w))


def weighted_std(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan
    x = x[ok]; w = w[ok]
    wm = np.sum(w)
    mu = np.sum(w * x) / wm
    var = np.sum(w * (x - mu) ** 2) / wm
    return float(np.sqrt(var))


def weighted_quantile(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    q = float(q)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    x = x[ok]; w = w[ok]
    idx = np.argsort(x)
    x = x[idx]; w = w[idx]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return np.nan
    p = cw / total
    return float(np.interp(q, p, x))


def weighted_kde_gaussian(xs, x, w, bw=None, bw_min=1e-3):
    """
    Weighted Gaussian KDE (DENSITY).
      f(xs) = sum_i w_i * N(xs | x_i, bw) / sum_i w_i
    """
    xs = np.asarray(xs, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)

    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.zeros_like(xs)

    x = x[ok]; w = w[ok]
    wm = np.sum(w)
    if wm <= 0:
        return np.zeros_like(xs)
    w_norm = w / wm

    if bw is None:
        n_eff = (np.sum(w) ** 2) / np.sum(w ** 2)

        s = np.std(x)
        if not np.isfinite(s) or s <= 1e-12:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            s = 1.4826 * mad
        if not np.isfinite(s) or s <= 1e-12:
            s = 1.0

        bw = s * (n_eff ** (-1.0 / 5.0))
        if not np.isfinite(bw) or bw <= 1e-12:
            bw = 1.0

    bw = max(float(bw), float(bw_min))

    z = (xs[:, None] - x[None, :]) / bw
    ker = np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * bw)
    dens = ker @ w_norm
    return dens


def weighted_kde_peak(x, w, bw_min=1e-3):
    """
    Peak (mode) of the weighted KDE density estimate.
    Returns x_peak where KDE(x) is maximal (on a grid).
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan

    x = x[ok]
    w = w[ok]

    x_lo = weighted_quantile(x, w, 0.01)
    x_hi = weighted_quantile(x, w, 0.99)
    if (not np.isfinite(x_lo)) or (not np.isfinite(x_hi)) or (x_hi <= x_lo):
        x_lo = float(np.nanmin(x))
        x_hi = float(np.nanmax(x))
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            return np.nan

    # a bit of padding so peak isn't forced to the edge
    pad = 0.06 * (x_hi - x_lo) if x_hi > x_lo else 0.0
    xs = np.linspace(x_lo - pad, x_hi + pad, 500)

    dens = weighted_kde_gaussian(xs, x, w, bw=None, bw_min=bw_min)
    if dens.size == 0 or (not np.any(np.isfinite(dens))):
        return np.nan

    return float(xs[int(np.nanargmax(dens))])


# =============================================================================
# Steering / segmentation
# =============================================================================

def compute_theta_geom_from_dxdy(g: pd.DataFrame):
    """
    theta_geom[i] = wrap( unwrap(phi_i) - unwrap(phi_{i-1}) )
    where phi_i = atan2(dY_i, dX_i), unwrap over valid steps only.
    """
    dx = pd.to_numeric(g[DX_COL], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(g[DY_COL], errors="coerce").to_numpy(float)

    step2 = dx * dx + dy * dy
    ok = np.isfinite(dx) & np.isfinite(dy) & (step2 > 1e-12)
    if ok.sum() < 3:
        return None

    phi = np.full(len(g), np.nan, float)
    phi[ok] = np.arctan2(dy[ok], dx[ok])

    idx = np.where(ok)[0]
    phi_ok = phi[idx]
    phi_unw_ok = np.unwrap(phi_ok)

    phi_unw = np.full_like(phi, np.nan)
    phi_unw[idx] = phi_unw_ok

    theta = np.full(len(g), np.nan, float)
    for j in range(1, len(idx)):
        i = idx[j]
        im1 = idx[j - 1]
        theta[i] = angle_wrap_pi(phi_unw[i] - phi_unw[im1])

    return theta


def fit_ar2(theta: np.ndarray):
    """
    theta[t] = a1*theta[t-1] + a2*theta[t-2] + c + eps
    """
    if len(theta) < 3:
        return None

    y = theta[2:]
    x1 = theta[1:-1]
    x2 = theta[:-2]
    X = np.column_stack([x1, x2, np.ones_like(x1)])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a1, a2, c = beta

    yhat = X @ beta
    resid = y - yhat
    sigma = np.std(resid, ddof=3) if len(resid) > 3 else np.std(resid)
    return a1, a2, c, sigma, resid, yhat


def ar2_to_params(a1, a2, c, sigma):
    rho = -a2
    kappa = 1 + rho - a1
    mu = np.nan
    if abs(kappa) > 1e-12:
        mu = c / kappa
    return rho, kappa, mu, sigma


def segment_runs_within_segment(df_seg: pd.DataFrame) -> pd.DataFrame:
    """
    Within a segment, define run_id_in_seg: increments when a crawl starts after non-crawl.
    """
    d = df_seg.sort_values([T0_COL]).copy()
    is_crawl = (d[TIPO_COL].astype(str).str.lower() == "crawl")
    prev_is_crawl = is_crawl.shift(1).fillna(False)
    new_run = is_crawl & (~prev_is_crawl)
    d["run_id_in_seg"] = new_run.cumsum()
    d.loc[~is_crawl, "run_id_in_seg"] = np.nan
    return d


# =============================================================================
# Head-cast helpers
# =============================================================================

def is_head_cast_series(tipo_series: pd.Series) -> pd.Series:
    """
    Heuristic: contains 'head' or contains 'cast', but not exactly 'crawl'.
    """
    s = tipo_series.astype(str).str.lower()
    is_crawl = (s == "crawl")
    is_hc = (s.str.contains("head", regex=False) | s.str.contains("cast", regex=False)) & (~is_crawl)
    return is_hc


def compute_phi_from_dxdy(df: pd.DataFrame):
    dx = pd.to_numeric(df[DX_COL], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(df[DY_COL], errors="coerce").to_numpy(float)
    step2 = dx*dx + dy*dy
    ok = np.isfinite(dx) & np.isfinite(dy) & (step2 > 1e-12)
    phi = np.full(len(df), np.nan, float)
    phi[ok] = np.arctan2(dy[ok], dx[ok])
    return phi


# =============================================================================
# Circular autocorrelation (per segment) + aggregation like your reference
# =============================================================================

def circular_autocorr_with_counts(series, max_lag):
    """
    Returns dicts:
      corr[lag] = <cos(theta_n - theta_{n+lag})>
      cnt[lag]  = number of pairs (N-lag)
    """
    series = np.asarray(series, dtype=float)
    out = {}
    cnt = {}
    for lag in range(1, max_lag + 1):
        if len(series) <= lag:
            out[lag] = np.nan
            cnt[lag] = 0
            continue
        x = series[:-lag]
        y = series[lag:]
        if len(x) < 3:
            out[lag] = np.nan
            cnt[lag] = int(len(x))
        else:
            out[lag] = float(np.mean(np.cos(x - y)))
            cnt[lag] = int(len(x))
    return out, cnt


def aggregate_autocorr_like_reference(ac_rows, max_lag):
    """
    ac_rows: rows with fields:
      state, kind, animal, lag, seg_corr, seg_cnt

    We build per (state, kind, animal): an autocorr dict lag->value using
    weighted sum over segments (weights=cnt) WITHOUT mixing across segment boundaries.
    Then aggregate across animals: mean/std over animals (std=0 if only 1 animal),
    exactly like your reference script's style.
    """
    if len(ac_rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(ac_rows)

    out_state = []
    lags = np.arange(1, max_lag + 1)

    for (state, kind), gsk in df.groupby(["state", "kind"]):
        # build per-animal dicts
        animal_dicts = []
        for animal, ga in gsk.groupby("animal"):
            # accumulate numerator/denominator over segments for each lag
            num = {lag: 0.0 for lag in lags}
            den = {lag: 0.0 for lag in lags}
            for _, r in ga.iterrows():
                lag = int(r["lag"])
                v = float(r["seg_corr"])
                c = float(r["seg_cnt"])
                if np.isfinite(v) and c > 0:
                    num[lag] += v * c
                    den[lag] += c
            # finalize animal dict
            d = {}
            for lag in lags:
                if den[lag] > 0:
                    d[lag] = num[lag] / den[lag]
                else:
                    d[lag] = np.nan
            animal_dicts.append(d)

        # aggregate across animals
        for lag in lags:
            vals = [d[lag] for d in animal_dicts if np.isfinite(d[lag])]
            n = len(vals)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals)) if n >= 2 else 0.0
            out_state.append({
                "state": state,
                "kind": kind,
                "lag": int(lag),
                "mean": mean,
                "std": std,
                "n_animals": int(n),
            })

    return pd.DataFrame(out_state).sort_values(["kind", "state", "lag"])


# =============================================================================
# Simulation extras: crawl length + vonmises headcast theta fit
# =============================================================================

def compute_crawl_length_from_dxdy(df_crawl: pd.DataFrame):
    dx = pd.to_numeric(df_crawl[DX_COL], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(df_crawl[DY_COL], errors="coerce").to_numpy(float)
    L = np.sqrt(dx*dx + dy*dy)
    L = L[np.isfinite(L)]
    return L


def fit_vonmises(theta):
    """
    Fit Von Mises distribution via mean resultant length approximation.
    Returns: mu, kappa, R
    """
    theta = np.asarray(theta, float)
    theta = theta[np.isfinite(theta)]
    if len(theta) < 3:
        return np.nan, np.nan, np.nan

    C = float(np.mean(np.cos(theta)))
    S = float(np.mean(np.sin(theta)))
    R = float(np.sqrt(C*C + S*S))
    mu = float(np.arctan2(S, C))

    # kappa approximation from R
    if R < 1e-12:
        kappa = 0.0
    elif R < 0.53:
        kappa = 2*R + R**3 + (5*R**5)/6
    elif R < 0.85:
        kappa = -0.4 + 1.39*R + 0.43/(1-R)
    else:
        kappa = 1/(R**3 - 4*R**2 + 3*R)

    if not np.isfinite(kappa) or kappa < 0:
        kappa = np.nan

    return mu, float(kappa), R


def vonmises_pdf(theta, mu, kappa):
    """
    Von Mises PDF on [-pi, pi]. Uses numpy.i0 (modified Bessel I0).
    """
    theta = np.asarray(theta, float)
    if (not np.isfinite(mu)) or (not np.isfinite(kappa)) or (kappa < 0):
        return np.full_like(theta, np.nan)
    return np.exp(kappa*np.cos(theta-mu)) / (2*np.pi*np.i0(kappa))


# =============================================================================
# Plots
# =============================================================================

def plot_segment_trajectory(df_seg: pd.DataFrame, outpath: str, title: str):
    need = {X0_COL, Y0_COL, X_COL, Y_COL, TIPO_COL}
    if not need.issubset(df_seg.columns):
        return False

    g = df_seg.sort_values(T0_COL).copy()
    x0 = pd.to_numeric(g[X0_COL], errors="coerce").to_numpy(float)
    y0 = pd.to_numeric(g[Y0_COL], errors="coerce").to_numpy(float)
    x1 = pd.to_numeric(g[X_COL],  errors="coerce").to_numpy(float)
    y1 = pd.to_numeric(g[Y_COL],  errors="coerce").to_numpy(float)

    ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    g = g.loc[ok].copy()
    if len(g) < 2:
        return False

    g = g.sort_values(T0_COL).copy()
    x0 = pd.to_numeric(g[X0_COL], errors="coerce").to_numpy(float)
    y0 = pd.to_numeric(g[Y0_COL], errors="coerce").to_numpy(float)
    x1 = pd.to_numeric(g[X_COL],  errors="coerce").to_numpy(float)
    y1 = pd.to_numeric(g[Y_COL],  errors="coerce").to_numpy(float)

    if RECENTER_TRAJ and np.isfinite(x0[0]) and np.isfinite(y0[0]):
        x00, y00 = x0[0], y0[0]
        x0 -= x00; y0 -= y00
        x1 -= x00; y1 -= y00

    tipos = g[TIPO_COL].astype(str).str.lower().to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    for i in range(len(g)):
        col = COLOR_CRAWL if tipos[i] == "crawl" else COLOR_CAST
        ax.plot([x0[i], x1[i]], [y0[i], y1[i]], lw=LW_SEG, alpha=ALPHA_LINE, color=col)

    ax.scatter([x0[0]], [y0[0]], s=25, marker="o", label="start")
    ax.scatter([x1[-1]], [y1[-1]], s=25, marker="s", label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    if SAVE_PDF_ALSO:
        plt.savefig(os.path.splitext(outpath)[0] + ".pdf")
    plt.close(fig)
    return True


def plot_param_box_weighted(seg_fit: pd.DataFrame, outpath_svg: str, weight_col: str):
    if len(seg_fit) == 0:
        return

    df = seg_fit.copy()
    df[SEG_ESTADO_COL] = df[SEG_ESTADO_COL].astype(str)

    state_order = sorted(df[SEG_ESTADO_COL].dropna().unique().tolist())
    params = ["rho", "kappa", "mu", "sigma"]

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.2))

    for ax, p in zip(axes, params):
        ax.set_title(p)
        ax.grid(True, linewidth=0.3, axis="y")

        for i, est in enumerate(state_order, start=1):
            sub = df[df[SEG_ESTADO_COL] == est]
            x = pd.to_numeric(sub[p], errors="coerce").to_numpy(float)
            w = pd.to_numeric(sub[weight_col], errors="coerce").to_numpy(float)

            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            x = x[ok]; w = w[ok]
            if len(x) == 0:
                continue

            q05 = weighted_quantile(x, w, 0.05)
            q25 = weighted_quantile(x, w, 0.25)
            q50 = weighted_quantile(x, w, 0.50)
            q75 = weighted_quantile(x, w, 0.75)
            q95 = weighted_quantile(x, w, 0.95)

            ax.plot([i, i], [q05, q95], linewidth=1.2)
            ax.plot([i-0.18, i+0.18], [q05, q05], linewidth=1.2)
            ax.plot([i-0.18, i+0.18], [q95, q95], linewidth=1.2)

            rect_x = [i-0.25, i+0.25, i+0.25, i-0.25, i-0.25]
            rect_y = [q25, q25, q75, q75, q25]
            ax.plot(rect_x, rect_y, linewidth=1.2)
            ax.plot([i-0.25, i+0.25], [q50, q50], linewidth=1.6)

            ws = np.sqrt(w / np.nanmedian(w)) if np.nanmedian(w) > 0 else np.ones_like(w)
            ws = np.clip(ws, 0.6, 3.0)
            jitter = i + 0.08*np.random.randn(len(x))
            ax.scatter(jitter, x, s=10*ws, alpha=0.45)

        ax.set_xticks(range(1, len(state_order) + 1))
        ax.set_xticklabels(state_order, rotation=30, ha="right")

    fig.suptitle(f"Weighted box summaries by strategy (weights = {weight_col})", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath_svg), exist_ok=True)
    plt.savefig(outpath_svg)
    if SAVE_PDF_ALSO:
        plt.savefig(os.path.splitext(outpath_svg)[0] + ".pdf")
    plt.close(fig)


def plot_param_kde_weighted(seg_fit: pd.DataFrame, outpath_svg: str, weight_col: str):
    if len(seg_fit) == 0:
        return

    df = seg_fit.copy()
    df[SEG_ESTADO_COL] = df[SEG_ESTADO_COL].astype(str)

    state_order = sorted(df[SEG_ESTADO_COL].dropna().unique().tolist())
    params = ["rho", "kappa", "abs_mu", "sigma"]

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.2))

    for ax, p in zip(axes, params):
        by_state = {}
        all_vals = []
        all_w = []

        for est in state_order:
            sub = df[df[SEG_ESTADO_COL] == est]
            x = pd.to_numeric(sub[p], errors="coerce").to_numpy(float)
            w = pd.to_numeric(sub[weight_col], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            x = x[ok]; w = w[ok]
            if len(x) >= 2:
                by_state[est] = (x, w)
                all_vals.append(x)
                all_w.append(w)

        if len(all_vals) == 0:
            ax.set_title(f"{p} (no data)")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        x_all = np.concatenate(all_vals)
        w_all = np.concatenate(all_w)

        x_lo = weighted_quantile(x_all, w_all, 0.01)
        x_hi = weighted_quantile(x_all, w_all, 0.99)
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            x_lo = np.nanmin(x_all)
            x_hi = np.nanmax(x_all)

        pad = 0.06 * (x_hi - x_lo) if x_hi > x_lo else 1.0
        xs = np.linspace(x_lo - pad, x_hi + pad, 350)

        bw_min = BW_MIN_MAP.get(p, 1e-3)

        for est, (x, w) in by_state.items():
            dens = weighted_kde_gaussian(xs, x, w, bw=None, bw_min=bw_min)
            ax.plot(xs, dens, lw=1.6, label=est)

        ax.set_title(r"|mu|" if p == "abs_mu" else p)
        ax.grid(True, linewidth=0.3, axis="y")
        if p != "sigma":
            lg = ax.get_legend()
            if lg:
                lg.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)), frameon=True, fontsize=8)

    fig.suptitle(f"Weighted KDE by strategy (density; weights = {weight_col})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(os.path.dirname(outpath_svg), exist_ok=True)
    plt.savefig(outpath_svg)
    if SAVE_PDF_ALSO:
        plt.savefig(os.path.splitext(outpath_svg)[0] + ".pdf")
    plt.close(fig)


def plot_headcast_rate_boxplot(seg_hc_metrics: pd.DataFrame, outpath_svg: str):
    if len(seg_hc_metrics) == 0:
        return

    df = seg_hc_metrics.copy()
    df[SEG_ESTADO_COL] = df[SEG_ESTADO_COL].astype(str)
    df["hc_rate_hz"] = pd.to_numeric(df["hc_rate_hz"], errors="coerce")

    states = sorted(df[SEG_ESTADO_COL].dropna().unique().tolist())
    data = []
    for st in states:
        vals = df.loc[df[SEG_ESTADO_COL] == st, "hc_rate_hz"]
        vals = vals[np.isfinite(vals)]
        data.append(vals.to_numpy(float))

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.2))
    ax.boxplot(data, labels=states, showfliers=False)
    ax.set_ylabel("head cast rate (Hz)")
    ax.set_title("Head cast frequency by strategy (per segment)")
    ax.grid(True, linewidth=0.3, axis="y")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath_svg), exist_ok=True)
    plt.savefig(outpath_svg)
    if SAVE_PDF_ALSO:
        plt.savefig(os.path.splitext(outpath_svg)[0] + ".pdf")
    plt.close(fig)


def plot_headcast_direction_polar(headcast_dirs_by_state: dict, outpath_svg: str, bins=24):
    states = sorted([k for k in headcast_dirs_by_state.keys() if len(headcast_dirs_by_state[k]) > 0])
    if len(states) == 0:
        return

    n = len(states)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(3.2*ncols, 3.0*nrows))

    edges = np.linspace(-np.pi, np.pi, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    for i, st in enumerate(states, start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="polar")
        phi = np.asarray(headcast_dirs_by_state[st], float)
        phi = phi[np.isfinite(phi)]
        if len(phi) == 0:
            ax.set_title(st)
            continue

        counts, _ = np.histogram(phi, bins=edges)
        density = counts / (np.sum(counts) * width) if np.sum(counts) > 0 else counts.astype(float)

        theta_plot = (centers + 2*np.pi) % (2*np.pi)
        ax.bar(theta_plot, density, width=width, align="center", alpha=0.85)
        ax.set_title(st, fontsize=10)
        ax.set_yticklabels([])
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath_svg), exist_ok=True)
    plt.savefig(outpath_svg)
    if SAVE_PDF_ALSO:
        plt.savefig(os.path.splitext(outpath_svg)[0] + ".pdf")
    plt.close(fig)


def plot_autocorr_by_state(df_state_corr: pd.DataFrame, outpath_svg: str):
    if len(df_state_corr) == 0:
        return

    states = sorted(df_state_corr["state"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200)

    for ax, kind in zip(axes, ["steering", "headcast_theta_csv"]):
        subk = df_state_corr[df_state_corr["kind"] == kind].copy()
        for st in states:
            g = subk[subk["state"] == st].sort_values("lag")
            if len(g) == 0:
                continue
            ax.errorbar(
                g["lag"].to_numpy(int),
                g["mean"].to_numpy(float),
                yerr=g["std"].to_numpy(float),
                fmt='o-',
                capsize=2,
                markersize=3,
                linewidth=1.2,
                label=f"{st} (n={int(g['n_animals'].max())})"
            )
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Delay (events)")
        ax.set_ylabel(r"$\langle\cos(\Delta\theta)\rangle$")
        ax.set_title("steering (crawls)" if kind == "steering" else "head casts (theta from CSV)")
        ax.grid(True, linewidth=0.3)

    axes[1].legend(loc="best", fontsize=7, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath_svg), exist_ok=True)
    plt.savefig(outpath_svg)
    if SAVE_PDF_ALSO:
        plt.savefig(os.path.splitext(outpath_svg)[0] + ".pdf")
    plt.close(fig)


def state_param_summary_table_weighted(seg_fit: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """
    Builds state_param_summary.csv:
      - mean/std (weighted)
      - PEAK (mode) from weighted KDE: *_peak_w
    """
    if len(seg_fit) == 0:
        return pd.DataFrame()

    df = seg_fit.copy()
    df[SEG_ESTADO_COL] = df[SEG_ESTADO_COL].astype(str)

    for p in ["rho", "kappa", "mu", "abs_mu", "sigma", weight_col]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    out_rows = []
    for est, sub in df.groupby(SEG_ESTADO_COL):
        w = sub[weight_col].to_numpy(float)
        row = {
            "state": est,
            "n_segments": int(len(sub)),
            "weight_sum": float(np.nansum(w[np.isfinite(w)])),
        }
        for p in ["rho", "kappa", "mu", "abs_mu", "sigma"]:
            if p not in sub.columns:
                row[f"{p}_mean_w"] = np.nan
                row[f"{p}_std_w"]  = np.nan
                row[f"{p}_peak_w"] = np.nan
                continue

            x = pd.to_numeric(sub[p], errors="coerce").to_numpy(float)
            row[f"{p}_mean_w"] = weighted_mean(x, w)
            row[f"{p}_std_w"]  = weighted_std(x, w)
            row[f"{p}_peak_w"] = weighted_kde_peak(x, w, BW_MIN_MAP.get(p, 1e-3))
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("state")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    figdir = os.path.join(OUTDIR, "figures")
    seg_figdir = os.path.join(figdir, "segments")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(seg_figdir, exist_ok=True)

    low = pd.read_csv(LOW_PATH)
    seg = pd.read_csv(SEG_PATH)

    # checks
    need_low = [ID_COL, T0_COL, TF_COL, TIPO_COL, DX_COL, DY_COL, X0_COL, Y0_COL, X_COL, Y_COL, THETA_COL]
    for col in need_low:
        if col not in low.columns:
            raise ValueError(f"event_level_steps.csv must contain column '{col}'.")

    for col in [ID_COL, SEG_ESTADO_COL, SEG_T0_COL, SEG_TF_COL]:
        if col not in seg.columns:
            raise ValueError(f"representative_hmm_segments.csv must contain column '{col}'.")

    # numeric casts
    low[T0_COL] = pd.to_numeric(low[T0_COL], errors="coerce")
    low[TF_COL] = pd.to_numeric(low[TF_COL], errors="coerce")
    low[THETA_COL] = pd.to_numeric(low[THETA_COL], errors="coerce")
    seg[SEG_T0_COL] = pd.to_numeric(seg[SEG_T0_COL], errors="coerce")
    seg[SEG_TF_COL] = pd.to_numeric(seg[SEG_TF_COL], errors="coerce")

    low = low.dropna(subset=[ID_COL, T0_COL, TF_COL, TIPO_COL]).copy()
    seg = seg.dropna(subset=[ID_COL, SEG_ESTADO_COL, SEG_T0_COL, SEG_TF_COL]).copy()
    low = low.sort_values([ID_COL, T0_COL]).copy()

    run_rows = []
    seg_rows = []
    hc_seg_rows = []
    headcast_dirs_by_state = {}

    # NEW accumulators for simulation params
    crawl_lengths_by_state = {}     # state -> list of lengths
    n_crawl_by_state = {}           # state -> total crawl events (within segments)
    n_hc_by_state = {}              # state -> total headcast events (within segments)
    headcast_theta_by_state = {}    # state -> list of theta from CSV for headcasts (within segments)

    # AUTOCORR accumulation rows
    autocorr_accum_rows = []  # state, kind, animal, lag, seg_corr, seg_cnt

    # stable segment_id
    seg = seg.reset_index(drop=True)
    seg["segment_id"] = np.arange(len(seg), dtype=int)

    for _, srow in seg.iterrows():
        seg_id = int(srow["segment_id"])
        animal = srow[ID_COL]
        state = str(srow[SEG_ESTADO_COL])
        t0 = float(srow[SEG_T0_COL])
        tf = float(srow[SEG_TF_COL])

        # init state buckets
        crawl_lengths_by_state.setdefault(state, [])
        n_crawl_by_state.setdefault(state, 0)
        n_hc_by_state.setdefault(state, 0)
        headcast_theta_by_state.setdefault(state, [])

        df_seg = low[(low[ID_COL] == animal) & (low[T0_COL] >= t0) & (low[T0_COL] <= tf)].copy()
        if len(df_seg) < 5:
            continue
        df_seg = df_seg.sort_values(T0_COL).copy()

        # plot segment
        out_seg_svg = os.path.join(
            seg_figdir, safe_slug(state),
            f"segment_{seg_id:04d}__{safe_slug(animal)}__t{t0:.2f}__t{tf:.2f}.svg"
        )
        title = f"{state} | {animal} | [{t0:.2f}, {tf:.2f}] s"
        plot_segment_trajectory(df_seg, out_seg_svg, title=title)

        # Identify crawls and headcasts inside THIS segment
        tipo_lower = df_seg[TIPO_COL].astype(str).str.lower()
        is_crawl = tipo_lower.eq("crawl")
        is_hc = is_head_cast_series(df_seg[TIPO_COL])

        # accumulate counts for simulation ratio
        n_crawl_by_state[state] += int(is_crawl.sum())
        n_hc_by_state[state] += int(is_hc.sum())

        # accumulate crawl lengths for this segment
        if is_crawl.any():
            df_c = df_seg[is_crawl].copy()
            L = compute_crawl_length_from_dxdy(df_c)
            if len(L) > 0:
                crawl_lengths_by_state[state].extend(L.tolist())

        # accumulate headcast theta (from CSV) for Von Mises fit
        if is_hc.any():
            th = pd.to_numeric(df_seg.loc[is_hc, THETA_COL], errors="coerce").dropna().to_numpy(float)
            if len(th) > 0:
                headcast_theta_by_state[state].extend(th.tolist())

        # Head casts (rate + direction + theta series FROM CSV)
        df_hc = df_seg[is_hc].copy().sort_values(T0_COL)

        seg_dur = float(tf - t0) if np.isfinite(tf - t0) else np.nan
        n_hc = int(len(df_hc))
        hc_rate = (n_hc / seg_dur) if (np.isfinite(seg_dur) and seg_dur > 1e-9) else np.nan

        # direction distribution
        phi_hc = np.array([], float)
        if n_hc > 0:
            phi_full = compute_phi_from_dxdy(df_hc)
            phi_hc = phi_full[np.isfinite(phi_full)]
            headcast_dirs_by_state.setdefault(state, [])
            headcast_dirs_by_state[state].extend(phi_hc.tolist())

        # theta series for head casts: USE THETA COLUMN FROM CSV
        theta_hc_csv = pd.to_numeric(df_hc[THETA_COL], errors="coerce").dropna().to_numpy(float)

        hc_seg_rows.append({
            "segment_id": seg_id,
            ID_COL: animal,
            "state": state,
            "t0_seg": t0,
            "tf_seg": tf,
            "seg_duration_s": seg_dur,
            "n_headcasts": n_hc,
            "hc_rate_hz": hc_rate,
            "n_hc_dir_valid": int(len(phi_hc)),
            "n_theta_hc_csv": int(len(theta_hc_csv)),
        })

        # Steering theta series (crawls): theta_geom within segment
        df_crawl = df_seg[is_crawl].copy().sort_values(T0_COL)

        theta_steer = np.array([], float)
        if len(df_crawl) >= 3:
            th_full = compute_theta_geom_from_dxdy(df_crawl)
            if th_full is not None:
                theta_steer = th_full[np.isfinite(th_full)]

        # Circular autocorr per segment
        if len(theta_steer) >= max(MAX_LAG + 2, MIN_EVENTS_PER_SEG_AUTOCORR):
            corr, cnt = circular_autocorr_with_counts(theta_steer, MAX_LAG)
            for lag in range(1, MAX_LAG + 1):
                if np.isfinite(corr[lag]) and cnt[lag] > 0:
                    autocorr_accum_rows.append({
                        "state": state,
                        "kind": "steering",
                        "animal": animal,
                        "lag": lag,
                        "seg_corr": float(corr[lag]),
                        "seg_cnt": int(cnt[lag]),
                    })

        if len(theta_hc_csv) >= max(MAX_LAG + 2, MIN_EVENTS_PER_SEG_AUTOCORR):
            corr, cnt = circular_autocorr_with_counts(theta_hc_csv, MAX_LAG)
            for lag in range(1, MAX_LAG + 1):
                if np.isfinite(corr[lag]) and cnt[lag] > 0:
                    autocorr_accum_rows.append({
                        "state": state,
                        "kind": "headcast_theta_csv",
                        "animal": animal,
                        "lag": lag,
                        "seg_corr": float(corr[lag]),
                        "seg_cnt": int(cnt[lag]),
                    })

        # Steering fit (AR2) inside segment using crawl runs
        df_seg2 = segment_runs_within_segment(df_seg)
        crawls = df_seg2[df_seg2["run_id_in_seg"].notna()].copy()
        if len(crawls) == 0:
            continue
        crawls["run_id_in_seg"] = crawls["run_id_in_seg"].astype(int)

        run_fits_this_seg = []

        for run_id, g in crawls.groupby("run_id_in_seg"):
            g = g.sort_values(T0_COL).copy()

            theta_geom_full = compute_theta_geom_from_dxdy(g)
            if theta_geom_full is None:
                continue

            g["theta_fit"] = theta_geom_full
            gg = g.dropna(subset=["theta_fit"]).copy()
            if len(gg) < MIN_POINTS_PER_RUN:
                continue

            theta = gg["theta_fit"].to_numpy(float)

            if CLIP_OUTLIERS and len(theta) >= 10:
                p_lo, p_hi = np.percentile(theta, CLIP_PCTS)
                theta = np.clip(theta, p_lo, p_hi)

            fit = fit_ar2(theta)
            if fit is None:
                continue

            a1, a2, c, sigma, resid, yhat = fit
            rho, kappa, mu, sigma = ar2_to_params(a1, a2, c, sigma)

            ok = True
            if REQUIRE_STABLE:
                if not (RHO_MIN < rho < RHO_MAX):
                    ok = False
                if not (kappa > KAPPA_MIN):
                    ok = False
                if not np.isfinite(mu):
                    ok = False
                if (MU_MAX is not None) and np.isfinite(mu) and (abs(mu) > MU_MAX):
                    ok = False
            if not ok:
                continue

            row = {
                "segment_id": seg_id,
                ID_COL: animal,
                "state": state,
                "t0_seg": t0,
                "tf_seg": tf,
                "run_id_in_seg": int(run_id),
                "n_points": int(len(theta)),
                "rho": float(rho),
                "kappa": float(kappa),
                "mu": float(mu),
                "sigma": float(sigma),
                "a1": float(a1),
                "a2": float(a2),
                "c": float(c),
                "theta_mean": float(np.mean(theta)),
                "theta_std": float(np.std(theta)),
                "t0_run": float(pd.to_numeric(gg[T0_COL], errors="coerce").min()),
                "tf_run": float(pd.to_numeric(gg[T0_COL], errors="coerce").max()),
            }
            run_rows.append(row)
            run_fits_this_seg.append(row)

        if len(run_fits_this_seg) == 0:
            continue

        rf = pd.DataFrame(run_fits_this_seg)
        w = pd.to_numeric(rf["n_points"], errors="coerce").to_numpy(float)

        seg_row = {
            "segment_id": seg_id,
            ID_COL: animal,
            "state": state,
            "t0_seg": t0,
            "tf_seg": tf,
            "n_events_in_seg": int(len(df_seg)),
            "n_runs_fit": int(len(rf)),
            "n_points_total": int(np.nansum(w[np.isfinite(w)])),
        }

        for p in ["rho", "kappa", "mu", "sigma"]:
            x = pd.to_numeric(rf[p], errors="coerce").to_numpy(float)
            seg_row[p] = weighted_mean(x, w)
            seg_row[f"{p}_std_withinseg"] = weighted_std(x, w)

        seg_rows.append(seg_row)

    # Build autocorr_by_state
    df_state_ac = aggregate_autocorr_like_reference(autocorr_accum_rows, MAX_LAG)

    # Save outputs
    run_fit = pd.DataFrame(run_rows)
    seg_fit = pd.DataFrame(seg_rows)
    hc_seg = pd.DataFrame(hc_seg_rows)

    run_csv = os.path.join(OUTDIR, "run_fits.csv")
    seg_csv = os.path.join(OUTDIR, "segment_fits.csv")
    hc_csv  = os.path.join(OUTDIR, "headcast_segment_metrics.csv")
    ac_state_csv = os.path.join(OUTDIR, "autocorr_by_state.csv")
    state_csv = os.path.join(OUTDIR, "state_param_summary.csv")

    if len(run_fit) > 0:
        run_fit = run_fit.sort_values(["state", ID_COL, "segment_id", "run_id_in_seg"])
    run_fit.to_csv(run_csv, index=False)

    if len(seg_fit) > 0:
        seg_fit["abs_mu"] = np.abs(pd.to_numeric(seg_fit["mu"], errors="coerce"))
        seg_fit = seg_fit.sort_values(["state", ID_COL, "segment_id"])
    seg_fit.to_csv(seg_csv, index=False)

    if len(hc_seg) > 0:
        hc_seg = hc_seg.sort_values(["state", ID_COL, "segment_id"])
    hc_seg.to_csv(hc_csv, index=False)

    df_state_ac.to_csv(ac_state_csv, index=False)

    # state summaries (NOW includes peaks)
    if len(seg_fit) > 0:
        st = state_param_summary_table_weighted(seg_fit, WEIGHT_COL_SEG)
    else:
        st = pd.DataFrame()
    st.to_csv(state_csv, index=False)

    # Build simulation_parameters.csv (state summary + extras)
    sim_rows = []
    if len(st) > 0:
        for _, row in st.iterrows():
            state = row["state"]

            # crawl length stats + PEAK (KDE mode)
            L = np.asarray(crawl_lengths_by_state.get(state, []), float)
            L = L[np.isfinite(L)]
            L_mean = float(np.mean(L)) if len(L) > 0 else np.nan
            L_std  = float(np.std(L))  if len(L) > 1 else np.nan

            if len(L) >= 3:
                wL = np.ones_like(L, dtype=float)
                L_peak = weighted_kde_peak(L, wL, bw_min=1e-3)
            else:
                L_peak = np.nan

            # event-based ratio headcast/crawl
            n_c = int(n_crawl_by_state.get(state, 0))
            n_h = int(n_hc_by_state.get(state, 0))
            p_hc = (float(n_h) / float(n_c)) if n_c > 0 else np.nan

            # headcast autocorr lag=1 from df_state_ac
            ac = df_state_ac[
                (df_state_ac["state"] == state) &
                (df_state_ac["kind"] == "headcast_theta_csv") &
                (df_state_ac["lag"] == 1)
            ]
            if len(ac) > 0:
                hc_corr_mean = float(ac["mean"].values[0])
                hc_corr_std  = float(ac["std"].values[0])
            else:
                hc_corr_mean = np.nan
                hc_corr_std  = np.nan

            # Von Mises fit on headcast theta
            theta_hc = np.asarray(headcast_theta_by_state.get(state, []), float)
            theta_hc = theta_hc[np.isfinite(theta_hc)]
            mu_hc, kappa_hc, R_hc = fit_vonmises(theta_hc)

            sim_row = dict(row)  # includes *_mean_w, *_std_w, *_peak_w
            sim_row.update({
                "crawl_length_mean": L_mean,
                "crawl_length_std": L_std,
                "crawl_length_peak": L_peak,

                "p_headcast_given_crawl": p_hc,

                "headcast_corr_lag1_mean": hc_corr_mean,
                "headcast_corr_lag1_std": hc_corr_std,

                "headcast_theta_mu": float(mu_hc) if np.isfinite(mu_hc) else np.nan,
                "headcast_theta_peak": float(mu_hc) if np.isfinite(mu_hc) else np.nan,  # peak of von mises == mu
                "headcast_theta_kappa": float(kappa_hc) if np.isfinite(kappa_hc) else np.nan,
                "headcast_theta_R": float(R_hc) if np.isfinite(R_hc) else np.nan,

                "n_crawls_in_segments": int(n_c),
                "n_headcasts_in_segments": int(n_h),
                "n_headcast_theta_points": int(len(theta_hc)),
            })
            sim_rows.append(sim_row)

    simulation_df = pd.DataFrame(sim_rows)
    sim_csv = os.path.join(OUTDIR, "simulation_parameters.csv")
    simulation_df.to_csv(sim_csv, index=False)

    # Figures
    if len(seg_fit) > 0:
        plot_param_box_weighted(seg_fit, os.path.join(figdir, "param_box_weighted.svg"), WEIGHT_COL_SEG)
        plot_param_kde_weighted(seg_fit, os.path.join(figdir, "param_kde_weighted.svg"), WEIGHT_COL_SEG)

    plot_headcast_rate_boxplot(hc_seg, os.path.join(figdir, "headcast_rate_boxplot.svg"))
    plot_headcast_direction_polar(headcast_dirs_by_state, os.path.join(figdir, "headcast_direction_polar.svg"), bins=HC_DIR_BINS)

    if len(df_state_ac) > 0:
        plot_autocorr_by_state(df_state_ac, os.path.join(figdir, "autocorr_by_state.svg"))

    # headcast theta Von Mises plots (polar)
    hc_fit_dir = os.path.join(figdir, "headcast_theta_fit")
    os.makedirs(hc_fit_dir, exist_ok=True)

    for state, theta_vals in headcast_theta_by_state.items():
        theta = np.asarray(theta_vals, float)
        theta = theta[np.isfinite(theta)]
        if len(theta) < 5:
            continue

        mu, kappa, R = fit_vonmises(theta)
        if not np.isfinite(mu) or not np.isfinite(kappa):
            continue

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111, projection="polar")

        bins = np.linspace(-np.pi, np.pi, 24+1)
        counts, edges = np.histogram(theta, bins=bins)
        centers = 0.5*(edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        density = counts / (np.sum(counts) * width) if np.sum(counts) > 0 else counts.astype(float)

        ax.bar((centers + 2*np.pi) % (2*np.pi),
               density, width=width, alpha=0.6)

        xs = np.linspace(-np.pi, np.pi, 400)
        ax.plot((xs + 2*np.pi) % (2*np.pi),
                vonmises_pdf(xs, mu, kappa),
                lw=2)

        ax.set_title(f"{state}", fontsize=10)
        ax.set_yticklabels([])
        ax.grid(True, linewidth=0.3)

        plt.tight_layout()
        out_svg = os.path.join(hc_fit_dir, f"{safe_slug(state)}.svg")
        plt.savefig(out_svg)
        if SAVE_PDF_ALSO:
            plt.savefig(os.path.splitext(out_svg)[0] + ".pdf")
        plt.close(fig)

    print("Done.")
    print(f"- segment fits: {seg_csv} ({len(seg_fit)} segments)")
    print(f"- run fits:     {run_csv} ({len(run_fit)} runs)")
    print(f"- headcast seg: {hc_csv}")
    print(f"- autocorr state: {ac_state_csv}")
    print(f"- state summary: {state_csv}")
    print(f"- simulation params: {sim_csv}")
    print(f"- figures: {figdir}/")
    print("")
    print("NOTE: head-cast autocorrelation uses the theta column from event_level_steps.csv.")
    print("NOTE: autocorr aggregation is per-animal (reference style), with no cross-segment pairs.")
    print("NOTE: Peaks: state_param_summary has *_peak_w (weighted KDE mode) and simulation has crawl_length_peak + headcast_theta_peak.")


if __name__ == "__main__":
    main()
