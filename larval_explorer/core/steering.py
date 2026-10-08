"""Stage 11 - AR(2) steering fit on representative HMM segments, by state.

A port of upstream ``11_fit_steering_by_strategy.py`` (D-013). The numerical
helpers are upstream's, unchanged. Upstream's ``main()`` is split from its
plotting and file writing and becomes ``stage_11``; its module-level
configuration is ``SteeringParams``.

One deliberate difference: the HMM segment table is read through the columns
stage 08 actually writes, ``start_time_s`` and ``end_time_s`` (D-014).

Upstream calls a track an "animal". With track segmentation on, an "animal"
here is one track segment, so ``n_animals`` counts track segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from larval_explorer.core import schema as S
from larval_explorer.core.params import SteeringParams
from larval_explorer.core.result import StageResult


# =============================================================================
# Helpers
# =============================================================================

def angle_wrap_pi(x):
    """Wrap to (-pi, pi]."""
    return (x + np.pi) % (2*np.pi) - np.pi


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
    dx = pd.to_numeric(g[S.DX_MM], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(g[S.DY_MM], errors="coerce").to_numpy(float)

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
    d = df_seg.sort_values([S.T_START_S]).copy()
    is_crawl = (d[S.EVENT_TYPE].astype(str).str.lower() == S.EVENT_CRAWL)
    # Upstream writes shift(1).fillna(False), which pandas is deprecating for
    # boolean data; fill_value gives the same values without the dtype detour.
    prev_is_crawl = is_crawl.shift(1, fill_value=False)
    new_run = is_crawl & (~prev_is_crawl)
    d[S.RUN_ID_IN_SEG] = new_run.cumsum()
    d.loc[~is_crawl, S.RUN_ID_IN_SEG] = np.nan
    return d


# =============================================================================
# Head-cast helpers
# =============================================================================

def is_head_cast_series(tipo_series: pd.Series) -> pd.Series:
    """
    Heuristic: contains 'head' or contains 'cast', but not exactly 'crawl'.
    """
    s = tipo_series.astype(str).str.lower()
    is_crawl = (s == S.EVENT_CRAWL)
    is_hc = (s.str.contains("head", regex=False) | s.str.contains("cast", regex=False)) & (~is_crawl)
    return is_hc


def compute_phi_from_dxdy(df: pd.DataFrame):
    dx = pd.to_numeric(df[S.DX_MM], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(df[S.DY_MM], errors="coerce").to_numpy(float)
    step2 = dx*dx + dy*dy
    ok = np.isfinite(dx) & np.isfinite(dy) & (step2 > 1e-12)
    phi = np.full(len(df), np.nan, float)
    phi[ok] = np.arctan2(dy[ok], dx[ok])
    return phi


# =============================================================================
# Circular autocorrelation (per segment) + aggregation like the reference
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
    exactly like the reference script's style.
    """
    if len(ac_rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(ac_rows)

    out_state = []
    lags = np.arange(1, max_lag + 1)

    for (state, kind), gsk in df.groupby([S.STATE, S.KIND]):
        # build per-animal dicts
        animal_dicts = []
        for animal, ga in gsk.groupby(S.ANIMAL):
            # accumulate numerator/denominator over segments for each lag
            num = {lag: 0.0 for lag in lags}
            den = {lag: 0.0 for lag in lags}
            for _, r in ga.iterrows():
                lag = int(r[S.LAG])
                v = float(r[S.SEG_CORR])
                c = float(r[S.SEG_CNT])
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
                S.STATE: state,
                S.KIND: kind,
                S.LAG: int(lag),
                S.MEAN: mean,
                S.STD: std,
                S.N_ANIMALS: int(n),
            })

    return pd.DataFrame(out_state).sort_values([S.KIND, S.STATE, S.LAG])


# =============================================================================
# Simulation extras: crawl length + vonmises headcast theta fit
# =============================================================================

def compute_crawl_length_from_dxdy(df_crawl: pd.DataFrame):
    dx = pd.to_numeric(df_crawl[S.DX_MM], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(df_crawl[S.DY_MM], errors="coerce").to_numpy(float)
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
    S_ = float(np.mean(np.sin(theta)))
    R = float(np.sqrt(C*C + S_*S_))
    mu = float(np.arctan2(S_, C))

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


def state_param_summary_table_weighted(seg_fit: pd.DataFrame, weight_col: str, params: SteeringParams) -> pd.DataFrame:
    """
    Builds the state parameter summary:
      - mean/std (weighted)
      - PEAK (mode) from weighted KDE: *_peak_w
    """
    if len(seg_fit) == 0:
        return pd.DataFrame()

    df = seg_fit.copy()
    df[S.STATE] = df[S.STATE].astype(str)

    for p in [*S.STEERING_SUMMARY_PARAMETERS, weight_col]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    out_rows = []
    for est, sub in df.groupby(S.STATE):
        w = sub[weight_col].to_numpy(float)
        row = {
            S.STATE: est,
            S.N_SEGMENTS: int(len(sub)),
            S.WEIGHT_SUM: float(np.nansum(w[np.isfinite(w)])),
        }
        for p in S.STEERING_SUMMARY_PARAMETERS:
            if p not in sub.columns:
                row[S.weighted_mean_column(p)] = np.nan
                row[S.weighted_std_column(p)]  = np.nan
                row[S.weighted_peak_column(p)] = np.nan
                continue

            x = pd.to_numeric(sub[p], errors="coerce").to_numpy(float)
            row[S.weighted_mean_column(p)] = weighted_mean(x, w)
            row[S.weighted_std_column(p)]  = weighted_std(x, w)
            row[S.weighted_peak_column(p)] = weighted_kde_peak(x, w, params.kde_bw_min.get(p, 1e-3))
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(S.STATE)


# =============================================================================
# Stage
# =============================================================================

def stage_11(event_steps: pd.DataFrame, hmm_segments: pd.DataFrame, params: SteeringParams) -> StageResult:
    """Fit the steering model inside each representative HMM segment.

    ``event_steps`` is the event-level step table from stage 10 and
    ``hmm_segments`` the segment table from stage 08.

    Tables: ``run_fits``, ``segment_fits``, ``headcast_segment_metrics``,
    ``autocorr_by_state``, ``state_param_summary``, ``simulation_parameters``
    (upstream's six CSVs), plus three long tables the figures are drawn from:
    ``headcast_directions``, ``headcast_thetas`` and ``crawl_lengths``.
    """
    # A recording with no detected events, or no HMM segments, is not an error:
    # the tables come out empty, with a warning.
    if event_steps.empty:
        event_steps = pd.DataFrame(columns=list(S.EVENT_LEVEL_STEPS.required))
    if hmm_segments.empty:
        hmm_segments = pd.DataFrame(columns=list(S.HMM_SEGMENTS.required))
    S.EVENT_LEVEL_STEPS.validate(event_steps)
    S.HMM_SEGMENTS.validate(hmm_segments)
    no_events = event_steps.empty
    max_lag = params.max_lag
    low = event_steps.copy()
    seg = hmm_segments.copy()

    # numeric casts
    low[S.T_START_S] = pd.to_numeric(low[S.T_START_S], errors="coerce")
    low[S.T_END_S] = pd.to_numeric(low[S.T_END_S], errors="coerce")
    low[S.THETA] = pd.to_numeric(low[S.THETA], errors="coerce")
    seg[S.START_TIME_S] = pd.to_numeric(seg[S.START_TIME_S], errors="coerce")
    seg[S.END_TIME_S] = pd.to_numeric(seg[S.END_TIME_S], errors="coerce")

    low = low.dropna(subset=[S.TRACK_ID, S.T_START_S, S.T_END_S, S.EVENT_TYPE]).copy()
    seg = seg.dropna(subset=[S.TRACK_ID, S.STATE, S.START_TIME_S, S.END_TIME_S]).copy()
    low = low.sort_values([S.TRACK_ID, S.T_START_S]).copy()

    run_rows = []
    seg_rows = []
    hc_seg_rows = []
    headcast_dirs_by_state = {}

    # accumulators for simulation params
    crawl_lengths_by_state = {}     # state -> list of lengths
    n_crawl_by_state = {}           # state -> total crawl events (within segments)
    n_hc_by_state = {}              # state -> total headcast events (within segments)
    headcast_theta_by_state = {}    # state -> list of theta for headcasts (within segments)

    # AUTOCORR accumulation rows
    autocorr_accum_rows = []  # state, kind, animal, lag, seg_corr, seg_cnt

    # stable segment_id
    seg = seg.reset_index(drop=True)
    seg[S.SEGMENT_ID] = np.arange(len(seg), dtype=int)

    for _, srow in seg.iterrows():
        seg_id = int(srow[S.SEGMENT_ID])
        animal = srow[S.TRACK_ID]
        state = str(srow[S.STATE])
        t0 = float(srow[S.START_TIME_S])
        tf = float(srow[S.END_TIME_S])

        # init state buckets
        crawl_lengths_by_state.setdefault(state, [])
        n_crawl_by_state.setdefault(state, 0)
        n_hc_by_state.setdefault(state, 0)
        headcast_theta_by_state.setdefault(state, [])

        df_seg = low[(low[S.TRACK_ID] == animal) & (low[S.T_START_S] >= t0) & (low[S.T_START_S] <= tf)].copy()
        if len(df_seg) < params.min_events_per_segment:
            continue
        df_seg = df_seg.sort_values(S.T_START_S).copy()

        # Identify crawls and headcasts inside THIS segment
        tipo_lower = df_seg[S.EVENT_TYPE].astype(str).str.lower()
        is_crawl = tipo_lower.eq(S.EVENT_CRAWL)
        is_hc = is_head_cast_series(df_seg[S.EVENT_TYPE])

        # accumulate counts for simulation ratio
        n_crawl_by_state[state] += int(is_crawl.sum())
        n_hc_by_state[state] += int(is_hc.sum())

        # accumulate crawl lengths for this segment
        if is_crawl.any():
            df_c = df_seg[is_crawl].copy()
            L = compute_crawl_length_from_dxdy(df_c)
            if len(L) > 0:
                crawl_lengths_by_state[state].extend(L.tolist())

        # accumulate headcast theta for Von Mises fit
        if is_hc.any():
            th = pd.to_numeric(df_seg.loc[is_hc, S.THETA], errors="coerce").dropna().to_numpy(float)
            if len(th) > 0:
                headcast_theta_by_state[state].extend(th.tolist())

        # Head casts (rate + direction + theta series)
        df_hc = df_seg[is_hc].copy().sort_values(S.T_START_S)

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

        # theta series for head casts: the theta column of the event-level steps
        theta_hc_csv = pd.to_numeric(df_hc[S.THETA], errors="coerce").dropna().to_numpy(float)

        hc_seg_rows.append({
            S.SEGMENT_ID: seg_id,
            S.TRACK_ID: animal,
            S.STATE: state,
            S.T0_SEG: t0,
            S.TF_SEG: tf,
            S.SEG_DURATION_S: seg_dur,
            S.N_HEADCASTS: n_hc,
            S.HC_RATE_HZ: hc_rate,
            S.N_HC_DIR_VALID: int(len(phi_hc)),
            S.N_THETA_HC_CSV: int(len(theta_hc_csv)),
        })

        # Steering theta series (crawls): theta_geom within segment
        df_crawl = df_seg[is_crawl].copy().sort_values(S.T_START_S)

        theta_steer = np.array([], float)
        if len(df_crawl) >= 3:
            th_full = compute_theta_geom_from_dxdy(df_crawl)
            if th_full is not None:
                theta_steer = th_full[np.isfinite(th_full)]

        # Circular autocorr per segment
        if len(theta_steer) >= max(max_lag + 2, params.min_events_per_seg_autocorr):
            corr, cnt = circular_autocorr_with_counts(theta_steer, max_lag)
            for lag in range(1, max_lag + 1):
                if np.isfinite(corr[lag]) and cnt[lag] > 0:
                    autocorr_accum_rows.append({
                        S.STATE: state,
                        S.KIND: S.AUTOCORR_STEERING,
                        S.ANIMAL: animal,
                        S.LAG: lag,
                        S.SEG_CORR: float(corr[lag]),
                        S.SEG_CNT: int(cnt[lag]),
                    })

        if len(theta_hc_csv) >= max(max_lag + 2, params.min_events_per_seg_autocorr):
            corr, cnt = circular_autocorr_with_counts(theta_hc_csv, max_lag)
            for lag in range(1, max_lag + 1):
                if np.isfinite(corr[lag]) and cnt[lag] > 0:
                    autocorr_accum_rows.append({
                        S.STATE: state,
                        S.KIND: S.AUTOCORR_HEADCAST,
                        S.ANIMAL: animal,
                        S.LAG: lag,
                        S.SEG_CORR: float(corr[lag]),
                        S.SEG_CNT: int(cnt[lag]),
                    })

        # Steering fit (AR2) inside segment using crawl runs
        df_seg2 = segment_runs_within_segment(df_seg)
        crawls = df_seg2[df_seg2[S.RUN_ID_IN_SEG].notna()].copy()
        if len(crawls) == 0:
            continue
        crawls[S.RUN_ID_IN_SEG] = crawls[S.RUN_ID_IN_SEG].astype(int)

        run_fits_this_seg = []

        for run_id, g in crawls.groupby(S.RUN_ID_IN_SEG):
            g = g.sort_values(S.T_START_S).copy()

            theta_geom_full = compute_theta_geom_from_dxdy(g)
            if theta_geom_full is None:
                continue

            theta_fit = pd.Series(theta_geom_full, index=g.index)
            gg = g[theta_fit.notna()].copy()
            if len(gg) < params.min_points_per_run:
                continue

            theta = theta_fit[theta_fit.notna()].to_numpy(float)

            if params.clip_outliers and len(theta) >= params.clip_min_points:
                p_lo, p_hi = np.percentile(theta, params.clip_pcts)
                theta = np.clip(theta, p_lo, p_hi)

            fit = fit_ar2(theta)
            if fit is None:
                continue

            a1, a2, c, sigma, resid, yhat = fit
            rho, kappa, mu, sigma = ar2_to_params(a1, a2, c, sigma)

            ok = True
            if params.require_stable:
                if not (params.rho_min < rho < params.rho_max):
                    ok = False
                if not (kappa > params.kappa_min):
                    ok = False
                if not np.isfinite(mu):
                    ok = False
                if (params.mu_max is not None) and np.isfinite(mu) and (abs(mu) > params.mu_max):
                    ok = False
            if not ok:
                continue

            row = {
                S.SEGMENT_ID: seg_id,
                S.TRACK_ID: animal,
                S.STATE: state,
                S.T0_SEG: t0,
                S.TF_SEG: tf,
                S.RUN_ID_IN_SEG: int(run_id),
                S.N_POINTS: int(len(theta)),
                S.RHO: float(rho),
                S.KAPPA: float(kappa),
                S.MU: float(mu),
                S.SIGMA: float(sigma),
                S.AR_A1: float(a1),
                S.AR_A2: float(a2),
                S.AR_C: float(c),
                S.THETA_MEAN: float(np.mean(theta)),
                S.THETA_STD: float(np.std(theta)),
                S.T0_RUN: float(pd.to_numeric(gg[S.T_START_S], errors="coerce").min()),
                S.TF_RUN: float(pd.to_numeric(gg[S.T_START_S], errors="coerce").max()),
            }
            run_rows.append(row)
            run_fits_this_seg.append(row)

        if len(run_fits_this_seg) == 0:
            continue

        rf = pd.DataFrame(run_fits_this_seg)
        w = pd.to_numeric(rf[S.N_POINTS], errors="coerce").to_numpy(float)

        seg_row = {
            S.SEGMENT_ID: seg_id,
            S.TRACK_ID: animal,
            S.STATE: state,
            S.T0_SEG: t0,
            S.TF_SEG: tf,
            S.N_EVENTS_IN_SEG: int(len(df_seg)),
            S.N_RUNS_FIT: int(len(rf)),
            S.N_POINTS_TOTAL: int(np.nansum(w[np.isfinite(w)])),
        }

        for p in S.STEERING_PARAMETERS:
            x = pd.to_numeric(rf[p], errors="coerce").to_numpy(float)
            seg_row[p] = weighted_mean(x, w)
            seg_row[S.within_segment_std(p)] = weighted_std(x, w)

        seg_rows.append(seg_row)

    # Build autocorr_by_state
    df_state_ac = aggregate_autocorr_like_reference(autocorr_accum_rows, max_lag)

    run_fit = pd.DataFrame(run_rows)
    seg_fit = pd.DataFrame(seg_rows)
    hc_seg = pd.DataFrame(hc_seg_rows)

    if len(run_fit) > 0:
        run_fit = run_fit.sort_values([S.STATE, S.TRACK_ID, S.SEGMENT_ID, S.RUN_ID_IN_SEG])

    if len(seg_fit) > 0:
        seg_fit[S.ABS_MU] = np.abs(pd.to_numeric(seg_fit[S.MU], errors="coerce"))
        seg_fit = seg_fit.sort_values([S.STATE, S.TRACK_ID, S.SEGMENT_ID])

    if len(hc_seg) > 0:
        hc_seg = hc_seg.sort_values([S.STATE, S.TRACK_ID, S.SEGMENT_ID])

    # state summaries (includes peaks)
    if len(seg_fit) > 0:
        st = state_param_summary_table_weighted(seg_fit, S.N_POINTS_TOTAL, params)
    else:
        st = pd.DataFrame()

    # Build simulation parameters (state summary + extras)
    sim_rows = []
    if len(st) > 0:
        for _, row in st.iterrows():
            state = row[S.STATE]

            # crawl length stats + PEAK (KDE mode)
            L = np.asarray(crawl_lengths_by_state.get(state, []), float)
            L = L[np.isfinite(L)]
            L_mean = float(np.mean(L)) if len(L) > 0 else np.nan
            L_std  = float(np.std(L))  if len(L) > 1 else np.nan

            if len(L) >= 3:
                wL = np.ones_like(L, dtype=float)
                L_peak = weighted_kde_peak(L, wL, bw_min=params.crawl_length_kde_bw_min)
            else:
                L_peak = np.nan

            # event-based ratio headcast/crawl
            n_c = int(n_crawl_by_state.get(state, 0))
            n_h = int(n_hc_by_state.get(state, 0))
            p_hc = (float(n_h) / float(n_c)) if n_c > 0 else np.nan

            # headcast autocorr lag=1 from df_state_ac
            if len(df_state_ac) > 0:
                ac = df_state_ac[
                    (df_state_ac[S.STATE] == state) &
                    (df_state_ac[S.KIND] == S.AUTOCORR_HEADCAST) &
                    (df_state_ac[S.LAG] == 1)
                ]
            else:
                ac = df_state_ac
            if len(ac) > 0:
                hc_corr_mean = float(ac[S.MEAN].values[0])
                hc_corr_std  = float(ac[S.STD].values[0])
            else:
                hc_corr_mean = np.nan
                hc_corr_std  = np.nan

            # Von Mises fit on headcast theta
            theta_hc = np.asarray(headcast_theta_by_state.get(state, []), float)
            theta_hc = theta_hc[np.isfinite(theta_hc)]
            mu_hc, kappa_hc, R_hc = fit_vonmises(theta_hc)

            sim_row = dict(row)  # includes *_mean_w, *_std_w, *_peak_w
            sim_row.update({
                S.CRAWL_LENGTH_MEAN: L_mean,
                S.CRAWL_LENGTH_STD: L_std,
                S.CRAWL_LENGTH_PEAK: L_peak,

                S.P_HEADCAST_GIVEN_CRAWL: p_hc,

                S.HEADCAST_CORR_LAG1_MEAN: hc_corr_mean,
                S.HEADCAST_CORR_LAG1_STD: hc_corr_std,

                S.HEADCAST_THETA_MU: float(mu_hc) if np.isfinite(mu_hc) else np.nan,
                S.HEADCAST_THETA_PEAK: float(mu_hc) if np.isfinite(mu_hc) else np.nan,  # peak of von mises == mu
                S.HEADCAST_THETA_KAPPA: float(kappa_hc) if np.isfinite(kappa_hc) else np.nan,
                S.HEADCAST_THETA_R: float(R_hc) if np.isfinite(R_hc) else np.nan,

                S.N_CRAWLS_IN_SEGMENTS: int(n_c),
                S.N_HEADCASTS_IN_SEGMENTS: int(n_h),
                S.N_HEADCAST_THETA_POINTS: int(len(theta_hc)),
            })
            sim_rows.append(sim_row)

    simulation_df = pd.DataFrame(sim_rows)

    def long_table(values_by_state: dict, value_column: str) -> pd.DataFrame:
        rows = [(state, value) for state, values in values_by_state.items() for value in values]
        return pd.DataFrame(rows, columns=[S.STATE, value_column])

    tables = {
        "run_fits": run_fit.reset_index(drop=True),
        "segment_fits": seg_fit.reset_index(drop=True),
        "headcast_segment_metrics": hc_seg.reset_index(drop=True),
        "autocorr_by_state": df_state_ac.reset_index(drop=True),
        "state_param_summary": st.reset_index(drop=True),
        "simulation_parameters": simulation_df,
        "headcast_directions": long_table(headcast_dirs_by_state, S.HEADING),
        "headcast_thetas": long_table(headcast_theta_by_state, S.THETA),
        "crawl_lengths": long_table(crawl_lengths_by_state, S.CRAWL_LENGTH),
    }

    warnings = []
    if no_events:
        warnings.append("No crawls or head casts were detected: nothing to fit.")
    elif len(seg) == 0:
        warnings.append("No representative HMM segments: nothing to fit.")
    elif len(seg_fit) == 0:
        warnings.append("No HMM segment produced a stable steering fit.")
    diagnostics = {
        "hmm_segments_in": len(seg),
        "hmm_segments_with_events": len(hc_seg),
        "hmm_segments_fitted": len(seg_fit),
        "runs_fitted": len(run_fit),
        "states_fitted": sorted(seg_fit[S.STATE].unique().tolist()) if len(seg_fit) else [],
    }
    return StageResult(tables, diagnostics, tuple(warnings))
