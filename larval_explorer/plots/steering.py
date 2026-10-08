"""Stage 11 figures: the steering fit by HMM state.

The first six are upstream's figures. Upstream took series colours from
matplotlib's cycle in plot order; here a state's colour comes from its stored
index, so it is the same in every figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from larval_explorer.core import schema as S
from larval_explorer.core import steering as fit
from larval_explorer.core.params import SteeringParams
from larval_explorer.plots import style

JITTER_SEED = 0
PARAMETER_TITLES = {S.RHO: "rho", S.KAPPA: "kappa", S.MU: "mu", S.ABS_MU: "|mu|", S.SIGMA: "sigma"}


def _state_order(table: pd.DataFrame) -> list[str]:
    return sorted(table[S.STATE].dropna().astype(str).unique().tolist())


def _state_handles(states) -> list[Patch]:
    return [Patch(facecolor=style.state_color(state), label=style.state_label(state)) for state in states]


def _draw_segment(ax, steps: pd.DataFrame, recenter: bool = True) -> bool:
    """One HMM segment as event-level steps: crawls dark, head casts red (upstream figure)."""
    g = steps.sort_values(S.T_START_S)
    coordinates = g[[S.X0_MM, S.Y0_MM, S.X1_MM, S.Y1_MM]].apply(pd.to_numeric, errors="coerce")
    g = g[np.isfinite(coordinates).all(axis=1)]
    if len(g) < 2:
        return False
    x0, y0 = g[S.X0_MM].to_numpy(float), g[S.Y0_MM].to_numpy(float)
    x1, y1 = g[S.X1_MM].to_numpy(float), g[S.Y1_MM].to_numpy(float)
    if recenter:
        origin_x, origin_y = x0[0], y0[0]
        x0, x1, y0, y1 = x0 - origin_x, x1 - origin_x, y0 - origin_y, y1 - origin_y
    is_crawl = (g[S.EVENT_TYPE].astype(str).str.lower() == S.EVENT_CRAWL).to_numpy()
    for mask, color in ((is_crawl, style.CRAWL_COLOR), (~is_crawl, style.HEAD_CAST_COLOR)):
        xs = np.column_stack([x0[mask], x1[mask], np.full(mask.sum(), np.nan)]).ravel()
        ys = np.column_stack([y0[mask], y1[mask], np.full(mask.sum(), np.nan)]).ravel()
        ax.plot(xs, ys, color=color, linewidth=1.3, alpha=0.95)
    ax.plot(x0[0], y0[0], "o", color=style.INK_SECONDARY, markersize=4)
    ax.plot(x1[-1], y1[-1], "s", color=style.INK_SECONDARY, markersize=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return True


def _segment_steps(event_steps: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
    return event_steps[
        (event_steps[S.TRACK_ID] == segment[S.TRACK_ID])
        & (event_steps[S.T_START_S] >= segment[S.START_TIME_S])
        & (event_steps[S.T_START_S] <= segment[S.END_TIME_S])
    ]


_SEGMENT_HANDLES = [
    Line2D([], [], color=style.CRAWL_COLOR, linewidth=1.3, label="crawl"),
    Line2D([], [], color=style.HEAD_CAST_COLOR, linewidth=1.3, label="head cast"),
    Line2D([], [], color=style.INK_SECONDARY, marker="o", linestyle="", markersize=4, label="start"),
    Line2D([], [], color=style.INK_SECONDARY, marker="s", linestyle="", markersize=4, label="end"),
]


def segment_trajectory(event_steps: pd.DataFrame, segment: pd.Series) -> Figure:
    """One representative HMM segment, recentred on its first step."""
    fig, axes = style.new_figure(3.6, 3.6)
    ax = axes[0, 0]
    if not _draw_segment(ax, _segment_steps(event_steps, segment)):
        return style.message_figure("Fewer than two events in this HMM segment.", 3.6, 3.0)
    ax.set_title(f"{style.state_label(segment[S.STATE])} | {segment[S.TRACK_ID]}\n"
                 f"[{segment[S.START_TIME_S]:.2f}, {segment[S.END_TIME_S]:.2f}] s",
                 fontsize=style.TICK_SIZE, color=style.INK)
    style.legend(fig, handles=_SEGMENT_HANDLES, loc="outside lower center", ncols=4)
    return fig


def segment_gallery(event_steps: pd.DataFrame, hmm_segments: pd.DataFrame, max_per_state: int = 12) -> Figure:
    """Representative HMM segments side by side, one row per state, longest first."""
    if hmm_segments.empty:
        return style.message_figure("No representative HMM segments to show.")
    states = sorted(int(state) for state in hmm_segments[S.STATE].unique())
    by_state = {state: hmm_segments[hmm_segments[S.STATE] == state].sort_values(S.N_STEPS, ascending=False)
                for state in states}
    columns = min(max_per_state, max(len(group) for group in by_state.values()))
    fig, axes = style.new_figure(1.9 * columns + 0.6, 2.1 * len(states) + 0.5, len(states), columns)
    for row, state in enumerate(states):
        group = by_state[state]
        for column in range(columns):
            ax = axes[row, column]
            if column >= len(group) or not _draw_segment(ax, _segment_steps(event_steps, group.iloc[column])):
                ax.set_axis_off()
                continue
            segment = group.iloc[column]
            ax.set_title(f"{style.short_track(segment[S.TRACK_ID])}\n{segment[S.N_STEPS]} steps", fontsize=style.TICK_SIZE - 1,
                         color=style.INK_SECONDARY)
        shown = min(columns, len(group))
        axes[row, 0].annotate(f"{style.state_label(state)} ({shown} of {len(group)})", (0, 1.28),
                              xycoords="axes fraction", fontsize=style.LABEL_SIZE, color=style.state_color(state),
                              fontweight="bold")
    style.legend(fig, handles=_SEGMENT_HANDLES, loc="outside lower center", ncols=4)
    return fig


def parameter_boxes(segment_fits: pd.DataFrame) -> Figure:
    """Weighted box summaries of rho, kappa, mu, sigma by state (upstream figure).

    Box = weighted quartiles, whiskers = weighted 5th and 95th percentiles,
    points = HMM segments sized by weight (points fitted).
    """
    if segment_fits.empty:
        return style.message_figure("No steering fits to show.")
    df = segment_fits.copy()
    df[S.STATE] = df[S.STATE].astype(str)
    state_order = _state_order(df)
    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = style.new_figure(13.0, 3.3, 1, 4)
    for ax, p in zip(axes[0], S.STEERING_PARAMETERS):
        style.style_axes(ax, grid="y")
        for i, est in enumerate(state_order, start=1):
            sub = df[df[S.STATE] == est]
            x = pd.to_numeric(sub[p], errors="coerce").to_numpy(float)
            w = pd.to_numeric(sub[S.N_POINTS_TOTAL], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            x, w = x[ok], w[ok]
            if len(x) == 0:
                continue
            color = style.state_color(est)
            q05, q25, q50, q75, q95 = (fit.weighted_quantile(x, w, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95))
            ax.plot([i, i], [q05, q95], linewidth=1.2, color=color)
            ax.plot([i - 0.18, i + 0.18], [q05, q05], linewidth=1.2, color=color)
            ax.plot([i - 0.18, i + 0.18], [q95, q95], linewidth=1.2, color=color)
            ax.plot([i - 0.25, i + 0.25, i + 0.25, i - 0.25, i - 0.25], [q25, q25, q75, q75, q25], linewidth=1.2, color=color)
            ax.plot([i - 0.25, i + 0.25], [q50, q50], linewidth=1.8, color=color)
            ws = np.sqrt(w / np.nanmedian(w)) if np.nanmedian(w) > 0 else np.ones_like(w)
            ws = np.clip(ws, 0.6, 3.0)
            ax.scatter(i + 0.08 * rng.standard_normal(len(x)), x, s=10 * ws, alpha=0.45, color=color, linewidths=0)
        ax.set_xticks(range(1, len(state_order) + 1), [f"{style.state_label(est)}\nn = {(df[S.STATE] == est).sum()}" for est in state_order])
        ax.set_xlim(0.4, len(state_order) + 0.6)
        style.label(ax, title=PARAMETER_TITLES[p])
    fig.suptitle("Weighted box summaries by state (weight = points fitted per HMM segment)",
                 fontsize=style.TITLE_SIZE, color=style.INK, x=0.01, ha="left")
    return fig


def parameter_kdes(segment_fits: pd.DataFrame, params: SteeringParams) -> Figure:
    """Weighted kernel densities of rho, kappa, |mu|, sigma by state (upstream figure)."""
    if segment_fits.empty:
        return style.message_figure("No steering fits to show.")
    df = segment_fits.copy()
    df[S.STATE] = df[S.STATE].astype(str)
    state_order = _state_order(df)
    fig, axes = style.new_figure(13.0, 3.3, 1, 4)
    drawn = set()
    for ax, p in zip(axes[0], (S.RHO, S.KAPPA, S.ABS_MU, S.SIGMA)):
        style.style_axes(ax, grid="y")
        by_state = {}
        for est in state_order:
            sub = df[df[S.STATE] == est]
            x = pd.to_numeric(sub[p], errors="coerce").to_numpy(float)
            w = pd.to_numeric(sub[S.N_POINTS_TOTAL], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            if ok.sum() >= 2:
                by_state[est] = (x[ok], w[ok])
        if not by_state:
            style.label(ax, title=f"{PARAMETER_TITLES[p]} (fewer than 2 segments per state)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        x_all = np.concatenate([x for x, _ in by_state.values()])
        w_all = np.concatenate([w for _, w in by_state.values()])
        x_lo, x_hi = fit.weighted_quantile(x_all, w_all, 0.01), fit.weighted_quantile(x_all, w_all, 0.99)
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            x_lo, x_hi = np.nanmin(x_all), np.nanmax(x_all)
        pad = 0.06 * (x_hi - x_lo) if x_hi > x_lo else 1.0
        xs = np.linspace(x_lo - pad, x_hi + pad, 350)
        for est, (x, w) in by_state.items():
            dens = fit.weighted_kde_gaussian(xs, x, w, bw=None, bw_min=params.kde_bw_min.get(p, 1e-3))
            ax.plot(xs, dens, linewidth=1.6, color=style.state_color(est))
            drawn.add(est)
        style.label(ax, title=PARAMETER_TITLES[p], y="density" if p == S.RHO else None)
    if drawn:
        style.legend(fig, handles=_state_handles(sorted(drawn)), loc="outside upper right", ncols=len(drawn))
    fig.suptitle("Weighted KDE by state (weight = points fitted per HMM segment)",
                 fontsize=style.TITLE_SIZE, color=style.INK, x=0.01, ha="left")
    return fig


def headcast_rate(headcast_segment_metrics: pd.DataFrame) -> Figure:
    """Head casts per second in each HMM segment, by state (upstream figure)."""
    if headcast_segment_metrics.empty:
        return style.message_figure("No head-cast metrics to show.")
    df = headcast_segment_metrics.copy()
    df[S.STATE] = df[S.STATE].astype(str)
    states = _state_order(df)
    data = []
    for state in states:
        values = pd.to_numeric(df.loc[df[S.STATE] == state, S.HC_RATE_HZ], errors="coerce").to_numpy(float)
        data.append(values[np.isfinite(values)])
    fig, axes = style.new_figure(5.6, 3.3)
    ax = axes[0, 0]
    style.style_axes(ax, grid="y")
    rng = np.random.default_rng(JITTER_SEED)
    boxes = ax.boxplot(data, showfliers=False, widths=0.5, patch_artist=True,
                       medianprops={"linewidth": 1.8}, boxprops={"linewidth": 1.2},
                       whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2})
    for index, state in enumerate(states):
        color = style.state_color(state)
        boxes["boxes"][index].set(facecolor="none", edgecolor=color)
        boxes["medians"][index].set(color=color)
        for part in ("whiskers", "caps"):
            for line in boxes[part][2 * index: 2 * index + 2]:
                line.set(color=color)
        ax.scatter(index + 1 + 0.06 * rng.standard_normal(len(data[index])), data[index], s=12, alpha=0.5,
                   color=color, linewidths=0)
    ax.set_xticks(range(1, len(states) + 1), [f"{style.state_label(state)}\nn = {len(values)}" for state, values in zip(states, data)])
    style.label(ax, title="Head-cast rate per HMM segment, by state", y="head casts per second")
    return fig


def _polar_density(ax, angles, color, bins: int) -> None:
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    counts, _ = np.histogram(angles, bins=edges)
    density = counts / (np.sum(counts) * width) if np.sum(counts) > 0 else counts.astype(float)
    ax.bar((centers + 2 * np.pi) % (2 * np.pi), density, width=width, align="center", alpha=0.85, color=color,
           edgecolor=style.SURFACE, linewidth=0.4)
    ax.set_yticklabels([])


def headcast_directions(headcast_directions_table: pd.DataFrame, bins: int = 24) -> Figure:
    """Direction of head-cast steps, atan2(dy, dx), by state (upstream figure)."""
    table = headcast_directions_table.dropna()
    if table.empty:
        return style.message_figure("No head casts inside HMM segments.")
    states = _state_order(table)
    rows, columns = style.grid_shape(len(states))
    fig, axes = style.new_figure(3.2 * columns, 3.2 * rows, rows, columns, polar=True)
    for ax in axes.flat[len(states):]:
        ax.set_axis_off()
    for ax, state in zip(axes.flat, states):
        angles = table.loc[table[S.STATE].astype(str) == state, S.HEADING].to_numpy(float)
        _polar_density(ax, angles, style.state_color(state), bins)
        ax.set_title(f"{style.state_label(state)} (n = {len(angles)})", fontsize=style.TITLE_SIZE, color=style.INK)
    fig.suptitle("Head-cast direction", fontsize=style.TITLE_SIZE, color=style.INK, x=0.01, ha="left")
    return fig


def headcast_theta_fit(headcast_thetas: pd.DataFrame, bins: int = 24, min_points: int = 5) -> Figure:
    """Head-cast turning angle by state with its von Mises fit (upstream figure)."""
    table = headcast_thetas.dropna()
    states = [state for state in (_state_order(table) if len(table) else [])
              if (table[S.STATE].astype(str) == state).sum() >= min_points]
    if not states:
        return style.message_figure(f"Fewer than {min_points} head-cast angles in every state.")
    rows, columns = style.grid_shape(len(states))
    fig, axes = style.new_figure(3.2 * columns, 3.3 * rows, rows, columns, polar=True)
    for ax in axes.flat[len(states):]:
        ax.set_axis_off()
    for ax, state in zip(axes.flat, states):
        theta = table.loc[table[S.STATE].astype(str) == state, S.THETA].to_numpy(float)
        _polar_density(ax, theta, style.state_color(state), bins)
        mu, kappa, _ = fit.fit_vonmises(theta)
        title = f"{style.state_label(state)} (n = {len(theta)})"
        if np.isfinite(mu) and np.isfinite(kappa):
            xs = np.linspace(-np.pi, np.pi, 400)
            ax.plot((xs + 2 * np.pi) % (2 * np.pi), fit.vonmises_pdf(xs, mu, kappa), linewidth=1.8, color=style.INK)
            title += f"\nvon Mises mu = {mu:.2f}, kappa = {kappa:.2f}"
        ax.set_title(title, fontsize=style.TICK_SIZE + 1, color=style.INK)
    fig.suptitle("Head-cast turning angle and von Mises fit (black)", fontsize=style.TITLE_SIZE, color=style.INK,
                 x=0.01, ha="left")
    return fig


def autocorrelation(autocorr_by_state: pd.DataFrame) -> Figure:
    """Circular autocorrelation against lag, for steering and for head casts (upstream figure)."""
    if autocorr_by_state.empty:
        return style.message_figure("No autocorrelation to show: no HMM segment had enough events.")
    states = _state_order(autocorr_by_state)
    fig, axes = style.new_figure(8.4, 3.3, 1, 2, sharey=True)
    panels = ((S.AUTOCORR_STEERING, "Steering (crawls)"), (S.AUTOCORR_HEADCAST, "Head casts"))
    for ax, (kind, title) in zip(axes[0], panels):
        style.style_axes(ax, grid="y")
        subk = autocorr_by_state[autocorr_by_state[S.KIND] == kind]
        for state in states:
            g = subk[subk[S.STATE].astype(str) == state].sort_values(S.LAG)
            if len(g) == 0:
                continue
            ax.errorbar(g[S.LAG].to_numpy(int), g[S.MEAN].to_numpy(float), yerr=g[S.STD].to_numpy(float),
                        fmt="o-", capsize=2, markersize=3.5, linewidth=1.2, color=style.state_color(state),
                        label=f"{style.state_label(state)} (n = {int(g[S.N_ANIMALS].max())} tracks)")
        ax.axhline(0, linewidth=0.8, color=style.REFERENCE)
        ax.set_xticks(sorted(autocorr_by_state[S.LAG].astype(int).unique()))
        style.label(ax, title=title, x="lag (events)", y=r"$\langle\cos(\Delta\theta)\rangle$" if kind == S.AUTOCORR_STEERING else None)
        if ax.get_legend_handles_labels()[0]:
            style.legend(ax, loc="best")
    return fig


def ar2_diagnostics(run_fits: pd.DataFrame, params: SteeringParams) -> Figure:
    """AR(2) coefficients of every fitted run, with the stationarity triangle, and run sizes.

    A run inside the triangle is a stationary AR(2) process. Runs shown here
    already passed the stability rules, so points outside it are worth a look.
    """
    if run_fits.empty:
        return style.message_figure("No run fits to show.")
    states = _state_order(run_fits)
    fig, axes = style.new_figure(9.6, 3.6, 1, 2)
    plane, sizes = axes[0]
    plane.plot([-2, 0, 2, -2], [-1, 1, -1, -1], color=style.REFERENCE, linewidth=1.0)
    for state in states:
        part = run_fits[run_fits[S.STATE].astype(str) == state]
        plane.scatter(part[S.AR_A1], part[S.AR_A2], s=14, alpha=0.6, color=style.state_color(state), linewidths=0)
    style.style_axes(plane, grid="both")
    style.label(plane, title=f"AR(2) coefficients of {len(run_fits)} fitted runs", x="a1", y="a2")
    style.legend(plane, handles=_state_handles(states) + [Line2D([], [], color=style.REFERENCE, label="stationary inside")],
                 loc="upper right")

    edges = np.linspace(params.min_points_per_run - 0.5, run_fits[S.N_POINTS].max() + 0.5, 21)
    sizes.hist([run_fits.loc[run_fits[S.STATE].astype(str) == state, S.N_POINTS] for state in states], bins=edges,
               stacked=True, color=[style.state_color(state) for state in states], edgecolor=style.SURFACE, linewidth=0.5)
    style.style_axes(sizes, grid="y")
    style.label(sizes, title=f"Points per fitted run (minimum {params.min_points_per_run})", x="points", y="runs")
    return fig
