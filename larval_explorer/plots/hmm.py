"""Stage 06, 07 and 08 figures: the HMM fit, the run filter, the HMM segments.

Stored state indices are upstream's and are never reordered (D-016); a state's
colour comes from that index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from larval_explorer.core import schema as S
from larval_explorer.plots import style

N_STATES = 3
PHI_KDE_BW_ADJUST = 0.7  # upstream's seaborn setting


def _states(steps: pd.DataFrame, column: str) -> list[int]:
    return sorted(int(state) for state in steps[column].dropna().unique())


def _state_legend(target, states, **kwargs) -> None:
    handles = [Patch(facecolor=style.state_color(state), label=style.state_label(state)) for state in states]
    style.legend(target, handles=handles, **kwargs)


def phi_by_state(steps: pd.DataFrame) -> Figure:
    """Stage 06 (upstream figure): density of phi in each state, on polar axes."""
    if steps.empty:
        return style.message_figure("No HMM-labelled steps to show.")
    fig, axes = style.new_figure(11.0, 4.0, 1, N_STATES, polar=True)
    for state, ax in zip(range(N_STATES), axes[0]):
        values = steps.loc[steps[S.STATE] == state, S.PHI]
        curve = style.kde_curve(values, bw_adjust=PHI_KDE_BW_ADJUST)
        if curve is not None:
            ax.plot(curve[0], curve[1], color=style.state_color(state), linewidth=style.LINE_WIDTH)
        ax.set_title(f"{style.state_label(state)} (n = {len(values)})", fontsize=style.TITLE_SIZE, color=style.INK)
    fig.suptitle(r"State-wise distributions of $\phi$", fontsize=style.TITLE_SIZE + 1, color=style.INK)
    return fig


def phi_overlay(steps: pd.DataFrame) -> Figure:
    """Stage 06 (upstream figure): the three phi densities on one polar axis."""
    if steps.empty:
        return style.message_figure("No HMM-labelled steps to show.")
    fig, axes = style.new_figure(5.6, 4.5, polar=True)
    ax = axes[0, 0]
    for state in range(N_STATES):
        curve = style.kde_curve(steps.loc[steps[S.STATE] == state, S.PHI], bw_adjust=PHI_KDE_BW_ADJUST)
        if curve is not None:
            ax.plot(curve[0], curve[1], color=style.state_color(state), linewidth=style.LINE_WIDTH,
                    label=style.state_label(state))
    ax.set_title(r"Overlaid distributions of $\phi$", fontsize=style.TITLE_SIZE, color=style.INK)
    style.legend(ax, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    return fig


def transition_matrix(matrix) -> Figure:
    """Stage 06: probability of moving from one state (row) to another (column)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return style.message_figure("No transition matrix to show.")
    fig, axes = style.new_figure(4.2, 3.6)
    ax = axes[0, 0]
    colormap = LinearSegmentedColormap.from_list("sequential_blue", style.SEQUENTIAL)
    image = ax.imshow(matrix, cmap=colormap, vmin=0.0, vmax=1.0)
    n = matrix.shape[0]
    for row in range(n):
        for column in range(n):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=style.LABEL_SIZE,
                    color=style.SURFACE if matrix[row, column] > 0.55 else style.INK)
    ax.set_xticks(range(n), [style.state_label(k) for k in range(n)])
    ax.set_yticks(range(n), [style.state_label(k) for k in range(n)])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    style.label(ax, title="Transition probabilities", x="to", y="from")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=style.TICK_SIZE, colors=style.INK_SECONDARY)
    colorbar.outline.set_visible(False)
    return fig


def _draw_raster(ax, steps: pd.DataFrame, column: str, tracks: list) -> None:
    for row, track_id in enumerate(tracks):
        track = steps[steps[S.TRACK_ID] == track_id]
        for state in _states(track, column):
            part = track[track[column] == state]
            spans = list(zip(part[S.T_START_S], part[S.T_END_S] - part[S.T_START_S]))
            ax.broken_barh(spans, (row - 0.38, 0.76), facecolor=style.state_color(state), linewidth=0)
    ax.set_yticks(range(len(tracks)), tracks)
    ax.set_ylim(len(tracks) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def state_raster(steps: pd.DataFrame, state_column: str = S.STATE) -> Figure:
    """Stage 06: the decoded state of every step, one row per track, against time."""
    if steps.empty:
        return style.message_figure("No HMM-labelled steps to show.")
    tracks = sorted(steps[S.TRACK_ID].unique(), key=style.natural_key)
    fig, axes = style.new_figure(10.0, 1.2 + 0.26 * len(tracks))
    ax = axes[0, 0]
    _draw_raster(ax, steps, state_column, tracks)
    style.label(ax, title="HMM state of each step", x="time (s)")
    _state_legend(fig, range(N_STATES), loc="outside lower center", ncols=N_STATES)
    return fig


def posterior_ribbons(steps: pd.DataFrame, track_id: str) -> Figure:
    """Stage 06: the posterior probability of each state along one track."""
    track = steps[steps[S.TRACK_ID] == track_id].sort_values(S.T_START_S)
    if track.empty:
        return style.message_figure(f"No HMM-labelled steps for {track_id}.")
    fig, axes = style.new_figure(10.0, 2.8)
    ax = axes[0, 0]
    edges = np.append(track[S.T_START_S].to_numpy(float), track[S.T_END_S].iloc[-1])
    lower = np.zeros(len(track))
    for state in range(N_STATES):
        probability = track[S.state_probability(state)].to_numpy(float)
        ax.stairs(lower + probability, edges, baseline=lower, fill=True, color=style.state_color(state),
                  edgecolor=style.SURFACE, linewidth=0.3)
        lower = lower + probability
    ax.set_ylim(0, 1)
    ax.set_xlim(edges[0], edges[-1])
    style.label(ax, title=f"{track_id}: posterior probability of each state, per step", x="time (s)", y="probability")
    _state_legend(fig, range(N_STATES), loc="outside lower center", ncols=N_STATES)
    return fig


def log_likelihood_trace(history) -> Figure:
    """Stage 06: log-likelihood after each EM iteration."""
    history = np.asarray(history, dtype=float)
    if history.size == 0:
        return style.message_figure("No convergence history to show.")
    fig, axes = style.new_figure(5.0, 3.0)
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(history) + 1), history, color=style.CATEGORICAL[0], linewidth=style.LINE_WIDTH,
            marker="o", markersize=3)
    style.style_axes(ax, grid="y")
    style.label(ax, title=f"EM convergence ({len(history)} iterations)", x="iteration", y="log-likelihood")
    return fig


def state_trajectory(smoothed: pd.DataFrame, steps: pd.DataFrame, track_id: str, state_column: str = S.STATE) -> Figure:
    """Stage 06/07: one track's RDP steps coloured by HMM state, over its trajectory."""
    track = smoothed[smoothed[S.TRACK_ID] == track_id].sort_values(S.TIME_S)
    track_steps = steps[steps[S.TRACK_ID] == track_id].sort_values(S.T_START_S)
    if track.empty:
        return style.message_figure(f"No trajectory for {track_id}.")
    fig, axes = style.new_figure(5.6, 5.2)
    ax = axes[0, 0]
    ax.plot(track[S.MOM_X], track[S.MOM_Y], color=style.DROPPED, linewidth=1.0, zorder=1)
    present = _states(track_steps, state_column)
    for state in present:
        part = track_steps[track_steps[state_column] == state]
        x = np.column_stack([part[S.X0_MM], part[S.X1_MM], np.full(len(part), np.nan)]).ravel()
        y = np.column_stack([part[S.Y0_MM], part[S.Y1_MM], np.full(len(part), np.nan)]).ravel()
        ax.plot(x, y, color=style.state_color(state), linewidth=1.8, marker="o", markersize=2.5, zorder=2)
    style.equal_aspect(ax)
    kind = "filtered state" if state_column == S.STATE_FILTERED else "state"
    style.label(ax, title=f"{track_id}: steps by HMM {kind}")
    if present:
        _state_legend(ax, present, loc="best")
    return fig


def filter_comparison(steps: pd.DataFrame, min_run: int) -> Figure:
    """Stage 07: the state raster before and after short runs are replaced."""
    if steps.empty:
        return style.message_figure("No filtered steps to show.")
    tracks = sorted(steps[S.TRACK_ID].unique(), key=style.natural_key)
    fig, axes = style.new_figure(10.0, 1.6 + 0.5 * len(tracks), 2, 1, sharex=True)
    before, after = axes[:, 0]
    _draw_raster(before, steps, S.STATE, tracks)
    _draw_raster(after, steps, S.STATE_FILTERED, tracks)
    changed = int((steps[S.STATE] != steps[S.STATE_FILTERED]).sum())
    style.label(before, title="Decoded states")
    style.label(after, title=f"After replacing runs shorter than {min_run} steps: {changed} of {len(steps)} steps relabelled",
                x="time (s)")
    _state_legend(fig, range(N_STATES), loc="outside lower center", ncols=N_STATES)
    return fig


def _run_lengths(steps: pd.DataFrame, column: str) -> pd.DataFrame:
    """Length and state of every run of equal consecutive states, per track."""
    rows = []
    for _, track in steps.sort_values([S.TRACK_ID, S.T_START_S]).groupby(S.TRACK_ID, sort=False):
        states = track[column].to_numpy()
        starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
        lengths = np.diff(np.r_[starts, len(states)])
        rows.extend(zip(states[starts].tolist(), lengths.tolist()))
    return pd.DataFrame(rows, columns=[S.STATE, S.N_STEPS])


def _run_length_histogram(ax, runs: pd.DataFrame, threshold: int, threshold_name: str) -> None:
    if runs.empty:
        return
    edges = np.arange(0.5, runs[S.N_STEPS].max() + 1.5)
    present = sorted(int(state) for state in runs[S.STATE].unique())
    ax.hist([runs.loc[runs[S.STATE] == state, S.N_STEPS] for state in present], bins=edges, stacked=True,
            color=[style.state_color(state) for state in present], edgecolor=style.SURFACE, linewidth=0.5)
    ax.axvline(threshold - 0.5, color=style.INK, linewidth=1.0, linestyle="--")
    ax.annotate(f"{threshold_name} = {threshold}", (threshold - 0.5, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -10), textcoords="offset points", fontsize=style.TICK_SIZE, color=style.INK_SECONDARY)
    style.style_axes(ax, grid="y")
    _state_legend(ax, present, loc="upper right")


def run_length_histogram(steps: pd.DataFrame, min_run: int) -> Figure:
    """Stage 07: lengths of the decoded state runs, with the filter threshold marked."""
    if steps.empty:
        return style.message_figure("No HMM-labelled steps to show.")
    runs = _run_lengths(steps, S.STATE)
    fig, axes = style.new_figure(6.0, 3.2)
    ax = axes[0, 0]
    _run_length_histogram(ax, runs, min_run, "min_run")
    short = int((runs[S.N_STEPS] < min_run).sum())
    style.label(ax, title=f"Run lengths before filtering: {short} of {len(runs)} runs are shorter than {min_run}",
                x="run length (steps)", y="runs")
    return fig


def hmm_segment_lengths(steps: pd.DataFrame, min_steps: int) -> Figure:
    """Stage 08: lengths of the filtered state runs against the representative-segment threshold."""
    if steps.empty:
        return style.message_figure("No filtered steps to show.")
    runs = _run_lengths(steps, S.STATE_FILTERED)
    fig, axes = style.new_figure(6.0, 3.2)
    ax = axes[0, 0]
    _run_length_histogram(ax, runs, min_steps, "min_steps")
    kept = int((runs[S.N_STEPS] >= min_steps).sum())
    style.label(ax, title=f"Filtered runs: {kept} of {len(runs)} reach {min_steps} steps and become HMM segments",
                x="run length (steps)", y="runs")
    return fig
