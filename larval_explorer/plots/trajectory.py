"""Stage 03, 04, 05 and 10 figures: smoothing, body length, RDP steps, step models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from larval_explorer.core import schema as S
from larval_explorer.plots import style


def _track(table: pd.DataFrame, track_id: str, time_column: str = S.TIME_S) -> pd.DataFrame:
    return table[table[S.TRACK_ID] == track_id].sort_values(time_column)


def _step_polyline(steps: pd.DataFrame):
    """Vertices of a step table: every step start, then the last step's end."""
    x = np.append(steps[S.X0_MM].to_numpy(float), steps[S.X1_MM].iloc[-1])
    y = np.append(steps[S.Y0_MM].to_numpy(float), steps[S.Y1_MM].iloc[-1])
    return x, y


def smoothing_overlay(unsmoothed: pd.DataFrame, smoothed: pd.DataFrame, track_id: str) -> Figure:
    """Stage 03: the centre of mass before and after Savitzky-Golay smoothing.

    The path, then x and y against time. Zoom into a turn to judge whether
    the window is eating real movement.
    """
    before, after = _track(unsmoothed, track_id), _track(smoothed, track_id)
    if before.empty or after.empty:
        return style.message_figure(f"No trajectory for {track_id}.")
    fig = Figure(figsize=(10.0, 4.6), dpi=100, layout="constrained", facecolor=style.SURFACE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])
    path = fig.add_subplot(grid[:, 0])
    x_time = fig.add_subplot(grid[0, 1])
    y_time = fig.add_subplot(grid[1, 1], sharex=x_time)
    for ax in (path, x_time, y_time):
        style.style_axes(ax)

    path.plot(before[S.MOM_X], before[S.MOM_Y], color=style.DROPPED, linewidth=1.0, label="before smoothing")
    path.plot(after[S.MOM_X], after[S.MOM_Y], color=style.CATEGORICAL[0], linewidth=style.LINE_WIDTH, label="smoothed")
    style.equal_aspect(path)
    style.label(path, title=str(track_id))
    style.legend(path, loc="best")

    for ax, column, name in ((x_time, S.MOM_X, "x (mm)"), (y_time, S.MOM_Y, "y (mm)")):
        ax.plot(before[S.TIME_S], before[column], color=style.DROPPED, linewidth=1.0)
        ax.plot(after[S.TIME_S], after[column], color=style.CATEGORICAL[0], linewidth=style.LINE_WIDTH)
        style.label(ax, y=name)
    x_time.tick_params(labelbottom=False)
    style.label(y_time, x="time (s)")
    return fig


def body_length(body_lengths: pd.DataFrame) -> Figure:
    """Stage 04: mean body length per track with its SD, and the distribution of means.

    An outlier here becomes an outlier in the RDP tolerance of that track.
    """
    if body_lengths.empty:
        return style.message_figure("No body lengths to show.")
    table = body_lengths.sort_values(S.TRACK_ID, key=lambda ids: ids.map(style.natural_key)).reset_index(drop=True)
    fig, axes = style.new_figure(9.0, max(2.6, 0.24 * len(table) + 1.2), 1, 2, width_ratios=[1.6, 1.0])
    strip, histogram = axes[0]
    rows = np.arange(len(table))
    strip.errorbar(table[S.BODY_LENGTH_MEAN], rows, xerr=table[S.BODY_LENGTH_STD].fillna(0.0), fmt="o",
                   color=style.CATEGORICAL[0], ecolor=style.REFERENCE, markersize=4, linewidth=1.0, capsize=2)
    median = float(table[S.BODY_LENGTH_MEAN].median())
    strip.axvline(median, color=style.REFERENCE, linewidth=0.8, linestyle="--")
    strip.set_yticks(rows, table[S.TRACK_ID])
    strip.set_ylim(len(table) - 0.5, -0.5)
    style.style_axes(strip, grid="x")
    style.label(strip, title=f"Mean body length per track, bars = SD (median {median:.2f} mm)", x="body length (mm)")

    histogram.hist(table[S.BODY_LENGTH_MEAN], bins=min(20, max(5, len(table) // 2)),
                   color=style.CATEGORICAL[0], edgecolor=style.SURFACE, linewidth=0.8)
    style.label(histogram, title="Distribution of track means", x="body length (mm)", y="tracks")
    return fig


def rdp_overlay(smoothed: pd.DataFrame, steps: pd.DataFrame, track_id: str) -> Figure:
    """Stage 05: the RDP polyline over the smoothed trajectory of one track."""
    track, track_steps = _track(smoothed, track_id), _track(steps, track_id, S.T_START_S)
    if track.empty:
        return style.message_figure(f"No trajectory for {track_id}.")
    fig, axes = style.new_figure(5.6, 5.2)
    ax = axes[0, 0]
    ax.plot(track[S.MOM_X], track[S.MOM_Y], color=style.DROPPED, linewidth=1.0, label="smoothed trajectory")
    title = f"{track_id}: no RDP steps"
    if len(track_steps):
        x, y = _step_polyline(track_steps)
        ax.plot(x, y, color=style.CATEGORICAL[0], linewidth=style.LINE_WIDTH, marker="o", markersize=3.5,
                label="RDP steps")
        epsilon = float(track_steps[S.EPSILON_VALUE_MM].iloc[0])
        style.scale_bar(ax, epsilon, f"tolerance {epsilon:.2f} mm")
        title = f"{track_id}: {len(track_steps)} steps"
    style.equal_aspect(ax)
    style.label(ax, title=title)
    style.legend(ax, loc="best")
    return fig


def step_distributions(steps: pd.DataFrame) -> Figure:
    """Stage 05: step length, speed and turning angle, and phi on polar axes."""
    if steps.empty:
        return style.message_figure("No RDP steps to show.")
    fig = Figure(figsize=(12.0, 3.0), dpi=100, layout="constrained", facecolor=style.SURFACE)
    panels = [
        (S.STEP_LENGTH, "Step length", "mm"),
        (S.VELOCITY, "Speed", "mm/s"),
        (S.THETA, "Turning angle theta", "rad"),
    ]
    for index, (column, title, unit) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, index)
        style.style_axes(ax)
        values = steps[column].dropna()
        ax.hist(values, bins=40, color=style.CATEGORICAL[0], edgecolor=style.SURFACE, linewidth=0.5)
        style.label(ax, title=f"{title} (n = {len(values)})", x=unit, y="steps" if index == 1 else None)
    polar = fig.add_subplot(1, 4, 4, projection="polar")
    style.style_axes(polar)
    phi = steps[S.PHI].dropna().to_numpy(float)
    edges = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(phi, bins=edges)
    polar.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=np.diff(edges), color=style.CATEGORICAL[0],
              edgecolor=style.SURFACE, linewidth=0.5)
    polar.set_yticklabels([])
    polar.set_title(f"phi (n = {len(phi)})", fontsize=style.TITLE_SIZE, color=style.INK)
    return fig


def step_models(
    smoothed: pd.DataFrame, run_anchor_steps: pd.DataFrame, event_level_steps: pd.DataFrame, track_id: str
) -> Figure:
    """Stage 10: the two step representations of one track, side by side."""
    track = _track(smoothed, track_id)
    if track.empty:
        return style.message_figure(f"No trajectory for {track_id}.")
    fig, axes = style.new_figure(10.0, 4.8, 1, 2, sharex=True, sharey=True)
    anchors, events = axes[0]
    for ax in (anchors, events):
        ax.plot(track[S.MOM_X], track[S.MOM_Y], color=style.DROPPED, linewidth=1.0, zorder=1)
        style.equal_aspect(ax)
        ax.set_aspect("equal", adjustable="box")  # the two panels share their limits

    runs = _track(run_anchor_steps, track_id, S.T_START_S)
    if len(runs):
        x, y = _step_polyline(runs)
        anchors.plot(x, y, color=style.CATEGORICAL[0], linewidth=style.LINE_WIDTH, marker="o", markersize=3.5)
    style.label(anchors, title=f"Run-anchor steps: {len(runs)} (anchors at head-cast midpoints)")

    steps = _track(event_level_steps, track_id, S.T_START_S)
    for event_type, color, name in ((S.EVENT_CRAWL, style.CRAWL_COLOR, "crawl"),
                                    (S.EVENT_HEAD_CAST, style.HEAD_CAST_COLOR, "head cast")):
        part = steps[steps[S.EVENT_TYPE] == event_type]
        segments_x = np.column_stack([part[S.X0_MM], part[S.X1_MM], np.full(len(part), np.nan)]).ravel()
        segments_y = np.column_stack([part[S.Y0_MM], part[S.Y1_MM], np.full(len(part), np.nan)]).ravel()
        events.plot(segments_x, segments_y, color=color, linewidth=style.LINE_WIDTH,
                    label=f"{name} ({len(part)})", zorder=3 if event_type == S.EVENT_HEAD_CAST else 2)
    style.label(events, title="Event-level steps")
    style.legend(events, loc="best")
    events.set_ylabel("")
    fig.suptitle(str(track_id), fontsize=style.TITLE_SIZE, color=style.INK, x=0.01, ha="left")
    return fig
