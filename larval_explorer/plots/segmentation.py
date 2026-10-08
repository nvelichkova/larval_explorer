"""Stage 01 and 02 figures: where the tracks are, and how much of each is real."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from larval_explorer.core import schema as S
from larval_explorer.core.params import CalibrationParams, SegmentationParams
from larval_explorer.plots import style


def tracks_overview(calibrated: pd.DataFrame, calibration: CalibrationParams) -> Figure:
    """Stage 01: every track in millimetres, with the coordinate range spelled out.

    The range is the sanity check on the pixel scale: it should fit the arena.
    """
    if calibrated.empty:
        return style.message_figure("No tracks to show.")
    fig, axes = style.new_figure(6.0, 5.6)
    ax = axes[0, 0]
    tracks = sorted(calibrated[S.TRACK_ID].unique(), key=style.natural_key)
    for index, track_id in enumerate(tracks):
        track = calibrated[calibrated[S.TRACK_ID] == track_id]
        color = style.categorical_color(index)
        ax.plot(track[S.MOM_X], track[S.MOM_Y], color=color, linewidth=1.0)
        ax.annotate(str(S.larva_index(track_id)), (track[S.MOM_X].iloc[0], track[S.MOM_Y].iloc[0]),
                    xytext=(3, 3), textcoords="offset points", fontsize=style.TICK_SIZE, color=style.INK)
        ax.plot(track[S.MOM_X].iloc[0], track[S.MOM_Y].iloc[0], "o", color=color, markersize=3.5)
    style.equal_aspect(ax)
    x, y = calibrated[S.MOM_X], calibrated[S.MOM_Y]
    scale = calibration.millimetres_per_pixel
    style.label(ax, title=(
        f"{len(tracks)} tracks, numbered at their first tracked position\n"
        f"x {x.min():.0f}-{x.max():.0f} mm, y {y.min():.0f}-{y.max():.0f} mm "
        f"(pixels {x.min() / scale:.0f}-{x.max() / scale:.0f}, {y.min() / scale:.0f}-{y.max() / scale:.0f})"
    ))
    return fig


def _draw_coverage(ax, track_segments: pd.DataFrame, recording_seconds: float) -> None:
    larvae = sorted(track_segments[S.LARVA_INDEX].unique())
    for row, larva in enumerate(larvae):
        ax.broken_barh([(0, recording_seconds)], (row - 0.35, 0.7), facecolor=style.GAP, linewidth=0)
        segments = track_segments[track_segments[S.LARVA_INDEX] == larva]
        for _, segment in segments.iterrows():
            color = style.categorical_color(segment[S.SEGMENT_INDEX]) if segment[S.KEPT] else style.DROPPED
            ax.broken_barh([(segment[S.START_S], segment[S.DURATION_S])], (row - 0.35, 0.7),
                           facecolor=color, edgecolor=style.SURFACE, linewidth=0.8)
    ax.set_yticks(range(len(larvae)), [f"larva {larva}" for larva in larvae])
    ax.set_ylim(len(larvae) - 0.4, -0.6)
    ax.set_xlim(0, recording_seconds)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    style.label(ax, x="time (s)")


def _draw_larva_trajectory(ax, calibrated, segmented, track_segments, source_id) -> None:
    """One larva: naive join in pale grey, dropped segments grey, kept segments coloured."""
    raw = calibrated[calibrated[S.TRACK_ID] == source_id].sort_values(S.TIME_S)
    ax.plot(raw[S.MOM_X], raw[S.MOM_Y], color=style.GAP, linewidth=0.8, zorder=1)
    segments = track_segments[track_segments[S.SOURCE_TRACK_ID] == source_id]
    for _, segment in segments[~segments[S.KEPT]].iterrows():
        part = raw[(raw[S.TIME_S] >= segment[S.START_S]) & (raw[S.TIME_S] <= segment[S.END_S])]
        ax.plot(part[S.MOM_X], part[S.MOM_Y], color=style.DROPPED, linewidth=1.0, zorder=2)
    for _, segment in segments[segments[S.KEPT]].iterrows():
        part = segmented[segmented[S.TRACK_ID] == segment[S.TRACK_ID]]
        ax.plot(part[S.MOM_X], part[S.MOM_Y], color=style.categorical_color(segment[S.SEGMENT_INDEX]),
                linewidth=style.LINE_WIDTH, zorder=3)
    style.equal_aspect(ax)
    kept = int(segments[S.KEPT].sum())
    style.label(ax, title=f"larva {S.larva_index(source_id)}: {kept} kept of {len(segments)}")


def segmentation_overview(
    calibrated: pd.DataFrame,
    segmented: pd.DataFrame,
    track_segments: pd.DataFrame,
    params: SegmentationParams,
    recording_id: str = "",
) -> Figure:
    """Stage 02: coverage timeline above a trajectory grid, one panel per larva.

    Coloured = a contiguous segment that is kept. Dark grey = tracked but too
    short. Pale grey on the timeline = not tracked; pale grey on a trajectory
    = the straight join across lost frames that upstream would analyse.
    """
    if track_segments.empty:
        return style.message_figure("No track segments to show.")
    sources = sorted(track_segments[S.SOURCE_TRACK_ID].unique(), key=style.natural_key)
    rows, columns = style.grid_shape(len(sources))
    fig = Figure(figsize=(3.4 * columns, 1.2 + 0.32 * len(sources) + 3.2 * rows), dpi=100,
                 layout="constrained", facecolor=style.SURFACE)
    grid = fig.add_gridspec(rows + 1, columns, height_ratios=[0.11 * len(sources) + 0.25] + [1.0] * rows)

    recording_seconds = float(calibrated[S.TIME_S].max()) if len(calibrated) else float(track_segments[S.END_S].max())
    timeline = fig.add_subplot(grid[0, :])
    style.style_axes(timeline)
    _draw_coverage(timeline, track_segments, recording_seconds)
    kept = int(track_segments[S.KEPT].sum())
    style.label(timeline, title=(
        f"{recording_id}\n{len(sources)} larvae, {len(track_segments)} segments found, {kept} kept "
        f"(bridge up to {params.bridge_max_frames} frames, keep {params.min_segment_seconds:g} s or longer)"
    ).strip())

    for index, source_id in enumerate(sources):
        ax = fig.add_subplot(grid[1 + index // columns, index % columns])
        style.style_axes(ax)
        ax.set_gid(str(source_id))  # lets a click on the panel identify the larva
        _draw_larva_trajectory(ax, calibrated, segmented, track_segments, source_id)

    handles = [
        Patch(facecolor=style.categorical_color(0), label="segment kept (colour = segment number)"),
        Patch(facecolor=style.DROPPED, label="segment too short, dropped"),
        Patch(facecolor=style.GAP, label="not tracked"),
        Line2D([], [], color=style.GAP, linewidth=1.2, label="straight join across lost frames"),
    ]
    style.legend(fig, handles=handles, loc="outside lower center", ncols=2)
    return fig


def coverage_timeline(track_segments: pd.DataFrame, recording_seconds: float) -> Figure:
    """The coverage timeline on its own, for the live segmentation controls."""
    if track_segments.empty:
        return style.message_figure("No track segments to show.")
    larvae = track_segments[S.LARVA_INDEX].nunique()
    fig, axes = style.new_figure(9.0, 0.9 + 0.32 * larvae)
    _draw_coverage(axes[0, 0], track_segments, recording_seconds)
    return fig
