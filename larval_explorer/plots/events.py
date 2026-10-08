"""Stage 09 figures: head-cast and crawl detection.

The per-track figures are the ones upstream drew inside its detection loop,
redrawn from the signals stage 09 returns. Upstream's "tail trajectory" figure
plotted the centre of mass a second time and is not reproduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from larval_explorer.core import schema as S
from larval_explorer.core.params import EventParams
from larval_explorer.plots import style

HEAD_CAST_SPAN_ALPHA = 0.25
THRESHOLD_VIEW_MULTIPLE = 5.0


def _for_track(table: pd.DataFrame, track_id: str) -> pd.DataFrame:
    return table[table[S.TRACK_ID] == track_id]


def _head_casts(events: pd.DataFrame, track_id: str) -> pd.DataFrame:
    track = _for_track(events, track_id)
    return track[track[S.EVENT_TYPE] == S.EVENT_HEAD_CAST]


def _threshold_row(thresholds: pd.DataFrame, track_id: str) -> pd.Series | None:
    rows = _for_track(thresholds, track_id)
    return rows.iloc[0] if len(rows) else None


def _shade_head_casts(ax, head_casts: pd.DataFrame) -> None:
    for start, end in zip(head_casts[S.START_S], head_casts[S.END_S]):
        ax.axvspan(start, end, color=style.HEAD_CAST_COLOR, alpha=HEAD_CAST_SPAN_ALPHA, linewidth=0)


def anterior_velocity(signals: pd.DataFrame, events: pd.DataFrame, thresholds: pd.DataFrame, track_id: str) -> Figure:
    """Angular velocity of the anterior body axis, its threshold, and the head casts found."""
    track, row = _for_track(signals, track_id), _threshold_row(thresholds, track_id)
    if track.empty or row is None:
        return style.message_figure(f"No event signals for {track_id}.")
    head_casts = _head_casts(events, track_id)
    fig, axes = style.new_figure(10.0, 3.2)
    ax = axes[0, 0]
    _shade_head_casts(ax, head_casts)
    ax.plot(track[S.TIME_S], track[S.ANTERIOR_ANGULAR_VELOCITY], color=style.CATEGORICAL[0], linewidth=0.9)
    for sign in (1, -1):
        ax.axhline(sign * row[S.ANTERIOR_THRESHOLD], color=style.REFERENCE, linewidth=0.9, linestyle="--")
    style.label(ax, title=f"{track_id}: anterior (head) angular velocity, {len(head_casts)} head casts",
                x="time (s)", y="rad/s")
    style.legend(ax, handles=[
        Line2D([], [], color=style.CATEGORICAL[0], label="anterior angular velocity"),
        Line2D([], [], color=style.REFERENCE, linestyle="--", label=f"threshold ±{row[S.ANTERIOR_THRESHOLD]:.3f}"),
        Patch(facecolor=style.HEAD_CAST_COLOR, alpha=HEAD_CAST_SPAN_ALPHA, label="head cast"),
    ], loc="upper right", ncols=3)
    return fig


def linear_speed(signals: pd.DataFrame, events: pd.DataFrame, track_id: str) -> Figure:
    """Head and tail speed; a head cast needs the head to outrun the tail."""
    track = _for_track(signals, track_id)
    if track.empty:
        return style.message_figure(f"No event signals for {track_id}.")
    fig, axes = style.new_figure(10.0, 3.2)
    ax = axes[0, 0]
    _shade_head_casts(ax, _head_casts(events, track_id))
    ax.plot(track[S.TIME_S], track[S.HEAD_SPEED], color=style.CATEGORICAL[0], linewidth=0.9, label="head")
    ax.plot(track[S.TIME_S], track[S.TAIL_SPEED], color=style.CATEGORICAL[1], linewidth=0.9, label="tail")
    style.label(ax, title=f"{track_id}: linear speed, head against tail", x="time (s)", y="speed (mm/s)")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=style.HEAD_CAST_COLOR, alpha=HEAD_CAST_SPAN_ALPHA, label="head cast"))
    style.legend(ax, handles=handles, loc="upper right", ncols=3)
    return fig


def threshold_distributions(
    signals: pd.DataFrame, thresholds: pd.DataFrame, track_id: str, params: EventParams
) -> Figure:
    """The |angular velocity| distributions and the two-line fits that set the thresholds.

    Only the anterior threshold is used for detection; upstream computes the
    posterior one as well and plots it.
    """
    track, row = _for_track(signals, track_id), _threshold_row(thresholds, track_id)
    if track.empty or row is None:
        return style.message_figure(f"No event signals for {track_id}.")
    fig, axes = style.new_figure(10.0, 3.4, 1, 2)
    panels = (
        ("anterior", S.ANTERIOR_ANGULAR_VELOCITY, S.ANTERIOR_THRESHOLD, "used for head-cast detection"),
        ("posterior", S.POSTERIOR_ANGULAR_VELOCITY, S.POSTERIOR_THRESHOLD, "diagnostic only"),
    )
    for ax, (side, column, threshold_column, note) in zip(axes[0], panels):
        counts, edges = np.histogram(np.abs(track[column].to_numpy(float)), bins=params.threshold_histogram_bins)
        probability = counts / counts.sum()
        centers = (edges[:-1] + edges[1:]) * 0.5
        split = int(row[S.threshold_fit_column(side, "split_bin")])
        low = (row[S.threshold_fit_column(side, "low_slope")], row[S.threshold_fit_column(side, "low_intercept")])
        high = (row[S.threshold_fit_column(side, "high_slope")], row[S.threshold_fit_column(side, "high_intercept")])
        ax.plot(centers, probability, color=style.CATEGORICAL[0], linewidth=style.LINE_WIDTH, label="distribution")
        ax.plot(centers[:split], np.polyval(low, centers[:split]), color=style.CATEGORICAL[1], linewidth=1.0,
                linestyle="--", label="fit, low side")
        ax.plot(centers[split:], np.polyval(high, centers[split:]), color=style.CATEGORICAL[2], linewidth=1.0,
                linestyle="--", label="fit, high side")
        ax.axvline(row[threshold_column], color=style.INK, linewidth=1.0, linestyle=":",
                   label=f"threshold {row[threshold_column]:.3f}")
        ax.set_ylim(bottom=min(0.0, float(probability.min())), top=float(probability.max()) * 1.08)
        # The distribution has a long thin tail; show the region where the two fits meet.
        shown = THRESHOLD_VIEW_MULTIPLE * float(row[threshold_column])
        if np.isfinite(shown) and 0 < shown < centers[-1]:
            ax.set_xlim(0, shown)
        style.label(ax, title=f"|{side} angular velocity| ({note})",
                    x=f"rad/s (full range reaches {centers[-1]:.1f})", y="probability")
        style.legend(ax, loc="upper right")
    fig.suptitle(str(track_id), fontsize=style.TITLE_SIZE, color=style.INK, x=0.01, ha="left")
    return fig


def _path_with_head_casts(ax, track: pd.DataFrame, head_casts: pd.DataFrame, highlight: bool) -> None:
    x, y, t = track[S.COM_X_SMOOTH].to_numpy(float), track[S.COM_Y_SMOOTH].to_numpy(float), track[S.TIME_S].to_numpy(float)
    in_cast = np.zeros(len(t), dtype=bool)
    for start, end in zip(head_casts[S.START_S], head_casts[S.END_S]):
        in_cast |= (t >= start) & (t <= end)
    pieces = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    if highlight:
        # Upstream colours a piece when either of its ends lies in a head cast.
        cast_piece = in_cast[:-1] | in_cast[1:]
        colors = np.where(cast_piece, style.HEAD_CAST_COLOR, style.CRAWL_COLOR)
    else:
        colors = style.CRAWL_COLOR
    ax.add_collection(LineCollection(pieces, colors=colors, linewidths=1.6))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale_view()
    style.equal_aspect(ax)


def head_cast_trajectory(signals: pd.DataFrame, events: pd.DataFrame, thresholds: pd.DataFrame, track_id: str) -> Figure:
    """The smoothed centre-of-mass path, red where a head cast is under way."""
    track, row = _for_track(signals, track_id).sort_values(S.TIME_S), _threshold_row(thresholds, track_id)
    if len(track) < 2 or row is None:
        return style.message_figure(f"No event signals for {track_id}.")
    head_casts = _head_casts(events, track_id)
    fig, axes = style.new_figure(5.4, 5.2)
    ax = axes[0, 0]
    _path_with_head_casts(ax, track, head_casts, highlight=True)
    style.scale_bar(ax, float(row[S.MEAN_SPINE_LENGTH]), f"body length {row[S.MEAN_SPINE_LENGTH]:.2f} mm")
    style.label(ax, title=f"{track_id}: trajectory by head casts")
    style.legend(ax, handles=[
        Line2D([], [], color=style.CRAWL_COLOR, linewidth=1.6, label="path"),
        Line2D([], [], color=style.HEAD_CAST_COLOR, linewidth=1.6, label=f"head cast ({len(head_casts)})"),
    ], loc="best")
    return fig


def head_cast_circles(signals: pd.DataFrame, events: pd.DataFrame, thresholds: pd.DataFrame, track_id: str) -> Figure:
    """The path in black with a circle around where each head cast happened."""
    track, row = _for_track(signals, track_id).sort_values(S.TIME_S), _threshold_row(thresholds, track_id)
    if len(track) < 2 or row is None:
        return style.message_figure(f"No event signals for {track_id}.")
    head_casts = _head_casts(events, track_id)
    mean_spine = float(row[S.MEAN_SPINE_LENGTH])
    fig, axes = style.new_figure(5.4, 5.2)
    ax = axes[0, 0]
    _path_with_head_casts(ax, track, head_casts, highlight=False)
    x, y, t = track[S.COM_X_SMOOTH].to_numpy(float), track[S.COM_Y_SMOOTH].to_numpy(float), track[S.TIME_S].to_numpy(float)
    for start, end in zip(head_casts[S.START_S], head_casts[S.END_S]):
        mask = (t >= start) & (t <= end)
        xs, ys = x[mask], y[mask]
        if xs.size == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.max(np.hypot(xs - cx, ys - cy)))
        if not np.isfinite(radius) or radius <= 0:
            radius = float(max(1e-3, 0.5 * mean_spine))
        ax.add_patch(Circle((cx, cy), 1.1 * radius, fill=False, edgecolor=style.HEAD_CAST_COLOR, linewidth=1.6))
    style.scale_bar(ax, mean_spine, f"body length {mean_spine:.2f} mm")
    style.label(ax, title=f"{track_id}: {len(head_casts)} head casts circled")
    return fig


def ethogram(events: pd.DataFrame) -> Figure:
    """Every track as a row: crawls in black, head casts in red, against time."""
    if events.empty:
        return style.message_figure("No events to show.")
    tracks = sorted(events[S.TRACK_ID].unique(), key=style.natural_key)
    fig, axes = style.new_figure(10.0, 1.2 + 0.26 * len(tracks))
    ax = axes[0, 0]
    counts = {}
    for row, track_id in enumerate(tracks):
        track = events[events[S.TRACK_ID] == track_id]
        for event_type, color, height in ((S.EVENT_CRAWL, style.CRAWL_COLOR, 0.5), (S.EVENT_HEAD_CAST, style.HEAD_CAST_COLOR, 0.8)):
            part = track[track[S.EVENT_TYPE] == event_type]
            counts[event_type] = counts.get(event_type, 0) + len(part)
            ax.broken_barh(list(zip(part[S.START_S], part[S.END_S] - part[S.START_S])), (row - height / 2, height),
                           facecolor=color, linewidth=0)
    ax.set_yticks(range(len(tracks)), tracks)
    ax.set_ylim(len(tracks) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    style.label(ax, title="Ethogram", x="time (s)")
    style.legend(fig, handles=[
        Patch(facecolor=style.CRAWL_COLOR, label=f"crawl ({counts.get(S.EVENT_CRAWL, 0)})"),
        Patch(facecolor=style.HEAD_CAST_COLOR, label=f"head cast ({counts.get(S.EVENT_HEAD_CAST, 0)})"),
    ], loc="outside lower center", ncols=2)
    return fig
