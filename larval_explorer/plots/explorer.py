"""The Trajectory Explorer figure: one track with independently switchable layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from larval_explorer.core import schema as S
from larval_explorer.plots import style

# Layer keys, in drawing order, with the label shown next to each switch.
LAYERS = {
    "raw": "Raw centre of mass",
    "bridged": "Bridged (interpolated) frames",
    "smoothed": "Smoothed centre of mass",
    "hmm_segments": "Representative HMM segments",
    "rdp": "RDP steps",
    "states": "HMM states",
    "filtered_states": "Filtered HMM states",
    "events": "Head casts",
}
DEFAULT_LAYERS = ("raw", "smoothed", "rdp")


@dataclass
class ExplorerData:
    """The tables the explorer draws from. A missing stage is an empty table."""

    trajectories: pd.DataFrame = field(default_factory=pd.DataFrame)   # stage 02
    smoothed: pd.DataFrame = field(default_factory=pd.DataFrame)       # stage 03
    rdp_steps: pd.DataFrame = field(default_factory=pd.DataFrame)      # stage 05
    hmm_steps: pd.DataFrame = field(default_factory=pd.DataFrame)      # stage 07 (or 06)
    hmm_segments: pd.DataFrame = field(default_factory=pd.DataFrame)   # stage 08
    events: pd.DataFrame = field(default_factory=pd.DataFrame)         # stage 09

    def tracks(self) -> list[str]:
        for table in (self.smoothed, self.trajectories):
            if len(table):
                return sorted(table[S.TRACK_ID].unique(), key=style.natural_key)
        return []

    def available_layers(self) -> set[str]:
        available = set()
        if len(self.trajectories):
            available.add("raw")
            if S.INTERPOLATED in self.trajectories.columns:
                available.add("bridged")
        if len(self.smoothed):
            available.add("smoothed")
        if len(self.rdp_steps):
            available.add("rdp")
        if len(self.hmm_steps) and S.STATE in self.hmm_steps.columns:
            available.add("states")
        if len(self.hmm_steps) and S.STATE_FILTERED in self.hmm_steps.columns:
            available.add("filtered_states")
        if len(self.hmm_segments):
            available.add("hmm_segments")
        if len(self.events) and len(self.smoothed):
            available.add("events")
        return available

    def path(self, track_id: str) -> pd.DataFrame:
        """The track's best available path (smoothed if present), in time order."""
        table = self.smoothed if len(self.smoothed) else self.trajectories
        if not len(table):
            return table
        return table[table[S.TRACK_ID] == track_id].sort_values(S.TIME_S)


def _of(table: pd.DataFrame, track_id: str, time_column: str) -> pd.DataFrame:
    if not len(table):
        return table
    return table[table[S.TRACK_ID] == track_id].sort_values(time_column)


def _broken(part: pd.DataFrame):
    x = np.column_stack([part[S.X0_MM], part[S.X1_MM], np.full(len(part), np.nan)]).ravel()
    y = np.column_stack([part[S.Y0_MM], part[S.Y1_MM], np.full(len(part), np.nan)]).ravel()
    return x, y


def position_at(path: pd.DataFrame, time_s: float) -> tuple[float, float]:
    """Centre-of-mass position at a time, interpolated along the path."""
    t = path[S.TIME_S].to_numpy(float)
    return (float(np.interp(time_s, t, path[S.MOM_X].to_numpy(float))),
            float(np.interp(time_s, t, path[S.MOM_Y].to_numpy(float))))


def trajectory_explorer(data: ExplorerData, track_id: str, layers, time_s: float | None = None) -> Figure:
    """One track with the chosen layers. ``fig.time_marker`` is the scrubber's dot."""
    layers = set(layers) & data.available_layers()
    path = data.path(track_id)
    if not len(path):
        return style.message_figure("Run the pipeline up to stage 02 to see trajectories.", 6.0, 5.0)
    fig, axes = style.new_figure(7.0, 6.2)
    ax = axes[0, 0]
    handles = []

    if "raw" in layers:
        raw = _of(data.trajectories, track_id, S.TIME_S)
        ax.plot(raw[S.MOM_X], raw[S.MOM_Y], color=style.DROPPED, linewidth=0.9, zorder=1)
        handles.append(Line2D([], [], color=style.DROPPED, label=LAYERS["raw"]))
    if "bridged" in layers:
        raw = _of(data.trajectories, track_id, S.TIME_S)
        bridged = raw[raw[S.INTERPOLATED].astype(bool)]
        ax.plot(bridged[S.MOM_X], bridged[S.MOM_Y], linestyle="", marker="o", markersize=3.5,
                color=style.CATEGORICAL[3], zorder=6)
        handles.append(Line2D([], [], color=style.CATEGORICAL[3], linestyle="", marker="o", markersize=4,
                              label=f"{LAYERS['bridged']} ({len(bridged)})"))
    if "smoothed" in layers:
        smoothed = _of(data.smoothed, track_id, S.TIME_S)
        ax.plot(smoothed[S.MOM_X], smoothed[S.MOM_Y], color=style.INK_SECONDARY, linewidth=1.1, zorder=2)
        handles.append(Line2D([], [], color=style.INK_SECONDARY, label=LAYERS["smoothed"]))

    hmm_steps = _of(data.hmm_steps, track_id, S.T_START_S)
    if "hmm_segments" in layers:
        segments = _of(data.hmm_segments, track_id, S.START_TIME_S)
        for _, segment in segments.iterrows():
            inside = hmm_steps[(hmm_steps[S.T_START_S] >= segment[S.START_TIME_S]) & (hmm_steps[S.T_START_S] <= segment[S.END_TIME_S])]
            if len(inside):
                x = np.append(inside[S.X0_MM].to_numpy(float), inside[S.X1_MM].iloc[-1])
                y = np.append(inside[S.Y0_MM].to_numpy(float), inside[S.Y1_MM].iloc[-1])
                ax.plot(x, y, color=style.state_color(segment[S.STATE]), linewidth=7, alpha=0.25,
                        solid_capstyle="round", zorder=3)
        handles.append(Line2D([], [], color=style.REFERENCE, linewidth=6, alpha=0.3,
                              label=f"{LAYERS['hmm_segments']} ({len(segments)})"))
    if "rdp" in layers:
        steps = _of(data.rdp_steps, track_id, S.T_START_S)
        if len(steps):
            x = np.append(steps[S.X0_MM].to_numpy(float), steps[S.X1_MM].iloc[-1])
            y = np.append(steps[S.Y0_MM].to_numpy(float), steps[S.Y1_MM].iloc[-1])
            ax.plot(x, y, color=style.INK, linewidth=0.9, marker="o", markersize=3, zorder=4)
        handles.append(Line2D([], [], color=style.INK, marker="o", markersize=3, label=f"{LAYERS['rdp']} ({len(steps)})"))

    state_layers = [("filtered_states", S.STATE_FILTERED)] if "filtered_states" in layers else []
    if "states" in layers and not state_layers:
        state_layers = [("states", S.STATE)]
    shown_states = set()
    for _, column in state_layers:
        for state in sorted(int(value) for value in hmm_steps[column].dropna().unique()):
            x, y = _broken(hmm_steps[hmm_steps[column] == state])
            ax.plot(x, y, color=style.state_color(state), linewidth=2.0, zorder=5)
            shown_states.add(state)
    handles.extend(Patch(facecolor=style.state_color(state), label=style.state_label(state)) for state in sorted(shown_states))

    if "events" in layers:
        smoothed = _of(data.smoothed, track_id, S.TIME_S)
        events = _of(data.events, track_id, S.START_S)
        head_casts = events[events[S.EVENT_TYPE] == S.EVENT_HEAD_CAST]
        t = smoothed[S.TIME_S].to_numpy(float)
        for start, end in zip(head_casts[S.START_S], head_casts[S.END_S]):
            part = smoothed[(t >= start) & (t <= end)]
            ax.plot(part[S.MOM_X], part[S.MOM_Y], color=style.HEAD_CAST_COLOR, linewidth=2.6, zorder=7)
        handles.append(Line2D([], [], color=style.HEAD_CAST_COLOR, linewidth=2.6,
                              label=f"{LAYERS['events']} ({len(head_casts)})"))

    marker_time = float(path[S.TIME_S].iloc[0]) if time_s is None else float(time_s)
    marker_x, marker_y = position_at(path, marker_time)
    (fig.time_marker,) = ax.plot([marker_x], [marker_y], marker="o", markersize=9, markerfacecolor="none",
                                 markeredgecolor=style.INK, markeredgewidth=1.6, linestyle="", zorder=10)
    fig.time_marker.set_visible(time_s is not None)

    style.equal_aspect(ax)
    ax.set_aspect("equal", adjustable="box")   # so a zoomed view can be restored exactly
    duration = float(path[S.TIME_S].iloc[-1] - path[S.TIME_S].iloc[0])
    style.label(ax, title=f"{track_id}   {path[S.TIME_S].iloc[0]:.1f}-{path[S.TIME_S].iloc[-1]:.1f} s ({duration:.0f} s)")
    if handles:
        style.legend(fig, handles=handles, loc="outside lower center", ncols=min(3, len(handles)))
    return fig
