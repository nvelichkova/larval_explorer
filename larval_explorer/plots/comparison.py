"""Cross-condition figures (Phase 5).

Every point is one unit: a track segment, a larva or a recording, as chosen.
Boxes show the median and quartiles of the units. Each condition's label gives
n at all three levels, so a figure cannot be read as having more independent
observations than it has.

Conditions are told apart by position and label. Colour is reserved for HMM
states, which keep the colours they have everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from larval_explorer.core import aggregate as A
from larval_explorer.core import schema as S
from larval_explorer.plots import style

JITTER_SEED = 0
BOX_WIDTH = 0.22
N_STATES = A.N_STATES

MEASURE_LABELS = {
    A.MEASURE_OCCUPANCY: ("State occupancy", "fraction of time"),
    A.MEASURE_DWELL: ("Dwell time", "mean stay in the state (s)"),
    A.MEASURE_HEAD_CAST_RATE: ("Head-cast rate", "head casts per minute tracked"),
    A.MEASURE_CRAWL_LENGTH: ("Crawl length", "mean crawl length (mm)"),
    S.RHO: ("rho", "rho"),
    S.KAPPA: ("kappa", "kappa"),
    S.MU: ("mu", "mu"),
    S.SIGMA: ("sigma", "sigma"),
}

FIGURE_TITLES = {
    "state_occupancy": "State occupancy by condition",
    "dwell_times": "Dwell time by state and condition",
    "steering_parameters": "Steering parameters by state and condition",
    "head_cast_rate": "Head-cast rate by condition",
    "crawl_length": "Crawl length by condition",
    "hmm_per_condition": "HMM states: pooled fit against per-condition fits",
}


def _unit_plural(unit: str) -> str:
    return {A.UNIT_SEGMENT: "track segments", A.UNIT_LARVA: "larvae", A.UNIT_RECORDING: "recordings"}[unit]


def _condition_ticks(comparison: A.Comparison) -> list[str]:
    counts = comparison.counts.set_index(S.CONDITION)
    labels = []
    for condition in comparison.conditions:
        row = counts.loc[condition]
        labels.append(
            f"{condition}\n{int(row[S.N_RECORDINGS])} recordings\n{int(row[S.N_LARVAE])} larvae\n"
            f"{int(row[S.N_TRACK_SEGMENTS])} track segments"
        )
    return labels


def _box(ax, position: float, values: np.ndarray, color: str, rng) -> None:
    """Median and quartiles of the units, the units themselves, and how many there are."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    ax.scatter(position + 0.04 * rng.standard_normal(len(values)), values, s=14, alpha=0.55, color=color,
               linewidths=0, zorder=3)
    if len(values) >= 3:
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        half = BOX_WIDTH / 2
        ax.plot([position - half, position + half, position + half, position - half, position - half],
                [q25, q25, q75, q75, q25], color=color, linewidth=1.1, zorder=2)
        ax.plot([position - half, position + half], [q50, q50], color=color, linewidth=2.0, zorder=2)
    else:
        ax.plot([position - BOX_WIDTH / 2, position + BOX_WIDTH / 2], [np.median(values)] * 2, color=color,
                linewidth=2.0, zorder=2)
    ax.annotate(str(len(values)), (position, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=style.TICK_SIZE - 1,
                color=style.INK_SECONDARY)


def _finish(ax, comparison: A.Comparison, title: str, y_label: str, show_ticks: bool = True) -> None:
    positions = np.arange(len(comparison.conditions))
    ax.set_xticks(positions, _condition_ticks(comparison) if show_ticks else [""] * len(positions))
    ax.set_xlim(-0.6, len(positions) - 0.4)
    style.style_axes(ax, grid="y")
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=style.TITLE_SIZE, color=style.INK, loc="left", pad=14)
    style.label(ax, y=y_label)


def _draw_by_state(ax, comparison: A.Comparison, measure: str, rng) -> None:
    values = comparison.values[comparison.values[S.MEASURE] == measure]
    offsets = (np.arange(N_STATES) - (N_STATES - 1) / 2) * (BOX_WIDTH + 0.05)
    for index, condition in enumerate(comparison.conditions):
        for state in range(N_STATES):
            part = values[(values[S.CONDITION] == condition) & (values[S.STATE] == state)]
            _box(ax, index + offsets[state], part[S.VALUE].to_numpy(float), style.state_color(state), rng)


def _draw_plain(ax, comparison: A.Comparison, measure: str, rng) -> None:
    values = comparison.values[comparison.values[S.MEASURE] == measure]
    for index, condition in enumerate(comparison.conditions):
        part = values[values[S.CONDITION] == condition]
        _box(ax, index, part[S.VALUE].to_numpy(float), style.INK, rng)


def _caption(fig: Figure, comparison: A.Comparison, by_state: bool) -> None:
    text = (f"Each point is one {comparison.unit}; boxes are the median and quartiles of the "
            f"{_unit_plural(comparison.unit)}.\n"
            f"The number above each box is how many {_unit_plural(comparison.unit)} contribute.")
    fig.supxlabel(text, fontsize=style.TICK_SIZE, color=style.INK_SECONDARY, x=0.01, ha="left")
    if by_state:
        handles = [Patch(facecolor=style.state_color(k), label=style.state_label(k)) for k in range(N_STATES)]
        style.legend(fig, handles=handles, loc="outside upper right", ncols=N_STATES)


def _width(comparison: A.Comparison, by_state: bool) -> float:
    return max(4.6, 1.3 + (2.0 if by_state else 1.5) * len(comparison.conditions))


def by_state(comparison: A.Comparison, measure: str) -> Figure:
    """One measure that exists per HMM state, by condition."""
    if not (comparison.values[S.MEASURE] == measure).any():
        return style.message_figure(f"No values for {MEASURE_LABELS[measure][0].lower()}.")
    title, y_label = MEASURE_LABELS[measure]
    fig, axes = style.new_figure(_width(comparison, True), 4.4)
    _draw_by_state(axes[0, 0], comparison, measure, np.random.default_rng(JITTER_SEED))
    _finish(axes[0, 0], comparison, title, y_label)
    if measure == A.MEASURE_OCCUPANCY:
        axes[0, 0].set_ylim(-0.02, 1.02)
    _caption(fig, comparison, by_state=True)
    return fig


def plain(comparison: A.Comparison, measure: str) -> Figure:
    """One measure that has no state, by condition."""
    if not (comparison.values[S.MEASURE] == measure).any():
        return style.message_figure(f"No values for {MEASURE_LABELS[measure][0].lower()}.")
    title, y_label = MEASURE_LABELS[measure]
    fig, axes = style.new_figure(_width(comparison, False), 4.4)
    _draw_plain(axes[0, 0], comparison, measure, np.random.default_rng(JITTER_SEED))
    _finish(axes[0, 0], comparison, title, y_label)
    axes[0, 0].set_ylim(bottom=0)
    _caption(fig, comparison, by_state=False)
    return fig


def steering_parameters(comparison: A.Comparison) -> Figure:
    """The four steering parameters by state and condition, one panel each."""
    present = [m for m in A.STEERING_MEASURES if (comparison.values[S.MEASURE] == m).any()]
    if not present:
        return style.message_figure("No steering fits to compare.")
    fig, axes = style.new_figure(_width(comparison, True) * 2, 7.6, 2, 2)
    rng = np.random.default_rng(JITTER_SEED)
    for ax, measure in zip(axes.flat, A.STEERING_MEASURES):
        _draw_by_state(ax, comparison, measure, rng)
        _finish(ax, comparison, MEASURE_LABELS[measure][0], MEASURE_LABELS[measure][1])
    _caption(fig, comparison, by_state=True)
    return fig


def hmm_per_condition(diagnostic: pd.DataFrame) -> Figure:
    """Fitted mean phi of each state in the pooled fit and in a fit per condition.

    If a condition's points sit far from the pooled ones, the shared states
    describe that condition poorly.
    """
    if diagnostic.empty:
        return style.message_figure("No per-condition fits to show.")
    fits = list(dict.fromkeys(diagnostic[S.FIT]))
    fig, axes = style.new_figure(max(4.6, 1.6 + 1.5 * len(fits)), 4.2)
    ax = axes[0, 0]
    for state in range(N_STATES):
        part = diagnostic[diagnostic[S.STATE] == state].set_index(S.FIT).reindex(fits)
        ax.plot(range(len(fits)), part[S.MEAN_PHI], marker="o", markersize=6, linewidth=1.0, linestyle=":",
                color=style.state_color(state), label=style.state_label(state))
        for position, steps in enumerate(part[S.N_STEPS]):
            if np.isfinite(part[S.MEAN_PHI].iloc[position]):
                ax.annotate(f"{int(steps)}", (position, part[S.MEAN_PHI].iloc[position]), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=style.TICK_SIZE - 1,
                            color=style.INK_SECONDARY)
    ax.axhline(0, color=style.REFERENCE, linewidth=0.8)
    first = diagnostic.groupby(S.FIT, sort=False).first().reindex(fits)
    labels = [f"{fit}\n{int(n)} recordings" + ("" if ok else "\nnot converged")
              for fit, n, ok in zip(fits, first[S.N_RECORDINGS], first[S.CONVERGED])]
    ax.set_xticks(range(len(fits)), labels)
    ax.set_xlim(-0.5, len(fits) - 0.3)
    style.style_axes(ax, grid="y")
    ax.set_title("Fitted mean phi per state: pooled fit against a fit per condition",
                 fontsize=style.TITLE_SIZE, color=style.INK, loc="left")
    style.label(ax, y="fitted mean phi (rad)")
    fig.supxlabel("Numbers beside the points are the steps decoded into that state.",
                  fontsize=style.TICK_SIZE, color=style.INK_SECONDARY, x=0.01, ha="left")
    style.legend(fig, loc="outside upper right", ncols=N_STATES)
    return fig


def comparison_figures(comparison: A.Comparison, diagnostic: pd.DataFrame | None = None) -> dict[str, Figure]:
    """The whole comparison set, by figure name."""
    figures = {
        "state_occupancy": by_state(comparison, A.MEASURE_OCCUPANCY),
        "dwell_times": by_state(comparison, A.MEASURE_DWELL),
        "steering_parameters": steering_parameters(comparison),
        "head_cast_rate": plain(comparison, A.MEASURE_HEAD_CAST_RATE),
        "crawl_length": plain(comparison, A.MEASURE_CRAWL_LENGTH),
    }
    if diagnostic is not None and len(diagnostic):
        figures["hmm_per_condition"] = hmm_per_condition(diagnostic)
    return figures
