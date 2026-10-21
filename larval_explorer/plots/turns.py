"""Stage 12 figures (T4): heading curves, signed fraction, migration lag, sensitivity, censoring.

Every factory takes a ``turn_analysis.TurnAnalysis`` and returns a ``Figure``;
nothing here writes a file. Conditions are told apart by position and label,
as in the comparison figures (D-025). The two heading axes have fixed colours:
the model-matched axis ``tail_sp1`` in ink, the primary biological axis
``tail_sp2`` in the first categorical colour. Model reference curves, when
given, are drawn in the reference grey.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from larval_explorer.core import schema as S
from larval_explorer.core import turn_analysis as TA
from larval_explorer.plots import style

JITTER_SEED = 0
BOX_WIDTH = 0.22
MIN_CURVE_TURNS = 5      # a distance or time curve is drawn only where this many turns still contribute
REFERENCE_DASHES = ((0, (2, 2)), (0, (6, 2, 1, 2)), (0, (1, 1)), (0, (8, 3)))   # one per model reference
OUTLIER_PERCENTILES = (1, 99)   # the signed-fraction axes span these; points beyond are counted at the edge
AXIS_COLORS = {"tail_sp1": style.INK, "tail_sp2": style.CATEGORICAL[0]}
AXIS_LABELS = {"tail_sp1": "tail→sp1 (model-matched)", "tail_sp2": "tail→sp2 (primary)"}
CURVE_LABELS = {
    S.CURVE_PROGRESS: "turn progress (onset 0, end 1)",
    S.CURVE_DISTANCE: "distance crawled since onset (segment lengths)",
    S.CURVE_TIME: "time since onset (s)",
}

FIGURE_TITLES = {
    "heading_curves_progress": "Heading change through the turn",
    "heading_curves_distance": "Heading change against distance crawled",
    "signed_fraction": "Fraction of the turn done by the cast end (signed)",
    "bend_lag": "Bend migration: sp1 → sp2",
    "smoothing_sensitivity": "Sensitivity to spine smoothing",
    "censoring": "Turns dropped by each filter",
}


def _headline(analysis: TA.TurnAnalysis) -> pd.DataFrame:
    return TA._headline_set(analysis.tables["turns"])


def _conditions(analysis: TA.TurnAnalysis) -> list[str]:
    return sorted(set(analysis.conditions.values()))


def _condition_tick(analysis: TA.TurnAnalysis, condition: str, turns: pd.DataFrame) -> str:
    mine = turns[turns[S.CONDITION] == condition]
    return (f"{condition}\n{mine[S.RECORDING_ID].nunique()} recordings\n"
            f"{mine[S.LARVA_ID].nunique()} larvae\n{len(mine)} turns")


def _box(ax, position: float, values: np.ndarray, color: str, rng, larva_means: np.ndarray | None = None) -> None:
    """Turns as points, median and quartiles as a box, larva means as open diamonds, n above."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    ax.scatter(position + 0.04 * rng.standard_normal(len(values)), values, s=10, alpha=0.4, color=color,
               linewidths=0, zorder=3)
    if larva_means is not None and len(larva_means):
        ax.scatter(np.full(len(larva_means), position + BOX_WIDTH / 2 + 0.08), larva_means, s=22, marker="D",
                   facecolors="none", edgecolors=color, linewidths=0.9, zorder=4)
    half = BOX_WIDTH / 2
    if len(values) >= 3:
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([position - half, position + half, position + half, position - half, position - half],
                [q25, q25, q75, q75, q25], color=color, linewidth=1.1, zorder=2)
        ax.plot([position - half, position + half], [q50, q50], color=color, linewidth=2.0, zorder=2)
    ax.annotate(str(len(values)), (position, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=style.TICK_SIZE - 1,
                color=style.INK_SECONDARY)


def _clip_outliers(ax, values: np.ndarray) -> None:
    """Limit the y-axis to the central percentiles (padded) and say how many points fall outside.

    A fraction is a ratio: a turn just above the 15 deg minimum can give values
    of several units, which would flatten every other point.
    """
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return
    lo, hi = np.percentile(values, OUTLIER_PERCENTILES)
    lo, hi = min(lo, -0.1), max(hi, 0.1)
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    below, above = int((values < lo - pad).sum()), int((values > hi + pad).sum())
    if below or above:
        ax.annotate(f"off scale: {below} below, {above} above", (1.0, 1.0), xycoords="axes fraction", xytext=(-2, -2),
                    textcoords="offset points", ha="right", va="top", fontsize=style.TICK_SIZE - 1, color=style.INK_SECONDARY)


def _condition_axis(ax, analysis, turns, title: str, y_label: str) -> None:
    conditions = _conditions(analysis)
    ax.set_xticks(np.arange(len(conditions)), [_condition_tick(analysis, c, turns) for c in conditions])
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    style.style_axes(ax, grid="y")
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=style.TITLE_SIZE, color=style.INK, loc="left", pad=14)
    style.label(ax, y=y_label)


def heading_curves(analysis: TA.TurnAnalysis, curve: str = S.CURVE_PROGRESS,
                   references: dict[str, pd.DataFrame] | None = None) -> Figure:
    """Median cumulative signed heading change (fraction of the turn's total), with quartiles.

    One panel per condition; both heading axes drawn. On the progress abscissa
    the median cast end (sp1 bend peak, solid) and |head angle| peak (dashed)
    are marked. ``references`` maps a label to a ``curve_summary`` table, e.g.
    the model through the synthetic control, drawn as grey dashed medians.
    """
    summary = analysis.tables["curve_summary"]
    summary = summary[summary[S.CURVE] == curve]
    if summary.empty:
        return style.message_figure("No heading curves: no turn has a fraction.")
    turns = _headline(analysis)
    conditions = _conditions(analysis)
    fig, axes = style.new_figure(max(6.0, 3.6 * len(conditions)), 3.8, ncols=len(conditions), sharey=True)
    for ax, condition in zip(axes[0], conditions):
        mine = summary[summary[S.CONDITION] == condition]
        n_max = mine[S.N_TURNS].max()
        for axis in (analysis.params.model_heading_axis, analysis.params.heading_axis):
            line = mine[mine[S.HEADING_AXIS] == axis].sort_values(S.CURVE_X)
            if curve != S.CURVE_PROGRESS:
                line = line[line[S.N_TURNS] >= max(MIN_CURVE_TURNS, 0.2 * n_max)]
            if line.empty:
                continue
            color = AXIS_COLORS.get(axis, style.INK_SECONDARY)
            ax.fill_between(line[S.CURVE_X], line[S.Q25], line[S.Q75], color=color, alpha=0.12, linewidth=0)
            ax.plot(line[S.CURVE_X], line[S.MEDIAN], color=color, linewidth=style.LINE_WIDTH)
        for (label, reference), dash in zip((references or {}).items(), REFERENCE_DASHES):
            ref = reference[(reference[S.CURVE] == curve)
                            & (reference[S.HEADING_AXIS] == analysis.params.model_heading_axis)].sort_values(S.CURVE_X)
            if curve != S.CURVE_PROGRESS and len(ref):
                ref = ref[ref[S.N_TURNS] >= max(MIN_CURVE_TURNS, 0.2 * ref[S.N_TURNS].max())]
            ax.plot(ref[S.CURVE_X], ref[S.MEDIAN], color=style.REFERENCE, linewidth=1.1, linestyle=dash)
        if curve == S.CURVE_PROGRESS:
            here = turns[turns[S.CONDITION] == condition]
            for column, dash in ((S.SWEEP_END_PROGRESS, "-"), (S.HEAD_ANGLE_PEAK_PROGRESS, (0, (5, 3)))):
                ax.axvline(here[column].median(), color=style.REFERENCE, linewidth=0.8, linestyle=dash, zorder=1)
        ax.axhline(0, color=style.REFERENCE, linewidth=0.6)
        ax.axhline(1, color=style.GRID, linewidth=0.6)
        style.style_axes(ax, grid="both")
        style.label(ax, title=_condition_tick(analysis, condition, turns).replace("\n", ", "), x=CURVE_LABELS[curve])
    style.label(axes[0, 0], y="heading change / turn total (signed)")
    handles = [Line2D([], [], color=AXIS_COLORS[a], linewidth=style.LINE_WIDTH, label=AXIS_LABELS[a]) for a in AXIS_COLORS]
    handles += [Line2D([], [], color=style.REFERENCE, linewidth=1.1, linestyle=dash, label=label)
                for label, dash in zip(references or {}, REFERENCE_DASHES)]
    if curve == S.CURVE_PROGRESS:
        handles += [Line2D([], [], color=style.REFERENCE, linewidth=0.8, label="cast end (median)"),
                    Line2D([], [], color=style.REFERENCE, linewidth=0.8, linestyle=(0, (5, 3)), label="|head angle| peak (median)")]
    style.legend(fig, handles=handles, loc="outside lower center", ncols=2)
    return fig


def signed_fraction(analysis: TA.TurnAnalysis) -> Figure:
    """Signed fraction at the cast end and at the |head angle| peak, model-matched axis, by condition."""
    turns = _headline(analysis)
    if turns.empty:
        return style.message_figure("No turns with a fraction.")
    panels = (
        (S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED), "at the cast end (sp1 bend peak)"),
        (S.model_axis_column(S.FRAC_SIGNED_AT_HEAD_ANGLE_PEAK), "at the |head angle| peak"),
    )
    conditions = _conditions(analysis)
    fig, axes = style.new_figure(max(6.4, 2.8 * len(conditions) * 2), 4.2, ncols=2, sharey=True)
    rng = np.random.default_rng(JITTER_SEED)
    for ax, (column, title) in zip(axes[0], panels):
        for index, condition in enumerate(conditions):
            mine = turns[turns[S.CONDITION] == condition]
            values = mine[column].to_numpy(float)
            larva = mine.groupby(S.LARVA_ID)[column].mean().to_numpy(float)
            _box(ax, index, values, style.INK, rng, larva)
            finite = values[np.isfinite(values)]
            if len(finite):
                ax.annotate(f"{np.mean(finite < 0):.0%} negative", (index, 0.0), xycoords=("data", "axes fraction"),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=style.TICK_SIZE - 1,
                            color=style.INK_SECONDARY)
        ax.axhline(0, color=style.REFERENCE, linewidth=0.8)
        _clip_outliers(ax, turns[column].to_numpy(float))
        _condition_axis(ax, analysis, turns, title, "signed fraction, tail→sp1 axis")
    fig.supxlabel("Points are turns; boxes are their median and quartiles; diamonds are larva means.\n"
                  "Below 0: the posterior turned against the final direction during the cast.",
                  fontsize=style.TICK_SIZE, color=style.INK_SECONDARY, x=0.01, ha="left")
    return fig


def bend_lag(analysis: TA.TurnAnalysis) -> Figure:
    """sp1 → sp2 bend-peak lag (s) and forward distance crawled during it (segment lengths)."""
    turns = _headline(analysis)
    if turns.empty:
        return style.message_figure("No turns with a fraction.")
    panels = ((S.BEND_LAG_12_S, "lag (s)"), (S.BEND_LAG_12_SEGLEN, "crawled during the lag (segment lengths)"))
    conditions = _conditions(analysis)
    fig, axes = style.new_figure(max(6.4, 2.8 * len(conditions) * 2), 4.2, ncols=2)
    rng = np.random.default_rng(JITTER_SEED)
    for ax, (column, y_label) in zip(axes[0], panels):
        for index, condition in enumerate(conditions):
            mine = turns[turns[S.CONDITION] == condition]
            _box(ax, index, mine[column].to_numpy(float), style.INK, rng,
                 mine.groupby(S.LARVA_ID)[column].mean().to_numpy(float))
        ax.axhline(0, color=style.REFERENCE, linewidth=0.6)
        _condition_axis(ax, analysis, turns, y_label.split(" (")[0].capitalize(), y_label)
    fig.supxlabel("From the sp1 bend peak (cast end) to the sp2 bend peak, in the cast's direction.\n"
                  "Segment length = straight-crawl 5-point polyline length / 11 (D-041).",
                  fontsize=style.TICK_SIZE, color=style.INK_SECONDARY, x=0.01, ha="left")
    return fig


def smoothing_sensitivity(analysis: TA.TurnAnalysis) -> Figure:
    """Median signed fraction and share negative against the spine smoothing window, both axes."""
    table = analysis.tables.get("smoothing_sensitivity", pd.DataFrame())
    if table.empty:
        return style.message_figure("No smoothing sensitivity in this analysis.")
    fig, axes = style.new_figure(7.0, 3.4, ncols=2)
    measures = {S.FRAC_HEADING_IN_SWEEP_SIGNED: "tail_sp2", S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED): "tail_sp1"}
    markers = ("o", "s", "^", "v", "D")
    for c_index, condition in enumerate(_conditions(analysis)):
        for measure, axis in measures.items():
            rows = table[(table[S.CONDITION] == condition) & (table[S.MEASURE] == measure)].sort_values(S.SPINE_SG_WINDOW)
            for ax, column in zip(axes[0], (S.TURN_MEDIAN, S.SHARE_NEGATIVE)):
                ax.plot(rows[S.SPINE_SG_WINDOW], rows[column], color=AXIS_COLORS[axis], marker=markers[c_index % 5],
                        markersize=4, linewidth=1.0, label=f"{condition}, {AXIS_LABELS[axis]}")
    for ax, (title, y_label) in zip(axes[0], (("Median signed fraction", "at the cast end"), ("Share negative", "fraction of turns"))):
        style.style_axes(ax, grid="both")
        style.label(ax, title=title, x="Savitzky-Golay window (frames)", y=y_label)
    axes[0, 0].axhline(0, color=style.REFERENCE, linewidth=0.6)
    low, high = axes[0, 0].get_ylim()
    middle, half = (low + high) / 2, max((high - low) / 2, 0.15)   # a small change must look small
    axes[0, 0].set_ylim(middle - half, middle + half)
    axes[0, 1].set_ylim(0, 1)
    style.legend(fig, *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    return fig


def censoring(analysis: TA.TurnAnalysis) -> Figure:
    """Turns dropped by each filter, per recording; filters that drop larger turns are marked."""
    table = analysis.tables["censoring"]
    table = table[table[S.FILTER] != S.CANDIDATES]
    recordings = list(dict.fromkeys(table[S.RECORDING_ID]))
    filters = list(dict.fromkeys(table[S.FILTER]))
    fig, axes = style.new_figure(7.0, 0.5 + 0.32 * len(filters) * max(1, len(recordings)) / 1.4)
    ax = axes[0, 0]
    height = 0.8 / max(1, len(recordings))
    for r_index, recording in enumerate(recordings):
        mine = table[table[S.RECORDING_ID] == recording].set_index(S.FILTER).reindex(filters)
        y = np.arange(len(filters)) + (r_index - (len(recordings) - 1) / 2) * height
        color = style.categorical_color(r_index)
        ax.barh(y, mine[S.N_DROPPED].fillna(0), height=height * 0.9, color=color, label=recording)
        for yy, dropped, biased in zip(y, mine[S.N_DROPPED].fillna(0), mine[S.BIASED_TOWARD_LARGE_TURNS].astype("boolean").fillna(False)):
            if biased:
                ax.annotate("larger turns dropped", (dropped, yy), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=style.TICK_SIZE - 1, color=style.HEAD_CAST_COLOR)
    ax.set_yticks(np.arange(len(filters)), filters)
    ax.invert_yaxis()
    ax.set_xlim(0, table[S.N_DROPPED].max() * 1.45 + 1)   # room for the bias notes
    not_exclusions = [i for i, name in enumerate(filters) if name in (S.SMALL_TURN, S.SENSITIVITY_NOT_WELL_ORIENTED)]
    if not_exclusions:
        ax.axhline(min(not_exclusions) - 0.5, color=style.REFERENCE, linewidth=0.8, linestyle=(0, (4, 2)))
        ax.annotate("below: not exclusions (fraction blanked; sensitivity check)", (1.0, min(not_exclusions) - 0.5),
                    xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points", ha="right",
                    va="bottom", fontsize=style.TICK_SIZE - 1, color=style.INK_SECONDARY)
    style.style_axes(ax, grid="x")
    style.label(ax, title="Turns dropped by each filter, in the order applied", x="turns dropped")
    style.legend(fig, loc="outside lower center", ncols=max(1, len(recordings)))
    return fig


def turn_figures(analysis: TA.TurnAnalysis, references: dict[str, pd.DataFrame] | None = None) -> dict[str, Figure]:
    """Every stage 12 figure, by name (see ``FIGURE_TITLES``)."""
    return {
        "heading_curves_progress": heading_curves(analysis, S.CURVE_PROGRESS, references),
        "heading_curves_distance": heading_curves(analysis, S.CURVE_DISTANCE, references),
        "signed_fraction": signed_fraction(analysis),
        "bend_lag": bend_lag(analysis),
        "smoothing_sensitivity": smoothing_sensitivity(analysis),
        "censoring": censoring(analysis),
    }
