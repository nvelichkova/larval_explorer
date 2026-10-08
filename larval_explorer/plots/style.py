"""Shared figure style: colours, fonts, and helpers every factory uses.

Colours follow the entity, never the plot order (SPEC Phase 2): a state, a
track segment or an event type has the same colour in every figure and for
every dataset, because it is looked up by its canonical index.

Figures are built from ``matplotlib.figure.Figure`` directly. pyplot is never
imported here (D-005), and nothing in ``plots/`` writes a file.
"""

from __future__ import annotations

import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
GAP = "#e8e8e6"          # untracked time
DROPPED = "#bdbdb9"      # tracked, but in a segment too short to keep
REFERENCE = "#8a8984"    # thresholds, zero lines, naive chords

# Fixed categorical order. The first three are distinguishable pairwise under
# colour-vision deficiency; the rest are safe as neighbours in this order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# HMM states by stored index (D-016): the hues upstream used (blue, orange,
# green), stepped so that orange and green stay apart for red-green deficiency.
STATE_COLORS = CATEGORICAL[:3]

CRAWL_COLOR = INK
HEAD_CAST_COLOR = "#e34948"

SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 10, 9, 8
LINE_WIDTH = 1.4


def state_color(state) -> str:
    """Colour of an HMM state, from its stored index."""
    return STATE_COLORS[int(state) % len(STATE_COLORS)]


def state_label(state) -> str:
    return f"State {int(state)}"


def categorical_color(index: int) -> str:
    """Colour of the index-th entity of a kind (track segment, larva)."""
    return CATEGORICAL[int(index) % len(CATEGORICAL)]


def short_track(track_id) -> str:
    """A track segment ID without its recording prefix: 'larva3__seg1'."""
    text = str(track_id)
    return text.split("__", 1)[1] if "__" in text else text


def natural_key(text) -> list:
    """Sort key that orders 'seg2' before 'seg10'."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def new_figure(width: float, height: float, nrows: int = 1, ncols: int = 1, polar: bool = False, **subplot_kw):
    """A styled figure and its axes (always a 2-D array of axes)."""
    fig = Figure(figsize=(width, height), dpi=100, layout="constrained", facecolor=SURFACE)
    if polar:
        subplot_kw["subplot_kw"] = {"projection": "polar"}
    axes = fig.subplots(nrows, ncols, squeeze=False, **subplot_kw)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes


def style_axes(ax, grid: str | None = None) -> None:
    """Recessive frame and ticks; an optional light grid on 'x', 'y' or 'both'."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=TICK_SIZE, colors=INK_SECONDARY, width=0.6, length=3)
    ax.xaxis.label.set(size=LABEL_SIZE, color=INK_SECONDARY)
    ax.yaxis.label.set(size=LABEL_SIZE, color=INK_SECONDARY)
    ax.title.set(size=TITLE_SIZE, color=INK)
    if ax.name == "polar":
        ax.grid(True, linewidth=0.4, color=GRID)
        ax.spines["polar"].set(color=GRID, linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 1)
        return
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set(color=INK_SECONDARY, linewidth=0.6)
    if grid:
        ax.grid(True, axis=grid, linewidth=0.4, color=GRID)
        ax.set_axisbelow(True)


def label(ax, title: str | None = None, x: str | None = None, y: str | None = None) -> None:
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, color=INK, loc="left")
    if x is not None:
        ax.set_xlabel(x, fontsize=LABEL_SIZE, color=INK_SECONDARY)
    if y is not None:
        ax.set_ylabel(y, fontsize=LABEL_SIZE, color=INK_SECONDARY)


def legend(target, *args, **kwargs):
    """A frameless legend in text colours, on an axes or a figure."""
    kwargs.setdefault("fontsize", TICK_SIZE)
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("labelcolor", INK_SECONDARY)
    return target.legend(*args, **kwargs)


def equal_aspect(ax, x_label: str = "x (mm)", y_label: str = "y (mm)") -> None:
    """A map of positions: one millimetre is the same length on both axes."""
    ax.set_aspect("equal", adjustable="datalim")
    label(ax, x=x_label, y=y_label)


def scale_bar(ax, length: float, text: str) -> None:
    """A horizontal bar of ``length`` data units in the lower-left corner."""
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    bar = AnchoredSizeBar(
        ax.transData, length, text, loc="lower left", frameon=False, color=INK, pad=0.4, sep=3,
        fontproperties=FontProperties(size=TICK_SIZE),
    )
    ax.add_artist(bar)


def message_figure(text: str, width: float = 5.0, height: float = 3.0) -> Figure:
    """A figure that says why there is nothing to draw."""
    fig, axes = new_figure(width, height)
    ax = axes[0, 0]
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=LABEL_SIZE, color=INK_SECONDARY,
            transform=ax.transAxes, wrap=True)
    return fig


def kde_curve(values, bw_adjust: float = 1.0, cut: float = 3.0, gridsize: int = 200):
    """Gaussian kernel density on a grid, as seaborn's ``kdeplot`` computes it.

    Scott's bandwidth times ``bw_adjust``, evaluated from ``cut`` bandwidths
    below the minimum to ``cut`` above the maximum. Returns ``(grid, density)``
    or ``None`` when there are too few distinct values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    kde = gaussian_kde(values, bw_method=lambda estimator: estimator.scotts_factor() * bw_adjust)
    bandwidth = float(np.sqrt(kde.covariance.squeeze()))
    grid = np.linspace(values.min() - cut * bandwidth, values.max() + cut * bandwidth, gridsize)
    return grid, kde(grid)


def grid_shape(n: int, max_columns: int = 4) -> tuple[int, int]:
    """Rows and columns for ``n`` small panels."""
    columns = max(1, min(max_columns, n))
    return int(np.ceil(n / columns)), columns
