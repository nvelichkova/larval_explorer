"""Stage 12 figures (T4) render from an analysis, return Figures, and export."""

import json

import pandas as pd
from matplotlib.figure import Figure

from larval_explorer.core import schema as S
from larval_explorer.core import turn_analysis as TA
from larval_explorer.plots import turns as turn_plots

from tests.test_turn_analysis import recording


def analysis():
    inputs = [recording("a1", 40, larvae=(0, 1), n_turns=3), recording("b1", 60, larvae=(0,), n_turns=3)]
    return TA.turn_analysis(inputs, {"a1": "A", "b1": "B"}, sensitivity_windows=(5,))


def test_every_turn_figure_renders_and_exports(tmp_path):
    result = analysis()
    reference = result.tables["curve_summary"].copy()
    figures = turn_plots.turn_figures(result, {"model": reference})
    assert set(figures) == set(turn_plots.FIGURE_TITLES)
    assert all(isinstance(figure, Figure) for figure in figures.values())
    manifest = TA.export_turn_analysis(tmp_path / "out", result, figures, turn_plots.FIGURE_TITLES, formats=("png",))
    entries = json.loads(manifest.read_text(encoding="utf-8"))["figures"]
    assert set(entries) == set(figures)
    assert all((tmp_path / "out" / f"{name}.png").stat().st_size > 0 for name in figures)


def test_figures_say_why_when_there_is_nothing_to_draw():
    result = analysis()
    empty = TA.TurnAnalysis(result.params, result.recording_ids, result.conditions, (), {},
                            {**result.tables, "turns": result.tables["turns"].iloc[0:0],
                             "curve_summary": result.tables["curve_summary"].iloc[0:0],
                             "smoothing_sensitivity": pd.DataFrame()}, {}, ())
    for build in (turn_plots.signed_fraction, turn_plots.bend_lag, turn_plots.smoothing_sensitivity):
        assert isinstance(build(empty), Figure)
    assert isinstance(turn_plots.heading_curves(empty, S.CURVE_PROGRESS), Figure)
