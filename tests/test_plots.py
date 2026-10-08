"""Every figure renders, off screen, from a recording's saved results."""

import ast
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from larval_explorer.core import params as P
from larval_explorer.core import pipeline as PL
from larval_explorer.core import schema as S
from larval_explorer.plots import catalog, events, hmm, segmentation, steering, style, trajectory
from tests.conftest import VALIDATION_RECORDING
from tests.test_pipeline import SHORT_SEGMENTS, write_synthetic_recording

PLOTS_ROOT = Path(__file__).resolve().parents[1] / "larval_explorer" / "plots"


def draw(figure: Figure) -> None:
    assert isinstance(figure, Figure)
    FigureCanvasAgg(figure)
    figure.canvas.draw()


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """A fully processed recording: the real one if present, else a synthetic one up to stage 10."""
    folder = tmp_path_factory.mktemp("plots")
    if VALIDATION_RECORDING.exists():
        pipeline = PL.RecordingPipeline("amiGA-amRNAi_n1_att2", VALIDATION_RECORDING, folder)
    else:
        source = write_synthetic_recording(folder / "synthetic.csv")
        pipeline = PL.RecordingPipeline("synthetic", source, folder / "out", SHORT_SEGMENTS)
    pipeline.run()
    return pipeline


@pytest.mark.parametrize("stage", catalog.STAGES_WITH_FIGURES)
def test_every_stage_figure_draws(processed, stage):
    figures = catalog.stage_figures(processed, stage, max_tracks=2)
    assert figures, stage
    for name, figure in figures.items():
        draw(figure)
        assert figure.axes, name


def test_catalog_covers_every_stage_that_has_a_figure(processed):
    assert catalog.STAGES_WITH_FIGURES == PL.STAGE_KEYS[1:]
    assert catalog.stage_figures(processed, "00") == {}
    per_track = catalog.stage_figures(processed, "03", max_tracks=1)
    assert len(per_track) == 1 and next(iter(per_track)).startswith("smoothing__")
    everything = catalog.stage_figures(processed, "03")
    assert len(everything) == processed.table("03", "smoothed")[S.TRACK_ID].nunique()


def test_figures_are_saved_through_the_pipeline_not_by_the_factories(processed):
    figures = catalog.stage_figures(processed, "02")
    written = processed.save_figures("02", figures, formats=("png", "pdf"), dpi=50)
    assert sorted(path.name for path in written) == ["segmentation_overview.pdf", "segmentation_overview.png"]
    assert all(path.stat().st_size > 1000 for path in written)
    assert written[0].parent == processed.figure_folder("02")


def test_disabled_segmentation_gets_an_explanatory_figure(tmp_path):
    source = write_synthetic_recording(tmp_path / "synthetic.csv")
    params = P.PipelineParams(stage_02=P.SegmentationParams(enabled=False, min_segment_seconds=20.0))
    pipeline = PL.RecordingPipeline("synthetic", source, tmp_path / "out", params)
    pipeline.run(["02"])
    figure = catalog.stage_figures(pipeline, "02")["segmentation_overview"]
    draw(figure)
    assert "disabled" in figure.axes[0].texts[0].get_text()


def test_empty_inputs_give_a_message_instead_of_an_error():
    empty = pd.DataFrame()
    figures = [
        segmentation.tracks_overview(empty, P.CalibrationParams()),
        segmentation.coverage_timeline(empty, 10.0),
        trajectory.body_length(empty),
        trajectory.step_distributions(empty),
        hmm.phi_by_state(empty), hmm.phi_overlay(empty), hmm.state_raster(empty),
        hmm.transition_matrix([]), hmm.log_likelihood_trace([]),
        hmm.filter_comparison(empty, 3), hmm.run_length_histogram(empty, 3), hmm.hmm_segment_lengths(empty, 9),
        events.ethogram(empty),
        steering.segment_gallery(empty, empty), steering.parameter_boxes(empty),
        steering.parameter_kdes(empty, P.SteeringParams()), steering.headcast_rate(empty),
        steering.headcast_directions(empty), steering.headcast_theta_fit(empty),
        steering.autocorrelation(empty), steering.ar2_diagnostics(empty, P.SteeringParams()),
    ]
    for figure in figures:
        draw(figure)
        assert figure.axes[0].texts, "expected an explanatory message"


def test_colours_follow_the_entity_not_the_plot_order():
    assert [style.state_color(k) for k in (0, 1, 2)] == list(style.STATE_COLORS)
    assert style.state_color("2") == style.state_color(2) == style.state_color(2.0)
    assert len(set(style.CATEGORICAL)) == len(style.CATEGORICAL)
    assert style.categorical_color(0) == style.categorical_color(len(style.CATEGORICAL))

    # A table holding only state 2 still draws it in state 2's colour.
    steps = pd.DataFrame({
        S.TRACK_ID: "t", S.T_START_S: [0.0, 1.0], S.T_END_S: [1.0, 2.0], S.STATE: [2, 2],
    })
    figure = hmm.state_raster(steps)
    draw(figure)
    from matplotlib.colors import to_hex
    face = figure.axes[0].collections[0].get_facecolor()[0]
    assert to_hex(face) == style.STATE_COLORS[2]


def test_kde_curve_matches_scipy_with_seaborn_settings():
    import numpy as np
    from scipy.stats import gaussian_kde

    values = np.random.default_rng(1).normal(size=200)
    grid, density = style.kde_curve(values, bw_adjust=0.7)
    reference = gaussian_kde(values, bw_method=gaussian_kde(values).scotts_factor() * 0.7)
    np.testing.assert_allclose(density, reference(grid))
    assert len(grid) == 200 and grid[0] < values.min() and grid[-1] > values.max()
    assert style.kde_curve([1.0]) is None and style.kde_curve([2.0, 2.0, 2.0]) is None


def test_plots_never_save_and_never_use_seaborn_or_pyplot():
    """No savefig in plots/ (SPEC Phase 2); seaborn would import pyplot behind our back."""
    for path in sorted(PLOTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "savefig", f"{path.name}:{node.lineno} calls savefig"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] + [getattr(node, "module", "") or ""]
                assert not any(name.split(".")[0] == "seaborn" for name in names), f"{path.name} imports seaborn"
