"""Phase 5: the pooled HMM fit, aggregation by condition, and the exported figure set."""

import json
import logging

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from larval_explorer.core import aggregate as A
from larval_explorer.core import params as P
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.core import schema as S
from larval_explorer.core import stages
from larval_explorer.core.project import Project
from larval_explorer.plots import comparison as C
from tests.test_pipeline import write_synthetic_recording

PARAMS = P.PipelineParams(stage_02=P.SegmentationParams(min_segment_seconds=20.0))
PATTERN = r"(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)"
RECORDINGS = ["amiGA_n1_att1", "amiGA_n2_att1", "w1118_n1_att1", "w1118_n2_att1"]


@pytest.fixture(autouse=True)
def quiet_hmmlearn():
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)   # tiny synthetic fits are under-determined


def make_project(folder, params=PARAMS):
    data = folder / "data"
    data.mkdir(parents=True)
    for seed, name in enumerate(RECORDINGS, start=1):
        write_synthetic_recording(data / f"{name}.csv", seed=seed)
    project = Project.create(folder / "project")
    project.registry = R.add_recordings(project.registry, [data])
    project.registry = R.apply_parse(project.registry, R.preview_parse(project.registry, PATTERN))
    project.save()
    return project


@pytest.fixture(scope="module")
def pooled_project(tmp_path_factory):
    """Four recordings of two genotypes, processed with one pooled HMM fit."""
    project = make_project(tmp_path_factory.mktemp("pooled"))
    project.outcome = A.run_pooled_analysis(project, RECORDINGS, PARAMS)
    return project


# ──────────────────────────────────────────────────────────────────────────────
# Pooled HMM
# ──────────────────────────────────────────────────────────────────────────────

def test_pooled_fit_gives_every_recording_the_same_states(pooled_project):
    outcome = pooled_project.outcome
    assert outcome.refitted and not outcome.failures
    assert outcome.pool.recording_ids == tuple(sorted(RECORDINGS))
    means = set()
    for recording_id in RECORDINGS:
        pipeline = pooled_project.pipeline(recording_id)
        assert set(pipeline.statuses().values()) == {PL.STATUS_OK}
        manifest = pipeline.manifest
        assert manifest["hmm_pool"] == sorted(RECORDINGS)
        assert manifest["hmm_pool_fingerprint"] == outcome.pool.fingerprint
        assert manifest["stages"]["06"]["diagnostics"]["pooled_over_recordings"] == 4
        means.add(json.dumps(manifest["hmm_state_mean_phi"], sort_keys=True))
        # A recording's stage 06 holds only its own steps.
        assert len(pipeline.table("06", "steps")) == int(pipeline.table("05", "steps")[S.PHI].notna().sum())
    assert len(means) == 1                                           # one state definition for all


def test_pooled_fit_is_reused_until_the_pool_or_its_inputs_change(pooled_project, tmp_path):
    again = A.run_pooled_analysis(pooled_project, RECORDINGS, PARAMS)
    assert not again.refitted and again.pool == pooled_project.outcome.pool

    project = make_project(tmp_path)
    first = A.run_pooled_analysis(project, RECORDINGS, PARAMS)
    fewer = A.run_pooled_analysis(project, RECORDINGS[:3], PARAMS)   # different membership
    assert fewer.refitted and fewer.pool.fingerprint != first.pool.fingerprint
    assert project.pipeline(RECORDINGS[0]).manifest["hmm_pool"] == sorted(RECORDINGS[:3])
    assert project.pipeline(RECORDINGS[3]).manifest["hmm_pool"] == sorted(RECORDINGS)   # left as it was

    changed = PARAMS.__class__(stage_02=PARAMS.stage_02, stage_05=P.RdpParams(epsilon_factor=0.4))
    assert A.run_pooled_analysis(project, RECORDINGS[:3], changed).refitted           # inputs changed


def test_a_pooled_recording_cannot_be_refitted_alone_until_it_leaves_the_pool(tmp_path):
    project = make_project(tmp_path)
    A.run_pooled_analysis(project, RECORDINGS[:2], PARAMS)
    pipeline = project.pipeline(RECORDINGS[0])
    assert pipeline.pool is not None and pipeline.status("06") == PL.STATUS_OK
    with pytest.raises(PL.PipelineError, match="pooled over 2 recordings"):
        pipeline.run(["06"], force=["06"])

    pipeline = project.pipeline(RECORDINGS[0])
    pipeline.reset("06")
    assert pipeline.pool is None and pipeline.manifest["hmm_pool_fingerprint"] is None
    pipeline.run(["06"])
    assert pipeline.manifest["hmm_pool"] == [RECORDINGS[0]]
    assert "pooled_over_recordings" not in pipeline.manifest["stages"]["06"]["diagnostics"]


def test_pooled_fit_keeps_recordings_apart_when_track_ids_repeat(tmp_path):
    """Segmentation off: both recordings contain trajectory_0_0 and trajectory_0_1 (D-015)."""
    params = P.PipelineParams(stage_02=P.SegmentationParams(enabled=False, min_segment_seconds=20.0))
    project = make_project(tmp_path, params)
    A.run_pooled_analysis(project, RECORDINGS[:2], params)
    pipelines = [project.pipeline(recording_id) for recording_id in RECORDINGS[:2]]
    tracks = [set(pipeline.table("06", "steps")[S.TRACK_ID]) for pipeline in pipelines]
    assert tracks[0] == tracks[1] == {"trajectory_0_0", "trajectory_0_1"}
    assert pipelines[0].manifest["stages"]["06"]["diagnostics"]["sequences"] == 4     # 2 recordings x 2 tracks


def test_hmm_refuses_a_table_whose_rows_are_not_grouped_by_sequence():
    """States are written back in row order; an unsorted table would get them on the wrong steps."""
    rng = np.random.default_rng(0)
    steps = pd.DataFrame({
        S.TRACK_ID: ["b"] * 30 + ["a"] * 30,
        S.T_START_S: np.arange(60.0), S.T_END_S: np.arange(60.0) + 1,
        S.X0_MM: 0.0, S.Y0_MM: 0.0, S.X1_MM: 1.0, S.Y1_MM: 1.0, S.DX_MM: 1.0, S.DY_MM: 1.0,
        S.THETA: 0.1, S.PHI: rng.normal(size=60),
    })
    with pytest.raises(ValueError, match="must be sorted by track_id"):
        stages.stage_06(steps, P.HmmParams())
    stages.stage_06(steps.sort_values(S.TRACK_ID, kind="mergesort"), P.HmmParams())


def test_per_condition_fits_are_reported_beside_the_pooled_fit(pooled_project):
    steps = {r: pooled_project.pipeline(r).table("05", "steps") for r in RECORDINGS}
    conditions = A.condition_labels(pooled_project.registry, RECORDINGS, ["genotype"])
    table = A.per_condition_hmm(steps, conditions, P.HmmParams())
    assert list(dict.fromkeys(table[S.FIT])) == [A.POOLED_FIT, "amiGA", "w1118"]
    assert len(table) == 9 and set(table[S.STATE]) == {0, 1, 2}
    pooled = table[table[S.FIT] == A.POOLED_FIT]
    assert pooled[S.N_RECORDINGS].iloc[0] == 4
    recorded = pooled_project.outcome.state_mean_phi
    assert pooled.set_index(S.STATE)[S.MEAN_PHI].to_dict() == pytest.approx({int(k): v for k, v in recorded.items()})


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_n_is_reported_at_segment_larva_and_recording_level(pooled_project):
    comparison = A.compare_recordings(pooled_project, RECORDINGS, ["genotype"])
    assert comparison.unit == A.UNIT_LARVA and comparison.conditions == ("amiGA", "w1118")
    counts = comparison.counts.set_index(S.CONDITION)
    for genotype in ("amiGA", "w1118"):
        row = counts.loc[genotype]
        # Two recordings, two larvae each; larva 1 is lost once, so it has two track segments.
        assert (row[S.N_RECORDINGS], row[S.N_LARVAE], row[S.N_TRACK_SEGMENTS], row[S.N_UNITS]) == (2, 4, 6, 4)
    assert not comparison.warnings
    assert comparison.hmm_pool == tuple(sorted(RECORDINGS))

    units = {unit: A.compare_recordings(pooled_project, RECORDINGS, ["genotype"], unit) for unit in A.UNITS}
    rate = {unit: (c.values[S.MEASURE] == A.MEASURE_HEAD_CAST_RATE).sum() for unit, c in units.items()}
    assert rate == {A.UNIT_SEGMENT: 12, A.UNIT_LARVA: 8, A.UNIT_RECORDING: 4}

    occupancy = comparison.values[comparison.values[S.MEASURE] == A.MEASURE_OCCUPANCY]
    totals = occupancy.groupby(S.UNIT_ID)[S.VALUE].sum()
    assert np.allclose(totals, 1.0) and set(occupancy[S.STATE]) == {0, 1, 2}
    plain = comparison.values[comparison.values[S.MEASURE] == A.MEASURE_CRAWL_LENGTH]
    assert plain[S.STATE].isna().all() and (plain[S.VALUE] > 0).all()


def test_grouping_by_any_metadata_and_refusing_blank_labels(pooled_project):
    by_two = A.compare_recordings(pooled_project, RECORDINGS, ["genotype", "n"])
    assert by_two.conditions == ("amiGA | 1", "amiGA | 2", "w1118 | 1", "w1118 | 2")
    assert any("Only one recording" in warning for warning in by_two.warnings)
    ungrouped = A.compare_recordings(pooled_project, RECORDINGS, [])
    assert ungrouped.conditions == (A.ALL_RECORDINGS,) and ungrouped.counts[S.N_RECORDINGS].iloc[0] == 4

    registry = pooled_project.registry.copy()
    registry.loc[registry[R.RECORDING_ID] == RECORDINGS[1], "genotype"] = ""
    with pytest.raises(A.AnalysisError, match=RECORDINGS[1]):
        A.condition_labels(registry, RECORDINGS, ["genotype"])
    with pytest.raises(A.AnalysisError, match="Not a metadata column"):
        A.condition_labels(registry, RECORDINGS, ["age"])
    with pytest.raises(A.AnalysisError, match="Unknown unit"):
        A.compare_recordings(pooled_project, RECORDINGS, ["genotype"], "animal")


def test_comparing_states_without_a_pooled_fit_is_flagged(tmp_path):
    project = make_project(tmp_path)
    for recording_id in RECORDINGS[:2]:
        project.pipeline(recording_id, PARAMS).run()                 # each fitted on its own
    comparison = A.compare_recordings(project, RECORDINGS[:2], [])
    assert any("one fit pooled over exactly these recordings" in warning for warning in comparison.warnings)
    with pytest.raises(A.AnalysisError, match="not fully processed"):
        A.compare_recordings(project, RECORDINGS[:3], [])


def test_values_pool_raw_quantities_within_a_unit_not_averages():
    """A larva with a long and a short track segment: hand-computed."""
    def steps(track, rows):
        return pd.DataFrame([{S.TRACK_ID: track, S.T_START_S: t, S.DURATION_S: d, S.STATE_FILTERED: s} for t, d, s in rows])

    recording = A.RecordingData(
        recording_id="rec",
        tracks=pd.DataFrame({S.TRACK_ID: ["a", "b", "c"], S.LARVA_INDEX: [0, 0, 1], S.TRACKED_S: [540.0, 60.0, 120.0]}),
        steps=pd.concat([
            steps("a", [(0, 90, 0), (90, 10, 2), (100, 50, 0), (150, 30, 0)]),    # state 0: 90, then 80 after a break
            steps("b", [(0, 10, 2), (10, 10, 2)]),
            steps("c", [(0, 40, 1)]),
        ], ignore_index=True),
        events=pd.DataFrame({S.TRACK_ID: ["a"] * 9 + ["b"] + ["a"], S.EVENT_TYPE: [S.EVENT_HEAD_CAST] * 10 + [S.EVENT_CRAWL],
                             S.START_S: 0.0, S.END_S: 1.0}),
        event_steps=pd.DataFrame({S.TRACK_ID: ["a", "a", "b", "c"], S.EVENT_TYPE: [S.EVENT_CRAWL] * 3 + [S.EVENT_HEAD_CAST],
                                  S.STEP_LENGTH: [1.0, 2.0, 6.0, 99.0]}),
        segment_fits=pd.DataFrame({S.TRACK_ID: ["a", "b"], S.STATE: [0, 0], S.N_POINTS_TOTAL: [30, 10],
                                   S.RHO: [0.2, 0.6], S.KAPPA: [1.0, 1.0], S.MU: [0.0, 0.0], S.SIGMA: [0.1, 0.1]}),
    )
    comparison = A.compare([recording], {"rec": "x"}, A.UNIT_LARVA)
    values = comparison.values.set_index([S.UNIT_ID, S.MEASURE, S.STATE], drop=False)[S.VALUE]

    def value(larva, measure, state=np.nan):
        part = comparison.values[(comparison.values[S.UNIT_ID] == f"rec::larva{larva}") & (comparison.values[S.MEASURE] == measure)]
        part = part[part[S.STATE].isna()] if np.isnan(state) else part[part[S.STATE] == state]
        return part[S.VALUE].iloc[0]

    assert value(0, A.MEASURE_OCCUPANCY, 0) == pytest.approx(170 / 200)        # not the mean of 170/180 and 0/20
    assert value(0, A.MEASURE_OCCUPANCY, 2) == pytest.approx(30 / 200)
    assert value(1, A.MEASURE_OCCUPANCY, 1) == 1.0 and value(1, A.MEASURE_OCCUPANCY, 0) == 0.0
    assert value(0, A.MEASURE_DWELL, 0) == pytest.approx((90 + 80) / 2)        # two stays in state 0
    assert value(0, A.MEASURE_DWELL, 2) == pytest.approx((10 + 20) / 2)        # one in each track segment
    assert value(0, S.RHO, 0) == pytest.approx((0.2 * 30 + 0.6 * 10) / 40)     # weighted by points fitted
    assert value(0, A.MEASURE_HEAD_CAST_RATE) == pytest.approx(10 / 10.0)      # 10 casts in 600 s tracked
    assert value(1, A.MEASURE_HEAD_CAST_RATE) == 0.0                           # tracked, no casts: a real zero
    assert value(0, A.MEASURE_CRAWL_LENGTH) == pytest.approx(3.0)
    assert len(values) > 0

    by_segment = A.compare([recording], {"rec": "x"}, A.UNIT_SEGMENT)
    assert by_segment.counts[S.N_UNITS].iloc[0] == 3 and comparison.counts[S.N_UNITS].iloc[0] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Figures and export
# ──────────────────────────────────────────────────────────────────────────────

def test_comparison_set_is_drawn_and_exported_in_one_action(pooled_project, tmp_path):
    comparison = A.compare_recordings(pooled_project, RECORDINGS, ["genotype"])
    steps = {r: pooled_project.pipeline(r).table("05", "steps") for r in RECORDINGS}
    conditions = A.condition_labels(pooled_project.registry, RECORDINGS, ["genotype"])
    diagnostic = A.per_condition_hmm(steps, conditions, P.HmmParams())
    figures = C.comparison_figures(comparison, diagnostic)
    assert list(figures) == ["state_occupancy", "dwell_times", "steering_parameters", "head_cast_rate",
                             "crawl_length", "hmm_per_condition"]
    for figure in figures.values():
        FigureCanvasAgg(figure)
        figure.canvas.draw()
    # n at all three levels is on the axis of every comparison figure.
    tick = figures["state_occupancy"].axes[0].get_xticklabels()[0].get_text()
    assert tick == "amiGA\n2 recordings\n4 larvae\n6 track segments"
    assert "Each point is one larva" in figures["head_cast_rate"].get_supxlabel()

    folder = A.new_comparison_folder(tmp_path)
    manifest_path = A.export_comparison(folder, comparison, figures, C.FIGURE_TITLES)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["unit"] == "larva" and manifest["grouped_by"] == ["genotype"]
    assert manifest["recordings"] == RECORDINGS and manifest["hmm_pool"] == sorted(RECORDINGS)
    assert manifest["n"][0] == {"condition": "amiGA", "n_recordings": 2, "n_larvae": 4, "n_track_segments": 6, "n_units": 4}
    for name, entry in manifest["figures"].items():
        assert entry["files"] == [f"{name}.pdf", f"{name}.svg", f"{name}.png"]
        assert all((folder / file).stat().st_size > 500 for file in entry["files"])
    exported = pd.read_csv(folder / "values.csv")
    assert set(exported[S.CONDITION]) == {"amiGA", "w1118"} and len(exported) == len(comparison.values)


def test_excel_export_holds_the_numbers_behind_every_figure(pooled_project, tmp_path):
    comparison = A.compare_recordings(pooled_project, RECORDINGS, ["genotype"])
    path = A.export_excel(tmp_path / "out" / "data.xlsx", comparison)
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == ["about", "n", "all_values_long", "state_occupancy", "dwell_time_s",
                            "steering_parameters", "head_cast_rate_per_min", "crawl_length_mm"]
    id_columns = ["condition", "genotype", "recording_id", "unit_id"]

    long = sheets["all_values_long"]
    assert list(long.columns) == id_columns + ["measure", "state", "value"]
    assert len(long) == len(comparison.values)
    assert (long["condition"] == long["genotype"]).all()               # grouped by genotype alone
    # Same numbers as the figures are drawn from, whatever the row order.
    key = ["recording_id", "unit_id", "measure", "state"]
    merged = long.merge(comparison.values, on=key, suffixes=("_excel", "_figure"))
    assert len(merged) == len(long) and np.allclose(merged["value_excel"], merged["value_figure"])

    occupancy = sheets["state_occupancy"]
    assert list(occupancy.columns) == id_columns + ["state_0", "state_1", "state_2"]
    assert len(occupancy) == 8 and occupancy["unit_id"].is_unique       # one row per larva
    assert np.allclose(occupancy[["state_0", "state_1", "state_2"]].sum(axis=1), 1.0)
    assert set(occupancy["genotype"]) == {"amiGA", "w1118"} and occupancy["recording_id"].nunique() == 4

    rate = sheets["head_cast_rate_per_min"]
    assert list(rate.columns) == id_columns + ["head_cast_rate_per_min"] and len(rate) == 8
    steering = sheets["steering_parameters"]
    assert list(steering.columns) == id_columns + [f"{p}_state_{k}" for p in ("rho", "kappa", "mu", "sigma") for k in range(3)]
    assert steering["unit_id"].is_unique
    assert sheets["n"].to_dict(orient="records")[0] == {
        "condition": "amiGA", "n_recordings": 2, "n_larvae": 4, "n_track_segments": 6, "n_units": 4}
    about = dict(zip(sheets["about"]["item"], sheets["about"]["value"]))
    assert about["unit (one row per)"] == "larva" and about["grouped by"] == "genotype"
    assert about["statistics"] == "None computed here." and "measure: state_occupancy" in about

    by_recording = A.compare_recordings(pooled_project, RECORDINGS, ["genotype", "n"], A.UNIT_RECORDING)
    sheets = pd.read_excel(A.export_excel(tmp_path / "by_recording.xlsx", by_recording), sheet_name=None)
    crawl = sheets["crawl_length_mm"]
    assert list(crawl.columns)[:5] == ["condition", "genotype", "n", "recording_id", "unit_id"] and len(crawl) == 4
    assert crawl.loc[0, "condition"] == "amiGA | 1" and (crawl["unit_id"] == crawl["recording_id"]).all()


def test_aggregate_tab_fits_compares_and_exports(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from larval_explorer.ui.main_window import MainWindow
    from tests.test_gui import idle

    project = make_project(tmp_path)
    app = QApplication.instance() or QApplication([])
    problems = []
    monkeypatch.setattr(MainWindow, "notify", lambda self, title, text: problems.append((title, text)))
    window = MainWindow()
    window.open_project(project.folder)
    window.params = PARAMS
    tab = window.aggregate_tab
    window.tabs.setCurrentWidget(tab)
    assert tab.ticked() == RECORDINGS and tab.group_keys() == ["genotype"]
    assert tab.unit_combo.currentText() == "larva"

    tab.compare_button.click()                                        # nothing processed yet
    assert "not fully processed" in tab.warning_label.text() and not tab.export_button.isEnabled()

    tab.diagnostic_box.setChecked(True)
    tab.fit_button.click()
    assert window.busy
    idle(app, window)
    assert not problems, problems
    assert "One HMM fitted over 4 recordings" in tab.status_label.text()
    names = [tab.figures.selector.itemText(i) for i in range(tab.figures.selector.count())]
    assert names == ["state_occupancy", "dwell_times", "steering_parameters", "head_cast_rate", "crawl_length",
                     "hmm_per_condition"]
    assert tab.counts.view.model().rowCount() == 2 and tab.warning_label.text() == ""
    for name in names:
        tab.figures.selector.setCurrentText(name)
        assert tab.figures.figure.axes

    tab.unit_combo.setCurrentText("recording")
    tab.compare_button.click()
    assert tab.comparison.unit == "recording" and tab.comparison.counts[S.N_UNITS].tolist() == [2, 2]

    tab.table.item(3, 0).setCheckState(Qt.Unchecked)                  # a different selection than was pooled
    tab.compare_button.click()
    assert "one fit pooled over exactly these recordings" in tab.warning_label.text()

    tab.export_to(tmp_path / "figure_set")
    idle(app, window)
    manifest = json.loads((tmp_path / "figure_set" / A.FIGURE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["unit"] == "recording" and len(manifest["recordings"]) == 3 and manifest["warnings"]
    assert (tmp_path / "figure_set" / "state_occupancy.pdf").is_file()
    assert (tmp_path / "figure_set" / A.EXCEL_FILE).is_file() and A.EXCEL_FILE in manifest["data_files"]

    assert tab.excel_button.isEnabled()
    assert tab.export_excel_to(tmp_path / "stats" / "my_data.xlsx")
    assert "Data written to" in tab.status_label.text()
    workbook = pd.read_excel(tmp_path / "stats" / "my_data.xlsx", sheet_name=None)
    assert len(workbook["head_cast_rate_per_min"]) == 3               # three recordings ticked, unit = recording

    # The recordings now carry the pooled states, visible on the HMM tab.
    window.set_active_recording(RECORDINGS[0])
    window.tabs.setCurrentWidget(window.stage_tabs["06"])
    idle(app, window)
    assert window.statuses["06"] == PL.STATUS_OK
    assert "one HMM fitted over 4 recordings" in window.stage_tabs["06"].messages.toPlainText()
    window.close()


def test_comparison_figures_cope_with_missing_measures():
    empty = A.Comparison(
        values=pd.DataFrame(columns=[S.CONDITION, S.RECORDING_ID, S.UNIT_ID, S.MEASURE, S.STATE, S.VALUE]),
        counts=pd.DataFrame({S.CONDITION: ["x"], S.N_RECORDINGS: [1], S.N_LARVAE: [0], S.N_TRACK_SEGMENTS: [0], S.N_UNITS: [0]}),
        unit=A.UNIT_LARVA, group_keys=(), conditions=("x",), recording_ids=("r",), hmm_pool=(), warnings=(),
    )
    for figure in C.comparison_figures(empty).values():
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        assert figure.axes[0].texts
