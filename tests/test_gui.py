"""The window, driven without a display.

Phase 3 acceptance: a full run by clicking only, outputs equal to the headless
pipeline, and stage state restored when the project is reopened.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pandas as pd
import pytest
from PyQt5.QtWidgets import QApplication

from larval_explorer.core import params as P
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.core import schema as S
from larval_explorer.ui.main_window import MainWindow
from larval_explorer.ui.widgets import ParamsForm
from tests.test_pipeline import write_synthetic_recording


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, condition, timeout=120.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def idle(app, window):
    wait_until(app, lambda: not window.busy and not window._background)


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    """A window with a project holding one synthetic recording, selected."""
    problems = []
    monkeypatch.setattr(MainWindow, "notify", lambda self, title, text: problems.append((title, text)))
    window = MainWindow()
    window.problems = problems
    window.create_project(tmp_path / "project")
    source = write_synthetic_recording(tmp_path / "w1118_n1_att1.csv")
    window.project_tab.add_paths([source])
    window.project_tab.table.selectRow(0)
    idle(app, window)
    yield window
    window.close()


def set_min_segment(window, seconds):
    """The synthetic recording is short; keep its segments."""
    window.tabs.setCurrentWidget(window.segmentation_tab)
    window.segmentation_tab.minimum_spin.setValue(seconds)


def test_params_form_round_trips_every_parameter_set(app):
    for field_name in (f"stage_{key}" for key in PL.STAGE_KEYS):
        params = getattr(P.PipelineParams(), field_name)
        form = ParamsForm(params)
        assert form.value() == params, field_name

    form = ParamsForm(P.SteeringParams())
    seen = []
    form.changed.connect(seen.append)
    form.findChild(type(form._editors["max_lag"]), "max_lag").setValue(6)
    mu_max = form._editors["mu_max"]
    mu_max.setText("")
    mu_max.editingFinished.emit()
    assert seen[-1].max_lag == 6 and seen[-1].mu_max is None
    assert seen[-1].clip_pcts == (1, 99) and P.params_to_dict(seen[-1])["clip_pcts"] == [1, 99]

    hmm = ParamsForm(P.HmmParams())
    seen.clear()
    hmm.changed.connect(seen.append)
    hmm._editors["n_states"].setValue(4)          # upstream fixes this at three
    assert not seen and "n_states must be 3" in hmm._error.text()


def test_project_tab_registry_and_filename_parsing(app, window):
    tab = window.project_tab
    assert window.pipeline is not None and window.pipeline.recording_id == "w1118_n1_att1"
    assert tab.table.rowCount() == 1

    tab.pattern_edit.setText(r"(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)")
    tab.preview_parse()
    assert "genotype" not in window.project.registry.columns       # a preview changes nothing
    assert tab.apply_button.isEnabled()
    tab.apply_parse()
    assert window.project.registry.loc[0, "genotype"] == "w1118"
    assert R.load_registry(window.project.folder / R.REGISTRY_NAME).loc[0, "attempt"] == "1"
    assert window.pipeline.manifest["metadata"]["genotype"] == "w1118"

    tab.add_paths([window.pipeline.source_file])                    # a duplicate is refused, with a message
    assert window.problems and "Duplicate" in window.problems[-1][1]
    assert tab.table.rowCount() == 1


def test_full_run_by_clicking_matches_the_headless_pipeline(app, window, tmp_path):
    set_min_segment(window, 20.0)
    assert window.params.stage_02.min_segment_seconds == 20.0
    assert window.run_all_button.isEnabled()
    window.run_all_button.click()
    assert window.busy and not window.run_all_button.isEnabled()
    idle(app, window)
    assert not window.problems, window.problems
    assert set(window.statuses.values()) == {PL.STATUS_OK}
    assert all(item.startswith("✓") for item in (window.stage_list.item(i).text() for i in range(window.stage_list.count())))

    headless = PL.RecordingPipeline(
        "w1118_n1_att1", window.pipeline.source_file, tmp_path / "headless",
        P.PipelineParams(stage_02=P.SegmentationParams(min_segment_seconds=20.0)),
    )
    headless.run()
    for stage in PL.STAGE_KEYS:
        for name in headless.manifest["stages"][stage]["outputs"]:
            ours = (window.pipeline.stage_folder(stage) / name).read_bytes()
            assert ours == (headless.stage_folder(stage) / name).read_bytes(), f"{stage}/{name}"

    # Every tab shows its result without error.
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        idle(app, window)
    assert not window.problems, window.problems
    tab = window.stage_tabs["06"]
    window.tabs.setCurrentWidget(tab)
    assert "phi_distribution_by_state" in [tab.figures.selector.itemText(i) for i in range(tab.figures.selector.count())]
    assert tab.table_preview.view.model().rowCount() > 0
    assert "converged: True" in tab.messages.toPlainText()


def test_reopening_restores_state_and_editing_marks_stages_stale(app, window, tmp_path):
    set_min_segment(window, 20.0)
    window.run_targets(["05"])
    idle(app, window)
    folder = window.project.folder
    window.close()

    reopened = MainWindow()
    reopened.problems = window.problems
    reopened.open_project(folder)
    reopened.project_tab.table.selectRow(0)
    idle(app, reopened)
    assert reopened.params.stage_02.min_segment_seconds == 20.0     # parameters come back with the results
    statuses = reopened.statuses
    assert [statuses[key] for key in ("00", "01", "02", "03", "04", "05")] == [PL.STATUS_OK] * 6
    assert statuses["06"] == PL.STATUS_MISSING

    tab = reopened.stage_tabs["03"]
    reopened.tabs.setCurrentWidget(tab)
    tab.forms["03"]._editors["window_size"].setValue(31)
    assert reopened.statuses["02"] == PL.STATUS_OK
    assert [reopened.statuses[key] for key in ("03", "04", "05")] == [PL.STATUS_STALE] * 3
    assert reopened.stage_list.item(3).text().startswith("⚠")
    assert "stale" in tab.status_label.text()

    tab.run_button.click()
    idle(app, reopened)
    assert reopened.statuses["03"] == PL.STATUS_OK and reopened.statuses["05"] == PL.STATUS_STALE
    assert reopened.pipeline.manifest["stages"]["03"]["params"]["window_size"] == 31
    reopened.close()


def test_segmentation_tab_previews_live_and_opens_the_explorer(app, window):
    window.run_targets(["01"])
    idle(app, window)
    tab = window.segmentation_tab
    window.tabs.setCurrentWidget(tab)
    idle(app, window)
    assert "1 kept" in tab.summary_label.text()                      # at 60 s only the unbroken larva survives

    tab.minimum_spin.setValue(20.0)
    wait_until(app, lambda: "3 kept" in tab.summary_label.text())
    assert tab.summary.view.model().rowCount() == 2                  # one row per larva
    assert len(tab.figures.figure.axes) == 3                         # timeline + two larvae
    assert window.statuses["02"] == PL.STATUS_MISSING                # a preview records nothing

    tab.bridge_spin.setValue(150)                                    # bridges the 100-frame gap of larva 1
    wait_until(app, lambda: "2 segments found" in tab.summary_label.text())

    tab.apply_button.click()
    idle(app, window)
    assert window.statuses["02"] == PL.STATUS_OK
    assert window.pipeline.manifest["stages"]["02"]["params"]["bridge_max_frames"] == 150

    window.run_targets(["03"])
    idle(app, window)
    window.open_in_explorer("w1118_n1_att1__larva1__seg0")
    idle(app, window)
    assert window.tabs.currentWidget() is window.explorer_tab
    assert window.explorer_tab.track_selector.currentText() == "w1118_n1_att1__larva1__seg0"


def test_explorer_layers_and_time_scrubber(app, window):
    set_min_segment(window, 20.0)
    window.run_targets(None)
    idle(app, window)
    tab = window.explorer_tab
    window.tabs.setCurrentWidget(tab)
    idle(app, window)
    assert tab.track_selector.count() == 3
    assert all(box.isEnabled() for box in tab.layer_boxes.values())
    for box in tab.layer_boxes.values():
        box.setChecked(True)
    legend_labels = [text.get_text() for text in tab.figures.figure.legends[0].get_texts()]
    assert any(label.startswith("RDP steps") for label in legend_labels)
    assert any(label.startswith("Head casts") for label in legend_labels)

    axes = tab.figures.figure.axes[0]
    axes.set_xlim(900, 950)
    tab.layer_boxes["rdp"].setChecked(False)                         # toggling a layer keeps the zoom
    assert tab.figures.figure.axes[0].get_xlim() == (900.0, 950.0)

    tab.time_box.setChecked(True)
    tab.time_slider.setValue(500)
    marker = tab.figures.figure.time_marker
    assert marker.get_visible() and tab.time_label.text().startswith("t = ")
    path = tab.data.path(tab.track_selector.currentText())
    assert path[S.MOM_X].min() <= marker.get_xdata()[0] <= path[S.MOM_X].max()


def test_a_failing_stage_is_reported_and_the_window_recovers(app, window):
    write_synthetic_recording(window.pipeline.source_file, drop_channel="tail_y")
    window.pipeline._source_hash = None
    window.run_targets(["01"])
    idle(app, window)
    assert window.statuses["00"] == PL.STATUS_FAILED
    assert window.problems and "tail_y" in window.problems[-1][1]
    assert window.stage_list.item(0).text().startswith("✗")
    assert window.run_all_button.isEnabled()                         # not stuck busy

    write_synthetic_recording(window.pipeline.source_file)
    window.pipeline._source_hash = None
    window.run_targets(["01"])
    idle(app, window)
    assert window.statuses["01"] == PL.STATUS_OK
