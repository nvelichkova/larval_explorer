"""Phase 4: many recordings, unattended, with one of them broken on purpose."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from larval_explorer.core import batch as B
from larval_explorer.core import manifest as M
from larval_explorer.core import params as P
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.core.project import Project
from tests.test_pipeline import write_synthetic_recording

PARAMS = P.PipelineParams(stage_02=P.SegmentationParams(min_segment_seconds=20.0))
PATTERN = r"(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)"
GOOD = ["w1118_n1_att1", "w1118_n2_att1", "amiGA_n1_att1"]
BROKEN = "amiGA_n2_att1"


@pytest.fixture
def project(tmp_path):
    """Four recordings of two genotypes; one is missing a required channel."""
    data = tmp_path / "data"
    data.mkdir()
    for name in GOOD:
        write_synthetic_recording(data / f"{name}.csv")
    write_synthetic_recording(data / f"{BROKEN}.csv", drop_channel="tail_y")
    project = Project.create(tmp_path / "project")
    project.registry = R.add_recordings(project.registry, [data])
    project.registry = R.apply_parse(project.registry, R.preview_parse(project.registry, PATTERN))
    project.save()
    return project


def by_id(outcomes):
    return {outcome.recording_id: outcome for outcome in outcomes}


def test_batch_runs_every_recording_and_isolates_the_failure(project):
    events = []
    outcomes, log_path = B.run_batch(
        project, options=B.BatchOptions(params=PARAMS), max_workers=2,
        progress=lambda recording, stage, event: events.append((recording, stage, event)),
    )
    results = by_id(outcomes)
    assert [outcome.recording_id for outcome in outcomes] == project.recording_ids()   # order as asked
    assert {name: results[name].result for name in GOOD} == {name: B.RESULT_OK for name in GOOD}
    failed = results[BROKEN]
    assert failed.result == B.RESULT_FAILED and failed.failed_stage == "00"
    assert "tail_y" in failed.message and "SchemaError" in failed.traceback

    # Complete, independent output trees with parameters and metadata in every manifest.
    for name in GOOD:
        manifest = M.load_manifest(project.output_root / name)
        assert set(manifest["stages"]) == set(PL.STAGE_KEYS)
        assert all(record["status"] == "ok" for record in manifest["stages"].values())
        assert manifest["stages"]["02"]["params"]["min_segment_seconds"] == 20.0
        assert manifest["metadata"]["genotype"] == name.split("_")[0]
        assert manifest["hmm_pool"] == [name]                         # nothing pooled across recordings
        assert results[name].stages_run == list(PL.STAGE_KEYS)
    assert M.load_manifest(project.output_root / BROKEN)["stages"]["00"]["status"] == "failed"
    assert {recording for recording, _, _ in events} == set(GOOD) | {BROKEN}

    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log_path.parent == project.output_root / B.LOG_FOLDER
    assert log["summary"] == {"failed": 1, "ok": 3} and log["finished_at"]
    assert log["options"]["params"]["02"]["min_segment_seconds"] == 20.0
    logged = {entry["recording_id"]: entry for entry in log["recordings"]}
    assert logged[BROKEN]["failed_stage"] == "00" and "SchemaError" in logged[BROKEN]["traceback"]


def test_second_batch_skips_what_is_done_and_force_reruns(project):
    options = B.BatchOptions(params=PARAMS, through_stage="05")
    first, _ = B.run_batch(project, GOOD, options)
    assert by_id(first)[GOOD[0]].stages_run == ["00", "01", "02", "03", "04", "05"]
    assert M.load_manifest(project.output_root / GOOD[0])["stages"].keys() == {"00", "01", "02", "03", "04", "05"}

    again, _ = B.run_batch(project, GOOD, options)
    assert {outcome.result for outcome in again} == {B.RESULT_UP_TO_DATE}
    assert all(not outcome.stages_run and len(outcome.stages_skipped) == 6 for outcome in again)

    own, _ = B.run_batch(project, GOOD, B.BatchOptions(through_stage="05"))        # each recording's saved parameters
    assert {outcome.result for outcome in own} == {B.RESULT_UP_TO_DATE}

    forced, _ = B.run_batch(project, GOOD[:1], B.BatchOptions(params=PARAMS, through_stage="03", force=True))
    assert forced[0].result == B.RESULT_OK and forced[0].stages_run == ["00", "01", "02", "03"]

    further, _ = B.run_batch(project, GOOD[:1], B.BatchOptions(params=PARAMS, through_stage="08"))
    # Re-running a stage on unchanged inputs gives the same result, so what follows it stays current.
    assert further[0].stages_skipped == ["00", "01", "02", "03", "04", "05"]
    assert further[0].stages_run == ["06", "07", "08"]


def test_cancel_leaves_consistent_trees(project):
    finished = []

    def progress(recording, stage, event):
        if event == "finished":
            finished.append((recording, stage))

    outcomes, log_path = B.run_batch(
        project, GOOD, B.BatchOptions(params=PARAMS), max_workers=1,
        progress=progress, should_cancel=lambda: len(finished) >= 3,
    )
    assert [outcome.result for outcome in outcomes] == [B.RESULT_CANCELLED] * 3
    assert outcomes[0].stages_run == ["00", "01", "02"] and outcomes[1].stages_run == []
    pipeline = project.pipeline(GOOD[0], PARAMS)
    assert [pipeline.status(key) for key in ("00", "01", "02", "03")] == ["ok", "ok", "ok", "missing"]
    assert json.loads(log_path.read_text(encoding="utf-8"))["summary"] == {"cancelled": 3}

    resumed, _ = B.run_batch(project, GOOD[:1], B.BatchOptions(params=PARAMS, through_stage="04"))
    assert resumed[0].stages_skipped == ["00", "01", "02"] and resumed[0].stages_run == ["03", "04"]


def test_figures_are_off_by_default_and_written_when_asked(project):
    plain, _ = B.run_batch(project, GOOD[:2], B.BatchOptions(params=PARAMS, through_stage="04"), max_workers=2)
    assert all(outcome.figures_written == 0 for outcome in plain)
    assert not (project.output_root / GOOD[0] / PL.FIGURES_FOLDER).exists()

    with_figures, _ = B.run_batch(
        project, GOOD[:2], B.BatchOptions(params=PARAMS, through_stage="04", export_figures=True), max_workers=2
    )
    assert all(outcome.result == B.RESULT_UP_TO_DATE and outcome.figures_written >= 4 for outcome in with_figures)
    figures = project.output_root / GOOD[1] / PL.FIGURES_FOLDER
    assert (figures / "02_segments" / "segmentation_overview.png").is_file()
    assert (figures / "04_body_length" / "body_length.png").is_file()


def test_unknown_recording_or_stage_is_reported_not_raised(project):
    outcomes, _ = B.run_batch(project, ["no_such_recording", GOOD[0]], B.BatchOptions(params=PARAMS, through_stage="00"))
    assert outcomes[0].result == B.RESULT_FAILED and "no_such_recording" in outcomes[0].message
    assert outcomes[1].result == B.RESULT_OK
    with pytest.raises(PL.PipelineError):
        B.BatchOptions(through_stage="12")


# ──────────────────────────────────────────────────────────────────────────────
# The Batch tab
# ──────────────────────────────────────────────────────────────────────────────

def test_batch_tab_runs_unattended_and_lets_you_inspect_a_recording(project, monkeypatch):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from larval_explorer.ui.main_window import MainWindow
    from tests.test_gui import idle, wait_until

    app = QApplication.instance() or QApplication([])
    problems = []
    monkeypatch.setattr(MainWindow, "notify", lambda self, title, text: problems.append((title, text)))
    window = MainWindow()
    window.open_project(project.folder)
    window.segmentation_tab.minimum_spin.setValue(20.0)       # no active recording: nothing changes
    window.params = PARAMS
    tab = window.batch_tab
    window.tabs.setCurrentWidget(tab)
    assert tab.table.rowCount() == 4 and tab.ticked() == window.project.recording_ids()
    headers = [tab.table.horizontalHeaderItem(i).text() for i in range(tab.table.columnCount())]
    assert headers == ["Recording", "genotype", "n", "attempt", "Progress", "Result"]

    tab.through_combo.setCurrentIndex(5)                       # through stage 05
    tab.run_button.click()
    assert window.busy and tab.running and not window.run_all_button.isEnabled()
    wait_until(app, lambda: not tab.running)
    idle(app, window)

    assert not problems, problems
    results = {outcome.recording_id: outcome.result for outcome in tab.outcomes}
    assert results == {**{name: B.RESULT_OK for name in GOOD}, BROKEN: B.RESULT_FAILED}
    result_column = tab.table.columnCount() - 1
    shown = {tab.table.item(row, 0).text(): tab.table.item(row, result_column).text() for row in range(4)}
    assert shown[BROKEN] == "✗ failed at stage 00" and shown[GOOD[0]] == "✓ done"
    assert "tail_y" in tab.log_view.toPlainText()
    assert "3 ok" in tab.summary_label.text() and "1 failed" in tab.summary_label.text()
    assert tab.progress.value() == 4
    logs = list((window.project.output_root / B.LOG_FOLDER).glob("batch_*.json"))
    assert len(logs) == 1 and json.loads(logs[0].read_text(encoding="utf-8"))["summary"] == {"failed": 1, "ok": 3}

    # Inspect one: it becomes the active recording with its results already there.
    row = next(row for row in range(4) if tab.table.item(row, 0).text() == GOOD[2])
    tab.table.itemDoubleClicked.emit(tab.table.item(row, 0))
    idle(app, window)
    assert window.pipeline.recording_id == GOOD[2]
    assert [window.statuses[key] for key in ("00", "03", "05", "06")] == ["ok", "ok", "ok", "missing"]
    window.tabs.setCurrentWidget(window.stage_tabs["05"])
    idle(app, window)
    assert window.stage_tabs["05"].figures.selector.count() > 1

    # Untick the broken one and run again: everything is already up to date.
    window.tabs.setCurrentWidget(tab)
    broken_row = next(row for row in range(4) if tab.table.item(row, 0).text() == BROKEN)
    tab.table.item(broken_row, 0).setCheckState(Qt.Unchecked)
    tab.run_button.click()
    wait_until(app, lambda: not tab.running)
    idle(app, window)
    assert {outcome.result for outcome in tab.outcomes} == {B.RESULT_UP_TO_DATE}
    assert window.pipeline is not None and window.pipeline.recording_id == GOOD[2]
    window.close()
