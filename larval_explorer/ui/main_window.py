"""The main window: recording and stage status on the left, tabs on the right."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QObject, Qt, QThreadPool, pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QComboBox,
    QShortcut,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from larval_explorer import __version__
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.core.params import PipelineParams
from larval_explorer.core.project import (
    STATE_COMPLETE,
    STATE_FAILED,
    STATE_NOT_RUN,
    STATE_PARTIAL,
    Project,
    ProjectError,
)
from larval_explorer.plots import catalog
from larval_explorer.ui.aggregate_tab import AggregateTab
from larval_explorer.ui.batch_tab import BatchTab
from larval_explorer.ui.tabs import ExplorerTab, IngestTab, ProjectTab, SegmentationTab, StageTab
from larval_explorer.workers.tasks import FunctionWorker, PipelineWorker

STATUS_SYMBOL = {PL.STATUS_OK: "✓", PL.STATUS_STALE: "⚠", PL.STATUS_MISSING: "○", PL.STATUS_FAILED: "✗"}
RUNNING_SYMBOL = "…"
RECORDING_SYMBOL = {STATE_COMPLETE: "✓", STATE_PARTIAL: "◐", STATE_FAILED: "✗", STATE_NOT_RUN: "○"}

STAGE_TABS = (
    ("03", "Smooth 03", "Savitzky-Golay smoothing of the centre of mass, per track. "
                         "Zoom into a turn to check the window is not eating real movement."),
    ("04", "Body 04", "Mean body length per track. It sets the RDP tolerance of that track."),
    ("05", "RDP 05", "Simplify each trajectory into straight steps. The tolerance is epsilon factor × body length."),
    ("06", "HMM 06", "Fit the three-state Gaussian HMM on phi. Stored state numbers are upstream's."),
    ("07", "Filter 07", "Replace state runs shorter than min run with a neighbouring state."),
    ("08", "HMM-Seg 08", "Keep runs of one filtered state that last at least min steps. "
                          "These are HMM segments, not the track segments of stage 02."),
    ("09", "Events 09", "Detect crawls and head casts from the per-frame trajectory."),
    ("10", "Steps 10", "Build the run-anchor and event-level step tables."),
    ("11", "Steering 11", "Fit the AR(2) steering model inside each HMM segment, by state."),
)


class _Relay(QObject):
    """Delivers a background result to a callback on the GUI thread.

    A QObject owned by the window, so the worker's signal is queued to the GUI
    thread whatever kind of callable the callback is.
    """

    def __init__(self, window: "MainWindow", worker: FunctionWorker, on_done: Callable, on_error: Callable):
        super().__init__(window)
        self._window, self._worker, self._on_done, self._on_error = window, worker, on_done, on_error

    def _finish(self, callback: Callable, value) -> None:
        if self in self._window._background:
            self._window._background.remove(self)
        self.deleteLater()
        callback(value)

    @pyqtSlot(object)
    def done(self, value) -> None:
        self._finish(self._on_done, value)

    @pyqtSlot(str)
    def error(self, message: str) -> None:
        self._finish(self._on_error, message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Larval Explorer {__version__}")
        self.resize(1500, 950)

        self.project: Project | None = None
        self.pipeline: PL.RecordingPipeline | None = None
        self.params = PipelineParams()
        self.statuses: dict[str, str] = {key: PL.STATUS_MISSING for key in PL.STAGE_KEYS}
        self.busy = False
        self.thread_pool = QThreadPool.globalInstance()
        self._worker: PipelineWorker | None = None
        self._background: list[_Relay] = []
        self._running_stage: str | None = None

        # Left: active recording, stage states, run controls.
        self.recording_label = QLabel("No recording selected.")
        self.recording_label.setWordWrap(True)
        # Switch recording from any tab: the tab stays where it is and redraws.
        self.recording_selector = QComboBox()
        self.recording_selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.recording_selector.setToolTip(
            "✓ every stage has a result   ◐ some stages   ✗ a stage failed   ○ not run" + chr(10)
            + "Ctrl+Down / Ctrl+Up: next / previous recording"
        )
        self.recording_selector.activated.connect(self._recording_chosen)
        self.previous_button, self.next_button = QPushButton("◀"), QPushButton("▶")
        self.previous_button.setToolTip("Previous recording (Ctrl+Up)")
        self.next_button.setToolTip("Next recording (Ctrl+Down)")
        for button in (self.previous_button, self.next_button):
            button.setFixedWidth(34)
        self.previous_button.clicked.connect(lambda: self.step_recording(-1))
        self.next_button.clicked.connect(lambda: self.step_recording(+1))
        QShortcut(QKeySequence("Ctrl+Up"), self, activated=lambda: self.step_recording(-1))
        QShortcut(QKeySequence("Ctrl+Down"), self, activated=lambda: self.step_recording(+1))
        self.stage_list = QListWidget()
        self.stage_list.itemClicked.connect(self._stage_clicked)
        self.run_all_button, self.cancel_button = QPushButton("Run all stages"), QPushButton("Cancel")
        self.run_all_button.clicked.connect(lambda: self.run_targets(None))
        self.cancel_button.clicked.connect(self.cancel_run)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Active recording"))
        selector_row = QHBoxLayout()
        selector_row.addWidget(self.recording_selector, 1)
        selector_row.addWidget(self.previous_button)
        selector_row.addWidget(self.next_button)
        left_layout.addLayout(selector_row)
        left_layout.addWidget(self.recording_label)
        left_layout.addWidget(QLabel("Stages   ✓ current   ⚠ stale   ○ not run   ✗ failed"))
        left_layout.addWidget(self.stage_list, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_all_button)
        buttons.addWidget(self.cancel_button)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.progress_label)

        # Right: tabs.
        self.tabs = QTabWidget()
        self.project_tab = ProjectTab(self)
        self.project_tab.registry_changed.connect(self._registry_changed)
        self.ingest_tab = IngestTab(self)
        self.segmentation_tab = SegmentationTab(self)
        self.explorer_tab = ExplorerTab(self)
        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(self.ingest_tab, "Ingest 00/01")
        self.tabs.addTab(self.segmentation_tab, "★ Segmentation 02")
        self.stage_tab_index = {"00": 1, "01": 1, "02": 2}
        self.stage_tabs: dict[str, StageTab] = {}
        for stage, title, intro in STAGE_TABS:
            tab = StageTab(self, (stage,), intro)
            self.stage_tabs[stage] = tab
            self.stage_tab_index[stage] = self.tabs.addTab(tab, title)
        self.explorer_index = self.tabs.addTab(self.explorer_tab, "Trajectory Explorer")
        self.batch_tab = BatchTab(self)
        self.tabs.addTab(self.batch_tab, "Batch")
        self.aggregate_tab = AggregateTab(self)
        self.tabs.addTab(self.aggregate_tab, "Aggregate")
        self.tabs.currentChanged.connect(lambda _index: self._refresh_current_tab())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1250])
        self.setCentralWidget(splitter)
        self.refresh()

    # ── messages ────────────────────────────────────────────────────────────

    def notify(self, title: str, text: str) -> None:
        """Tell the user something went wrong. One place, so tests can intercept it."""
        self.statusBar().showMessage(f"{title}: {text.splitlines()[0] if text else ''}", 15000)
        QMessageBox.warning(self, title, text)

    # ── project ─────────────────────────────────────────────────────────────

    def new_project_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the new project")
        if folder:
            self.create_project(Path(folder))

    def open_project_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open project folder")
        if folder:
            self.open_project(Path(folder))

    def create_project(self, folder: Path) -> None:
        try:
            self._set_project(Project.create(folder))
        except (ProjectError, OSError) as error:
            self.notify("New project", str(error))

    def open_project(self, folder: Path) -> None:
        try:
            self._set_project(Project.open(folder))
        except (ProjectError, R.RegistryError, OSError, ValueError) as error:
            self.notify("Open project", str(error))

    def _set_project(self, project: Project) -> None:
        if self.busy:
            self.notify("Project", "A run is in progress. Cancel it or wait for it to finish.")
            return
        self.project, self.pipeline, self.params = project, None, PipelineParams()
        self.project_tab.pattern_edit.setText(project.filename_pattern)
        self.setWindowTitle(f"Larval Explorer {__version__} — {project.folder}")
        self.refresh()

    def _registry_changed(self) -> None:
        """The registry or output folder was edited: drop the recording if it no longer applies."""
        if self.pipeline is None:
            self._refresh_recording_selector()
            return
        still_there = self.pipeline.recording_id in self.project.recording_ids()
        if still_there and self.pipeline.folder.parent == Path(self.project.output_root):
            self.pipeline.metadata = R.metadata_for(self.project.registry, self.pipeline.recording_id)
            self.pipeline.manifest["metadata"] = self.pipeline.metadata
            self.recording_label.setText(self._recording_caption())
            self._refresh_recording_selector()
        else:
            recording_id = self.pipeline.recording_id if still_there else None
            self.pipeline = None
            if recording_id is not None:
                self.set_active_recording(recording_id)
            self.refresh()

    def _recording_caption(self) -> str:
        if self.pipeline is None:
            return "No recording selected."
        metadata = ", ".join(f"{key}: {value}" for key, value in self.pipeline.metadata.items() if str(value).strip())
        return metadata or self.pipeline.recording_id

    def _refresh_recording_selector(self) -> None:
        """List every recording with how far it has got, and mark the active one."""
        selector = self.recording_selector
        ids = self.project.recording_ids() if self.project is not None else []
        selector.blockSignals(True)
        selector.clear()
        for recording_id in ids:
            symbol = RECORDING_SYMBOL[self.project.recording_state(recording_id)]
            selector.addItem(f"{symbol}  {recording_id}", recording_id)
        active = self.pipeline.recording_id if self.pipeline is not None else None
        selector.setCurrentIndex(ids.index(active) if active in ids else -1)
        selector.blockSignals(False)
        idle = not self.busy
        selector.setEnabled(idle and bool(ids))
        self.previous_button.setEnabled(idle and len(ids) > 1)
        self.next_button.setEnabled(idle and len(ids) > 1)

    def _recording_chosen(self, index: int) -> None:
        recording_id = self.recording_selector.itemData(index)
        if recording_id is not None:
            self.set_active_recording(recording_id)

    def step_recording(self, step: int) -> None:
        """Make the next (+1) or previous (-1) recording active, wrapping round."""
        if self.busy or self.project is None:
            return
        ids = self.project.recording_ids()
        if not ids:
            return
        active = self.pipeline.recording_id if self.pipeline is not None else None
        position = ids.index(active) + step if active in ids else (0 if step > 0 else len(ids) - 1)
        self.set_active_recording(ids[position % len(ids)])

    def set_active_recording(self, recording_id: str) -> None:
        if self.busy or self.project is None:
            return
        if self.pipeline is not None and self.pipeline.recording_id == recording_id:
            return
        try:
            self.pipeline = self.project.pipeline(recording_id)
        except (PL.PipelineError, ProjectError, R.RegistryError, ValueError) as error:
            self.pipeline = None
            self.notify("Recording", str(error))
        self.params = self.pipeline.params if self.pipeline is not None else PipelineParams()
        self.refresh()

    # ── parameters and state ────────────────────────────────────────────────

    def set_stage_params(self, stage: str, params, refresh_tabs: bool = True) -> None:
        """Take an edited parameter set. Affected stages show as stale at once."""
        self.params = dataclasses.replace(self.params, **{f"stage_{stage}": params})
        if self.pipeline is not None and not self.busy:
            self.pipeline.set_params(self.params)
            self.statuses = self.pipeline.statuses()
            self._refresh_stage_list()
            if refresh_tabs:
                self._refresh_current_tab()

    def refresh(self) -> None:
        """Re-read stage states (when idle) and update everything visible."""
        if self.pipeline is not None and not self.busy:
            try:
                self.statuses = self.pipeline.statuses()
            except PL.PipelineError as error:
                self.statuses = {key: PL.STATUS_MISSING for key in PL.STAGE_KEYS}
                self.progress_label.setText(str(error))
        elif self.pipeline is None:
            self.statuses = {key: PL.STATUS_MISSING for key in PL.STAGE_KEYS}
        self.recording_label.setText(self._recording_caption())
        self._refresh_recording_selector()
        self.run_all_button.setEnabled(self.pipeline is not None and not self.busy)
        self.cancel_button.setEnabled(self.busy and self._worker is not None)
        self._refresh_stage_list()
        self.project_tab.refresh()
        self._refresh_current_tab()

    def _refresh_stage_list(self) -> None:
        self.stage_list.clear()
        for key in PL.STAGE_KEYS:
            symbol = RUNNING_SYMBOL if key == self._running_stage else STATUS_SYMBOL[self.statuses[key]]
            item = QListWidgetItem(f"{symbol}  {key}  {PL.spec(key).label}")
            item.setData(Qt.UserRole, key)
            self.stage_list.addItem(item)

    def _refresh_current_tab(self) -> None:
        tab = self.tabs.currentWidget()
        if tab is not self.project_tab:
            tab.refresh()

    def _stage_clicked(self, item: QListWidgetItem) -> None:
        self.tabs.setCurrentIndex(self.stage_tab_index[item.data(Qt.UserRole)])

    def open_in_explorer(self, track_id: str) -> None:
        self.tabs.setCurrentIndex(self.explorer_index)
        self.explorer_tab.select_track(track_id)

    # ── running ─────────────────────────────────────────────────────────────

    def run_targets(self, targets: list[str] | None, force_stages: list[str] | None = None) -> None:
        """Bring stages up to date in a worker thread. ``None`` means every stage."""
        if self.pipeline is None or self.busy:
            return
        self.pipeline.set_params(self.params)
        self._worker = PipelineWorker(self.pipeline, targets, force=force_stages or False)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.failed.connect(self._on_failed)
        self._worker.signals.cancelled.connect(self._on_cancelled)
        self._set_busy(True, "Running…")
        self.thread_pool.start(self._worker)

    def cancel_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.progress_label.setText("Cancelling after the current stage…")

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        if not busy:
            self._worker, self._running_stage = None, None
        self.progress_label.setText(message)
        self.refresh()

    def _on_progress(self, stage: str, event: str, statuses: dict) -> None:
        self.statuses = statuses
        self._running_stage = stage if event == "started" else None
        if event == "started":
            self.progress_label.setText(f"Running stage {stage} ({PL.spec(stage).label})…")
        self._refresh_stage_list()

    def _on_finished(self, outcome: dict, statuses: dict) -> None:
        ran = [stage for stage, what in outcome.items() if what == "finished"]
        self.statuses = statuses
        self._set_busy(False, f"Done. Ran {', '.join(ran)}." if ran else "Everything was already up to date.")

    def _on_failed(self, stage: str, message: str, statuses: dict) -> None:
        self.statuses = statuses
        self._set_busy(False, f"Stage {stage} failed." if stage else "The run failed.")
        self.notify(f"Stage {stage} failed" if stage else "Run failed", message)

    def _on_cancelled(self, statuses: dict) -> None:
        self.statuses = statuses
        self._set_busy(False, "Cancelled. Finished stages are kept.")

    def begin_batch(self, message: str) -> None:
        """A batch owns the output folders until it ends: single runs are disabled."""
        self._set_busy(True, message)

    def end_batch(self, message: str) -> None:
        # The batch wrote through its own pipeline objects; reload the active recording from disk.
        recording_id = self.pipeline.recording_id if self.pipeline is not None else None
        self.pipeline = None
        self.busy = False
        if recording_id is not None and recording_id in self.project.recording_ids():
            self.set_active_recording(recording_id)
        self._set_busy(False, message)

    def reset_stage(self, stage: str) -> None:
        if self.pipeline is None or self.busy:
            return
        self.pipeline.reset(stage)
        self.refresh()

    def run_in_background(self, function: Callable, on_done: Callable, on_error: Callable, *args, **kwargs) -> None:
        """Run a core function off the GUI thread; call back on the GUI thread."""
        worker = FunctionWorker(function, *args, **kwargs)
        relay = _Relay(self, worker, on_done, on_error)
        self._background.append(relay)
        worker.signals.finished.connect(relay.done)
        worker.signals.failed.connect(relay.error)
        self.thread_pool.start(worker)

    def export_stage_figures(self, stage: str) -> None:
        """Write every figure of a stage to its figures folder, in a worker."""
        if self.pipeline is None or self.busy or self.statuses[stage] not in (PL.STATUS_OK, PL.STATUS_STALE):
            return
        pipeline = self.pipeline
        self._set_busy(True, f"Exporting figures of stage {stage}…")

        def export():
            return pipeline.save_figures(stage, catalog.stage_figures(pipeline, stage), formats=("png", "pdf"))

        def done(written):
            self._set_busy(False, f"Wrote {len(written)} files to {pipeline.figure_folder(stage)}")

        def failed(message):
            self._set_busy(False, "Figure export failed.")
            self.notify("Export figures", message)

        self.run_in_background(export, done, failed)

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.batch_tab.cancel()
        self.batch_tab.pool.waitForDone(60000)
        self.thread_pool.waitForDone(60000)
        super().closeEvent(event)
