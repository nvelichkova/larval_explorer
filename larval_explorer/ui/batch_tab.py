"""The Batch tab: run many recordings unattended, then pick one to inspect."""

from __future__ import annotations

import threading

from PyQt5.QtCore import Qt, QThreadPool, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larval_explorer.core import batch as B
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.workers.tasks import BatchRecordingWorker

RESULT_TINT = {
    B.RESULT_OK: QColor("#dff3e4"),
    B.RESULT_UP_TO_DATE: QColor("#eef1f4"),
    B.RESULT_FAILED: QColor("#f8d7d7"),
    B.RESULT_CANCELLED: QColor("#fdf0d5"),
}
RESULT_TEXT = {
    B.RESULT_OK: "✓ done",
    B.RESULT_UP_TO_DATE: "✓ already up to date",
    B.RESULT_FAILED: "✗ failed",
    B.RESULT_CANCELLED: "cancelled",
}


class BatchTab(QWidget):
    """Tick recordings, choose how far to run, press Run. Failures do not stop the rest."""

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.pool = QThreadPool(self)
        self._workers: list[BatchRecordingWorker] = []
        self._cancel = threading.Event()
        self._log: B.BatchLog | None = None
        self._pending = 0
        self._rows: dict[str, int] = {}
        self.outcomes: list[B.RecordingOutcome] = []
        self.running = False

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemDoubleClicked.connect(self._inspect)
        select_all, select_none = QPushButton("Tick all"), QPushButton("Tick none")
        select_all.clicked.connect(lambda: self._tick_all(True))
        select_none.clicked.connect(lambda: self._tick_all(False))

        self.through_combo = QComboBox()
        for key in PL.STAGE_KEYS:
            self.through_combo.addItem(f"{key}  {PL.spec(key).label}", key)
        self.through_combo.setCurrentIndex(len(PL.STAGE_KEYS) - 1)
        self.force_box = QCheckBox("Re-run stages even if their results are up to date")
        self.figures_box = QCheckBox("Also write figure files (slow: stage 09 draws five figures per track)")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(B.DEFAULT_WORKERS)
        self.same_params = QRadioButton("Same parameters for every recording (those shown in the stage tabs)")
        self.own_params = QRadioButton("Each recording keeps the parameters it was last run with")
        self.same_params.setChecked(True)
        self.run_button, self.cancel_button = QPushButton("Run batch"), QPushButton("Cancel")
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Failures appear here with the stage and the message.")

        form = QFormLayout()
        form.addRow("Run through stage", self.through_combo)
        form.addRow("Recordings at a time", self.workers_spin)
        form.addRow(self.same_params)
        form.addRow(self.own_params)
        form.addRow(self.force_box)
        form.addRow(self.figures_box)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        note = QLabel("Each recording is processed on its own; nothing is merged across recordings. "
                      "Stages that are already up to date are skipped.")
        note.setWordWrap(True)
        left_layout.addWidget(note)
        left_layout.addLayout(form)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.log_view, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        ticks = QHBoxLayout()
        ticks.addWidget(QLabel("Recordings. Double-click one to make it the active recording and inspect it."), 1)
        ticks.addWidget(select_all)
        ticks.addWidget(select_none)
        right_layout.addLayout(ticks)
        right_layout.addWidget(self.table, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 900])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ── table ──

    def refresh(self) -> None:
        project = self.host.project
        idle = project is not None and not self.host.busy
        self.run_button.setEnabled(idle and len(project.registry) > 0)
        self.cancel_button.setEnabled(self.running)
        for widget in (self.through_combo, self.workers_spin, self.same_params, self.own_params,
                       self.force_box, self.figures_box):
            widget.setEnabled(idle)
        if self.running:
            return
        registry = project.registry if project is not None else R.empty_registry()
        ids = registry[R.RECORDING_ID].tolist()
        metadata = R.metadata_columns(registry)
        headers = ["Recording", *metadata, "Progress", "Result"]
        shown = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        if ids == list(self._rows) and headers == shown:
            for row in range(len(ids)):   # same recordings: keep ticks and results, refresh metadata
                for column, name in enumerate(metadata, start=1):
                    self.table.item(row, column).setText(str(registry.iloc[row][name]))
            return
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(ids))
        self._rows = {}
        for row, recording_id in enumerate(ids):
            self._rows[recording_id] = row
            first = QTableWidgetItem(recording_id)
            first.setFlags(first.flags() | Qt.ItemIsUserCheckable)
            first.setCheckState(Qt.Checked)
            self.table.setItem(row, 0, first)
            for column, name in enumerate(metadata, start=1):
                self.table.setItem(row, column, QTableWidgetItem(str(registry.iloc[row][name])))
            self.table.setItem(row, 1 + len(metadata), QTableWidgetItem(""))
            self.table.setItem(row, 2 + len(metadata), QTableWidgetItem(""))
        self.table.resizeColumnsToContents()

    def _tick_all(self, ticked: bool) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(Qt.Checked if ticked else Qt.Unchecked)

    def ticked(self) -> list[str]:
        return [self.table.item(row, 0).text() for row in range(self.table.rowCount())
                if self.table.item(row, 0).checkState() == Qt.Checked]

    def _set_cell(self, recording_id: str, offset_from_end: int, text: str, tint: QColor | None = None) -> None:
        row = self._rows.get(recording_id)
        if row is None:
            return
        item = self.table.item(row, self.table.columnCount() - offset_from_end)
        item.setText(text)
        if tint is not None:
            item.setBackground(tint)

    def _inspect(self, item: QTableWidgetItem) -> None:
        if not self.running:
            self.host.set_active_recording(self.table.item(item.row(), 0).text())

    # ── running ──

    def options(self) -> B.BatchOptions:
        return B.BatchOptions(
            through_stage=self.through_combo.currentData(),
            force=self.force_box.isChecked(),
            params=self.host.params if self.same_params.isChecked() else None,
            export_figures=self.figures_box.isChecked(),
        )

    def start(self) -> None:
        project, recording_ids = self.host.project, self.ticked()
        if project is None or self.host.busy or not recording_ids:
            return
        options = self.options()
        self.outcomes = []
        self._cancel = threading.Event()
        self._pending = len(recording_ids)
        self._log = B.BatchLog(project.output_root, recording_ids, options, self.workers_spin.value())
        self.running = True
        self.host.begin_batch(f"Batch: {len(recording_ids)} recordings…")
        self.progress.setRange(0, len(recording_ids))
        self.progress.setValue(0)
        self.summary_label.setText("")
        self.log_view.setPlainText("")
        for recording_id in self._rows:
            self._set_cell(recording_id, 2, "queued" if recording_id in recording_ids else "")
            self._set_cell(recording_id, 1, "", QColor("white"))
        self.pool.setMaxThreadCount(self.workers_spin.value())
        self._workers = []
        for recording_id in recording_ids:
            worker = BatchRecordingWorker(project, recording_id, options, self._cancel)
            worker.signals.progress.connect(self._on_progress)
            worker.signals.finished.connect(self._on_finished)
            self._workers.append(worker)
            self.pool.start(worker)

    def cancel(self) -> None:
        if not self.running:
            return
        self._cancel.set()
        self.summary_label.setText("Cancelling: recordings in progress stop after their current stage.")

    @pyqtSlot(str, str, str)
    def _on_progress(self, recording_id: str, stage: str, event: str) -> None:
        if event == "started":
            text = "writing figures…" if stage == "figures" else f"stage {stage} ({PL.spec(stage).label})…"
            self._set_cell(recording_id, 2, text)

    @pyqtSlot(object)
    def _on_finished(self, outcome: B.RecordingOutcome) -> None:
        self.outcomes.append(outcome)
        self._log.record(outcome)
        ran = f"ran {', '.join(outcome.stages_run)}" if outcome.stages_run else "nothing to run"
        if outcome.result == B.RESULT_FAILED:
            ran = f"stopped at stage {outcome.failed_stage}" if outcome.failed_stage else "could not start"
        self._set_cell(outcome.recording_id, 2, f"{ran} ({outcome.duration_s:.1f} s)")
        result = RESULT_TEXT[outcome.result]
        if outcome.result == B.RESULT_FAILED and outcome.failed_stage:
            result += f" at stage {outcome.failed_stage}"
        self._set_cell(outcome.recording_id, 1, result, RESULT_TINT[outcome.result])
        if outcome.result == B.RESULT_FAILED:
            self.log_view.appendPlainText(f"{outcome.recording_id}: {outcome.message}\n")
        self._pending -= 1
        self.progress.setValue(self.progress.maximum() - self._pending)
        if self._pending == 0:
            self._finish()

    def _finish(self) -> None:
        data = self._log.finish()
        summary = ", ".join(f"{count} {result}" for result, count in data["summary"].items())
        self.running = False
        self._workers = []
        self.table.resizeColumnsToContents()
        self.host.end_batch(f"Batch finished: {summary}.")
        self.summary_label.setText(f"Finished: {summary}.")
        self.log_view.appendPlainText(f"Run log: {self._log.path}")
