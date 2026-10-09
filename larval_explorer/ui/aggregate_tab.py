"""The Aggregate tab: one HMM for all ticked recordings, then compare conditions."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larval_explorer.core import aggregate as A
from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.plots import comparison as C
from larval_explorer.ui.widgets import FigurePanel, TablePreview


def _pooled_fit_and_check(project, recording_ids, params, group_keys, with_diagnostic, progress):
    """Runs in a worker: the pooled fit, then optionally a fit per condition for comparison."""
    outcome = A.run_pooled_analysis(project, recording_ids, params, progress=progress)
    diagnostic = None
    if with_diagnostic and group_keys:
        conditions = A.condition_labels(project.registry, recording_ids, group_keys)
        steps = {r: project.pipeline(r).table("05", "steps") for r in recording_ids}
        diagnostic = A.per_condition_hmm(steps, conditions, params.stage_06)
    return outcome, diagnostic


class AggregateTab(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.comparison: A.Comparison | None = None
        self.diagnostic = None
        self._ids: list[str] = []
        self._columns: list[str] = []

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.group_box = QGroupBox("Group recordings by")
        self.group_layout = QVBoxLayout(self.group_box)
        self.group_checks: dict[str, QCheckBox] = {}
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(list(A.UNITS))
        self.unit_combo.setCurrentText(A.DEFAULT_UNIT)
        self.diagnostic_box = QCheckBox("Also fit each condition separately, as a check on the shared states")
        self.fit_button = QPushButton("1. Fit one HMM across the ticked recordings")
        self.compare_button = QPushButton("2. Compare conditions")
        self.export_button = QPushButton("Export figure set…")
        self.excel_button = QPushButton("Export data to Excel…")
        self.excel_button.setToolTip("The numbers behind the figures, one row per unit, for running statistics.")
        self.excel_button.clicked.connect(self.excel_dialog)
        self.excel_button.setEnabled(False)
        self.fit_button.clicked.connect(self.fit_pooled)
        self.compare_button.clicked.connect(self.compare)
        self.export_button.clicked.connect(self.export_dialog)
        self.export_button.setEnabled(False)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #a15c00;")
        self.counts = TablePreview()

        form = QFormLayout()
        form.addRow("Each point is one", self.unit_combo)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        note = QLabel(
            "Step 1 fits the HMM once on all ticked recordings, so a state means the same thing in each, and "
            "re-runs their later stages on it using the parameters shown in the stage tabs. Step 2 summarises "
            "each unit and groups the units by the metadata you tick. Segments of one larva are not independent "
            "observations; the default unit is the larva."
        )
        note.setWordWrap(True)
        left_layout.addWidget(note)
        left_layout.addWidget(QLabel("Recordings"))
        left_layout.addWidget(self.table, 2)
        left_layout.addWidget(self.group_box)
        left_layout.addLayout(form)
        left_layout.addWidget(self.diagnostic_box)
        left_layout.addWidget(self.fit_button)
        buttons = QHBoxLayout()
        buttons.addWidget(self.compare_button)
        buttons.addWidget(self.export_button)
        left_layout.addLayout(buttons)
        buttons = QHBoxLayout()
        buttons.addWidget(self.excel_button)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.warning_label)
        left_layout.addWidget(QLabel("n per condition"))
        left_layout.addWidget(self.counts, 1)

        self.figures = FigurePanel()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.figures)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 900])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def refresh(self) -> None:
        project = self.host.project
        idle = project is not None and not self.host.busy
        for widget in (self.fit_button, self.compare_button, self.unit_combo, self.diagnostic_box, self.group_box):
            widget.setEnabled(idle)
        self.export_button.setEnabled(idle and self.comparison is not None)
        self.excel_button.setEnabled(idle and self.comparison is not None)
        if self.host.busy:
            return
        registry = project.registry if project is not None else R.empty_registry()
        ids, columns = registry[R.RECORDING_ID].tolist(), R.metadata_columns(registry)
        if ids != self._ids or columns != self._columns:
            self._ids, self._columns = ids, columns
            self.table.setColumnCount(1 + len(columns))
            self.table.setHorizontalHeaderLabels(["Recording", *columns])
            self.table.setRowCount(len(ids))
            for row, recording_id in enumerate(ids):
                first = QTableWidgetItem(recording_id)
                first.setFlags(first.flags() | Qt.ItemIsUserCheckable)
                first.setCheckState(Qt.Checked)
                self.table.setItem(row, 0, first)
                for column in range(len(columns)):
                    self.table.setItem(row, column + 1, QTableWidgetItem(""))
            for check in self.group_checks.values():
                self.group_layout.removeWidget(check)
                check.deleteLater()
            self.group_checks = {}
            for index, column in enumerate(columns):
                check = QCheckBox(column)
                check.setChecked(index == 0)
                self.group_checks[column] = check
                self.group_layout.addWidget(check)
        for row in range(len(ids)):
            for column, name in enumerate(columns, start=1):
                self.table.item(row, column).setText(str(registry.iloc[row][name]))
        self.table.resizeColumnsToContents()

    def ticked(self) -> list[str]:
        return [self.table.item(row, 0).text() for row in range(self.table.rowCount())
                if self.table.item(row, 0).checkState() == Qt.Checked]

    def group_keys(self) -> list[str]:
        return [column for column, check in self.group_checks.items() if check.isChecked()]

    # ── step 1 ──

    def fit_pooled(self) -> None:
        project, recording_ids = self.host.project, self.ticked()
        if project is None or self.host.busy or not recording_ids:
            return
        self.host.begin_batch(f"Pooled HMM over {len(recording_ids)} recordings…")
        self.status_label.setText("Working…")
        self.host.run_in_background(
            _pooled_fit_and_check, self._fit_done, self._failed,
            project, recording_ids, self.host.params, self.group_keys(), self.diagnostic_box.isChecked(), None,
        )

    def _fit_done(self, result) -> None:
        outcome, self.diagnostic = result
        means = ", ".join(f"state {k}: {v:.2f}" for k, v in sorted(outcome.state_mean_phi.items()))
        text = (f"One HMM fitted over {len(outcome.pool.recording_ids)} recordings."
                if outcome.refitted else "The pooled fit was already up to date.")
        text += f"\nFitted mean phi — {means}"
        if outcome.failures:
            text += "\nFailed after the fit: " + "; ".join(f"{r}: {m}" for r, m in outcome.failures.items())
        self.host.end_batch("Pooled HMM fit finished.")
        self.status_label.setText(text)
        self.compare()

    def _failed(self, message: str) -> None:
        self.host.end_batch("The analysis stopped.")
        # The last line of a traceback is the error itself; AnalysisError messages say what to change.
        self.status_label.setText(message.strip().splitlines()[-1].split("AnalysisError: ")[-1])

    # ── step 2 ──

    def compare(self) -> None:
        project, recording_ids = self.host.project, self.ticked()
        if project is None or self.host.busy or not recording_ids:
            return
        try:
            self.comparison = A.compare_recordings(project, recording_ids, self.group_keys(), self.unit_combo.currentText())
        except (A.AnalysisError, PL.PipelineError) as error:
            self.comparison = None
            self.warning_label.setText(str(error))
            self.figures.set_builders({}, "Nothing to compare yet.")
            self.counts.set_table(None)
            self.export_button.setEnabled(False)
            self.excel_button.setEnabled(False)
            return
        comparison, diagnostic = self.comparison, self.diagnostic
        builders = {
            "state_occupancy": lambda: C.by_state(comparison, A.MEASURE_OCCUPANCY),
            "dwell_times": lambda: C.by_state(comparison, A.MEASURE_DWELL),
            "steering_parameters": lambda: C.steering_parameters(comparison),
            "head_cast_rate": lambda: C.plain(comparison, A.MEASURE_HEAD_CAST_RATE),
            "crawl_length": lambda: C.plain(comparison, A.MEASURE_CRAWL_LENGTH),
        }
        if diagnostic is not None and len(diagnostic):
            builders["hmm_per_condition"] = lambda: C.hmm_per_condition(diagnostic)
        self.figures.set_builders(builders)
        self.counts.set_table(comparison.counts, "n per condition")
        self.warning_label.setText("\n".join(comparison.warnings))
        self.export_button.setEnabled(True)
        self.excel_button.setEnabled(True)

    def export_to(self, folder) -> None:
        """Write the whole comparison set to a folder, in a worker."""
        if self.comparison is None or self.host.busy:
            return
        comparison, diagnostic = self.comparison, self.diagnostic
        self.host.begin_batch("Exporting the comparison…")

        def export():
            return A.export_comparison(folder, comparison, C.comparison_figures(comparison, diagnostic), C.FIGURE_TITLES)

        def done(manifest_path):
            self.host.end_batch("Comparison exported.")
            self.status_label.setText(f"Exported to {manifest_path.parent}")

        self.host.run_in_background(export, done, self._failed)

    def export_excel_to(self, path) -> bool:
        """Write the current comparison's numbers to an Excel workbook."""
        if self.comparison is None:
            return False
        try:
            written = A.export_excel(path, self.comparison)
        except PermissionError:
            self.host.notify("Export data", f"Could not write {path}. If it is open in Excel, close it and try again.")
            return False
        except OSError as error:
            self.host.notify("Export data", str(error))
            return False
        self.status_label.setText(f"Data written to {written}")
        return True

    def excel_dialog(self) -> None:
        project = self.host.project
        if project is None or self.comparison is None:
            return
        suggested = A.new_comparison_folder(project.output_root)
        suggested.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export data to Excel", str(suggested.parent / f"{suggested.name}.xlsx"), "Excel workbook (*.xlsx)"
        )
        if path:
            self.export_excel_to(path if path.lower().endswith(".xlsx") else path + ".xlsx")

    def export_dialog(self) -> None:
        project = self.host.project
        if project is None:
            return
        suggested = A.new_comparison_folder(project.output_root)
        suggested.parent.mkdir(parents=True, exist_ok=True)
        parent = QFileDialog.getExistingDirectory(
            self, "Where to put the figure set (a dated subfolder is created)", str(suggested.parent)
        )
        if parent:
            self.export_to(Path(parent) / suggested.name)
