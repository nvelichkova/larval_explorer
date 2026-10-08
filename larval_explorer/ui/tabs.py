"""The tabs of the main window.

A tab shows state and collects input. It computes nothing scientific: every
number comes from ``core`` and every figure from ``plots`` (CLAUDE.md §3).
Tabs talk to the window through a small host interface: ``pipeline``,
``params``, ``busy``, ``set_stage_params``, ``run_targets``, ``reset_stage``,
``run_in_background`` and ``open_in_explorer``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larval_explorer.core import pipeline as PL
from larval_explorer.core import registry as R
from larval_explorer.core import schema as S
from larval_explorer.core.project import EXAMPLE_FILENAME_PATTERN
from larval_explorer.core.segments import segment_tracks
from larval_explorer.plots import catalog, explorer, segmentation, style
from larval_explorer.ui.widgets import FigurePanel, ParamsForm, TablePreview

STATUS_TEXT = {
    PL.STATUS_OK: "✓ up to date",
    PL.STATUS_STALE: "⚠ stale: parameters or inputs changed since this ran",
    PL.STATUS_MISSING: "○ not run yet",
    PL.STATUS_FAILED: "✗ failed",
}
UNLABELLED_TINT = QColor("#fdf0d5")
UNMATCHED_TINT = QColor("#f8d7d7")


def _scrolling(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setFrameShape(QScrollArea.NoFrame)
    return area


# ──────────────────────────────────────────────────────────────────────────────
# Project
# ──────────────────────────────────────────────────────────────────────────────

class _AutocompleteDelegate(QStyledItemDelegate):
    """Offers the values already used in a metadata column, so typos do not make new groups."""

    def __init__(self, tab: "ProjectTab"):
        super().__init__(tab)
        self._tab = tab

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        project = self._tab.host.project
        if project is not None:
            column = self._tab.table.horizontalHeaderItem(index.column()).text()
            if column not in R.REQUIRED_COLUMNS:
                completer = QCompleter(R.existing_values(project.registry, column), editor)
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                editor.setCompleter(completer)
        return editor


class ProjectTab(QWidget):
    """Create or open a project, edit the registry, fill metadata from file names."""

    registry_changed = pyqtSignal()

    def __init__(self, host):
        super().__init__()
        self.host = host
        self._filling = False

        self.folder_label = QLabel("No project open.")
        new_button, open_button = QPushButton("New project…"), QPushButton("Open project…")
        new_button.clicked.connect(self.host.new_project_dialog)
        open_button.clicked.connect(self.host.open_project_dialog)
        self.output_edit = QLineEdit()
        self.output_edit.editingFinished.connect(self._output_edited)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)

        top = QHBoxLayout()
        top.addWidget(new_button)
        top.addWidget(open_button)
        top.addWidget(self.folder_label, 1)
        output = QHBoxLayout()
        output.addWidget(QLabel("Output folder"))
        output.addWidget(self.output_edit, 1)
        output.addWidget(browse)

        self.table = QTableWidget()
        self.table.setItemDelegate(_AutocompleteDelegate(self))
        self.table.itemChanged.connect(self._cell_edited)
        self.add_files_button, self.add_folder_button = QPushButton("Add recordings…"), QPushButton("Add folder…")
        self.add_column_button, self.remove_button = QPushButton("Add metadata column…"), QPushButton("Remove selected")
        self.add_files_button.clicked.connect(self._add_files)
        self.add_folder_button.clicked.connect(self._add_folder)
        self.add_column_button.clicked.connect(self._add_column)
        self.remove_button.clicked.connect(self._remove_selected)
        buttons = QHBoxLayout()
        for button in (self.add_files_button, self.add_folder_button, self.add_column_button, self.remove_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.unlabelled_label = QLabel("")

        self.pattern_edit = QLineEdit()
        self.pattern_edit.setPlaceholderText(EXAMPLE_FILENAME_PATTERN)
        self.preview_button, self.apply_button = QPushButton("Preview"), QPushButton("Apply to blank cells")
        self.overwrite_box = QCheckBox("overwrite filled cells")
        self.preview_button.clicked.connect(self.preview_parse)
        self.apply_button.clicked.connect(self.apply_parse)
        self.apply_button.setEnabled(False)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_label = QLabel("Nothing is changed until you apply a preview.")
        self._preview: pd.DataFrame | None = None
        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Pattern"))
        pattern_row.addWidget(self.pattern_edit, 1)
        pattern_row.addWidget(self.preview_button)
        pattern_row.addWidget(self.apply_button)
        pattern_row.addWidget(self.overwrite_box)
        parse_box = QGroupBox("Fill metadata from file names")
        parse_layout = QVBoxLayout(parse_box)
        parse_layout.addLayout(pattern_row)
        parse_layout.addWidget(self.preview_label)
        parse_layout.addWidget(self.preview_table)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(output)
        layout.addWidget(QLabel("Recordings (one CSV each). Select a row to make it the active recording."))
        layout.addWidget(self.table, 3)
        layout.addLayout(buttons)
        layout.addWidget(self.unlabelled_label)
        layout.addWidget(parse_box, 2)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.refresh()

    # ── display ──

    def refresh(self) -> None:
        project = self.host.project
        enabled = project is not None
        for widget in (self.output_edit, self.table, self.add_files_button, self.add_folder_button,
                       self.add_column_button, self.remove_button, self.pattern_edit, self.preview_button):
            widget.setEnabled(enabled)
        if project is None:
            self.folder_label.setText("No project open.")
            self.table.setRowCount(0)
            return
        self.folder_label.setText(str(project.folder))
        self.output_edit.setText(str(project.output_root))
        if not self.pattern_edit.text():
            self.pattern_edit.setText(project.filename_pattern)
        self._fill_table()

    def _fill_table(self) -> None:
        registry = self.host.project.registry
        unlabelled = set(R.unlabelled(registry)[R.RECORDING_ID])
        self._filling = True
        self.table.setColumnCount(len(registry.columns))
        self.table.setHorizontalHeaderLabels([str(column) for column in registry.columns])
        self.table.setRowCount(len(registry))
        for row in range(len(registry)):
            for column, name in enumerate(registry.columns):
                item = QTableWidgetItem(str(registry.iloc[row][name]))
                if name == R.PATH:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if registry.iloc[row][R.RECORDING_ID] in unlabelled:
                    item.setBackground(UNLABELLED_TINT)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._filling = False
        count = len(unlabelled)
        self.unlabelled_label.setText(
            f"{count} recording(s) have a blank metadata cell (tinted). They would form a group of their own."
            if count else ""
        )

    # ── edits ──

    def _commit(self, registry: pd.DataFrame) -> None:
        project = self.host.project
        try:
            R.validate_registry(registry)
        except R.RegistryError as error:
            self.host.notify("Registry", str(error))
            self._fill_table()
            return
        project.registry = registry
        project.save()
        self._fill_table()
        self.registry_changed.emit()

    def _cell_edited(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        registry = self.host.project.registry.copy()
        registry.iat[item.row(), item.column()] = item.text().strip()
        self._commit(registry)

    def _output_edited(self) -> None:
        project = self.host.project
        if project is None or not self.output_edit.text().strip():
            return
        if Path(self.output_edit.text().strip()) != project.output_root:
            project.output_root = Path(self.output_edit.text().strip())
            project.save()
            self.registry_changed.emit()

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)
            self._output_edited()

    def add_paths(self, paths) -> None:
        try:
            registry = R.add_recordings(self.host.project.registry, paths)
        except R.RegistryError as error:
            self.host.notify("Add recordings", str(error))
            return
        self._commit(registry)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add recordings", "", "FIM-Track export (*.csv)")
        if paths:
            self.add_paths(paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add every CSV in a folder")
        if folder:
            self.add_paths([folder])

    def _add_column(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add metadata column", "Column name (for example: age)")
        name = name.strip()
        if not accepted or not name:
            return
        registry = self.host.project.registry.copy()
        if name in registry.columns:
            self.host.notify("Add metadata column", f"'{name}' already exists.")
            return
        registry[name] = pd.Series("", index=registry.index, dtype="string")
        self._commit(registry)

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            return
        answer = QMessageBox.question(
            self, "Remove recordings",
            f"Remove {len(rows)} recording(s) from the registry? Their result folders are not deleted.",
        )
        if answer == QMessageBox.Yes:
            registry = self.host.project.registry.drop(index=self.host.project.registry.index[rows]).reset_index(drop=True)
            self._commit(registry)

    def _selection_changed(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) == 1 and not self._filling:
            self.host.set_active_recording(self.host.project.registry.iloc[rows.pop()][R.RECORDING_ID])

    # ── filename parsing ──

    def preview_parse(self) -> None:
        project = self.host.project
        pattern = self.pattern_edit.text().strip() or EXAMPLE_FILENAME_PATTERN
        try:
            preview = R.preview_parse(project.registry, pattern)
        except R.RegistryError as error:
            self.preview_label.setText(str(error))
            self.apply_button.setEnabled(False)
            return
        self._preview = preview
        self.pattern_edit.setText(pattern)
        project.filename_pattern = pattern
        project.save()
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels([str(column) for column in preview.columns])
        self.preview_table.setRowCount(len(preview))
        for row in range(len(preview)):
            matched = bool(preview.iloc[row][R.MATCHED])
            for column, name in enumerate(preview.columns):
                text = ("yes" if matched else "NO MATCH") if name == R.MATCHED else str(preview.iloc[row][name])
                item = QTableWidgetItem(text)
                if not matched:
                    item.setBackground(UNMATCHED_TINT)
                self.preview_table.setItem(row, column, item)
        self.preview_table.resizeColumnsToContents()
        unmatched = int((~preview[R.MATCHED]).sum())
        self.preview_label.setText(
            f"{len(preview) - unmatched} of {len(preview)} names match. "
            + (f"{unmatched} do not and will be left blank." if unmatched else "")
            + " Nothing is changed until you apply."
        )
        self.apply_button.setEnabled(len(preview) > unmatched)

    def apply_parse(self) -> None:
        if self._preview is None:
            return
        self._commit(R.apply_parse(self.host.project.registry, self._preview, overwrite=self.overwrite_box.isChecked()))
        self.apply_button.setEnabled(False)
        self.preview_label.setText("Applied. Edit any cell in the registry to override it.")


# ──────────────────────────────────────────────────────────────────────────────
# A stage: parameters, run, figures, output table
# ──────────────────────────────────────────────────────────────────────────────

class StageTab(QWidget):
    """Parameters on the left; figures and an output preview on the right."""

    PREVIEW_TRACKS = None  # list every track in the figure selector

    def __init__(self, host, stages: tuple[str, ...], intro: str = ""):
        super().__init__()
        self.host = host
        self.stages = stages
        self.forms: dict[str, ParamsForm] = {}

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        if intro:
            text = QLabel(intro)
            text.setWordWrap(True)
            controls_layout.addWidget(text)
        self.extra_layout = QVBoxLayout()
        controls_layout.addLayout(self.extra_layout)
        for stage in stages:
            box = QGroupBox(f"{stage}  {PL.spec(stage).label}")
            form = ParamsForm(getattr(host.params, f"stage_{stage}"))
            form.changed.connect(lambda params, stage=stage: self.host.set_stage_params(stage, params))
            self.forms[stage] = form
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(form)
            controls_layout.addWidget(box)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.run_button = QPushButton("Run")
        self.force_button = QPushButton("Run again")
        self.reset_button = QPushButton("Reset…")
        self.export_button = QPushButton("Export all figures")
        self.run_button.clicked.connect(lambda: self.host.run_targets(list(self.stages)))
        self.force_button.clicked.connect(lambda: self.host.run_targets(list(self.stages), force_stages=list(self.stages)))
        self.reset_button.clicked.connect(self._reset)
        self.export_button.clicked.connect(lambda: self.host.export_stage_figures(self.stages[-1]))
        row = QHBoxLayout()
        for button in (self.run_button, self.force_button, self.reset_button):
            row.addWidget(button)
        controls_layout.addLayout(row)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.status_label)
        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setPlaceholderText("Counts and warnings from the last run appear here.")
        controls_layout.addWidget(self.messages, 1)

        self.figures = FigurePanel()
        self.table_selector = QComboBox()
        self.table_selector.currentTextChanged.connect(self._show_table)
        self.table_preview = TablePreview()
        tables = QWidget()
        tables_layout = QVBoxLayout(tables)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Output table"))
        selector_row.addWidget(self.table_selector, 1)
        tables_layout.addLayout(selector_row)
        tables_layout.addWidget(self.table_preview, 1)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.figures)
        right.addWidget(tables)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        right.setSizes([620, 260])
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(_scrolling(controls))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _reset(self) -> None:
        stage = self.stages[0]
        downstream = PL.dependents(stage)
        answer = QMessageBox.question(
            self, "Reset",
            f"Delete the results of stage {stage} and of every stage after it ({', '.join(downstream) or 'none'})?",
        )
        if answer == QMessageBox.Yes:
            self.host.reset_stage(stage)

    def _tables(self) -> dict[str, tuple[str, str]]:
        return {f"{stage}: {output.table}": (stage, output.table) for stage in self.stages for output in PL.spec(stage).outputs}

    def _show_table(self, label: str) -> None:
        pipeline, tables = self.host.pipeline, self._tables()
        if pipeline is None or label not in tables:
            self.table_preview.set_table(None)
            return
        stage, name = tables[label]
        try:
            self.table_preview.set_table(pipeline.load_table(stage, name), label)
        except PL.PipelineError:
            self.table_preview.set_table(None)

    def refresh(self) -> None:
        """Bring the tab in line with the active recording and its stage states."""
        pipeline, busy = self.host.pipeline, self.host.busy
        for stage, form in self.forms.items():
            form.set_value(getattr(self.host.params, f"stage_{stage}"))
            form.setEnabled(pipeline is not None and not busy)
        for button in (self.run_button, self.force_button, self.reset_button, self.export_button):
            button.setEnabled(pipeline is not None and not busy)
        if pipeline is None:
            self.status_label.setText("Pick a recording on the Project tab.")
            self.figures.set_builders({}, "Pick a recording on the Project tab.")
            self.table_preview.set_table(None)
            self.messages.setPlainText("")
            return
        statuses = self.host.statuses
        self.status_label.setText("\n".join(f"Stage {stage}: {STATUS_TEXT[statuses[stage]]}" for stage in self.stages))
        if busy:
            return
        self._refresh_results(pipeline, statuses)

    def _refresh_results(self, pipeline, statuses) -> None:
        shown = [stage for stage in self.stages if statuses[stage] in (PL.STATUS_OK, PL.STATUS_STALE)]
        builders = {}
        for stage in shown:
            builders.update(catalog.figure_builders(pipeline, stage, self.PREVIEW_TRACKS))
        self.figures.set_builders(builders, "No result yet. Press Run.")
        labels = [label for label, (stage, _) in self._tables().items() if stage in shown]
        current = self.table_selector.currentText()
        self.table_selector.blockSignals(True)
        self.table_selector.clear()
        self.table_selector.addItems(labels)
        if current in labels:
            self.table_selector.setCurrentText(current)
        self.table_selector.blockSignals(False)
        self._show_table(self.table_selector.currentText())

        lines = []
        for stage in self.stages:
            record = pipeline.manifest["stages"].get(stage)
            if record is None:
                continue
            if record["status"] == "failed":
                lines.append(f"Stage {stage} failed:\n{record.get('error', '')}")
                continue
            lines.append(f"Stage {stage} ({record['duration_s']} s)")
            if stage == "06" and pipeline.pool is not None:
                lines.append(f"  states come from one HMM fitted over {len(pipeline.pool.recording_ids)} recordings: "
                             + ", ".join(pipeline.pool.recording_ids))
            lines += [f"  {key}: {value}" for key, value in record["diagnostics"].items()
                      if not isinstance(value, (list, dict)) or len(str(value)) < 80]
            lines += [f"  warning: {warning}" for warning in record["warnings"]]
        self.messages.setPlainText("\n".join(lines))


class IngestTab(StageTab):
    """Stages 00 and 01, with the validation report of the raw file."""

    def __init__(self, host):
        super().__init__(host, ("00", "01"), "Check the raw FIM-Track export, then reorganise and calibrate it. "
                                              "Frame rate and pixel scale are per recording: confirm them against your rig.")
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(130)
        self.report.setPlaceholderText("The channel validation report appears here.")
        self.extra_layout.addWidget(self.source_label)
        self.extra_layout.addWidget(self.report)
        self._validated_for: str | None = None

    def refresh(self) -> None:
        super().refresh()
        pipeline = self.host.pipeline
        if pipeline is None:
            self.source_label.setText("")
            self.report.setPlainText("")
            self._validated_for = None
            return
        self.source_label.setText(f"File: {pipeline.source_file}")
        key = str(pipeline.source_file)
        if key != self._validated_for:
            self._validated_for = key
            self.report.setPlainText("Reading the file…")
            fps = self.host.params.stage_01.frame_rate_fps
            self.host.run_in_background(_validation_text, self._show_report, self._show_report, pipeline.source_file, fps)

    def _show_report(self, text: str) -> None:
        self.report.setPlainText(text)


def _validation_text(source_file: Path, frame_rate_fps: float) -> str:
    if not Path(source_file).is_file():
        return f"File not found: {source_file}"
    report = S.validate_fimtrack(PL.read_fimtrack_csv(source_file))
    duration = report.duration_seconds(frame_rate_fps)
    return f"{report.summary()}\nduration at {frame_rate_fps:g} fps: {duration:.1f} s ({duration / 60:.1f} min)"


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation (stage 02)
# ──────────────────────────────────────────────────────────────────────────────

class SegmentationTab(QWidget):
    """Live preview of track segmentation while the two tolerances are dragged.

    The preview is computed in memory and recorded nowhere. "Apply and run"
    stores the parameters and runs stage 02 through the pipeline.
    """

    PREVIEW_DELAY_MS = 300

    def __init__(self, host):
        super().__init__()
        self.host = host
        self._request = 0
        self._calibrated: pd.DataFrame | None = None
        self._calibrated_for: tuple | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start_preview)

        self.enabled_box = QCheckBox("Split tracks at gaps (off = upstream behaviour: gaps ignored)")
        self.bridge_spin, self.bridge_slider = QSpinBox(), QSlider(Qt.Horizontal)
        self.bridge_spin.setRange(0, 5000)
        self.bridge_slider.setRange(0, 200)
        self.minimum_spin, self.minimum_slider = QDoubleSpinBox(), QSlider(Qt.Horizontal)
        self.minimum_spin.setRange(0.0, 36000.0)
        self.minimum_spin.setDecimals(1)
        self.minimum_slider.setRange(0, 600)
        self.bridge_slider.valueChanged.connect(self.bridge_spin.setValue)
        self.bridge_spin.valueChanged.connect(lambda value: (self._sync(self.bridge_slider, value), self._edited()))
        self.minimum_slider.valueChanged.connect(lambda value: self.minimum_spin.setValue(float(value)))
        self.minimum_spin.valueChanged.connect(lambda value: (self._sync(self.minimum_slider, int(value)), self._edited()))
        self.enabled_box.stateChanged.connect(self._edited)

        form = QFormLayout()
        form.addRow(self.enabled_box)
        bridge_row, minimum_row = QHBoxLayout(), QHBoxLayout()
        bridge_row.addWidget(self.bridge_slider, 1)
        bridge_row.addWidget(self.bridge_spin)
        minimum_row.addWidget(self.minimum_slider, 1)
        minimum_row.addWidget(self.minimum_spin)
        form.addRow("Bridge gaps up to (frames)", bridge_row)
        form.addRow("Keep segments of at least (s)", minimum_row)
        self.apply_button = QPushButton("Apply and run stage 02")
        self.apply_button.clicked.connect(self.apply)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary = TablePreview()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        note = QLabel("A lost track leaves a gap. Short gaps are bridged by interpolation; longer ones split the "
                      "track, and each piece is analysed as its own track. Click a larva panel to open it in the explorer.")
        note.setWordWrap(True)
        left_layout.addWidget(note)
        left_layout.addLayout(form)
        left_layout.addWidget(self.apply_button)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.summary, 1)

        self.figures = FigurePanel()
        self.figures.selector.hide()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.figures)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    @staticmethod
    def _sync(slider: QSlider, value: int) -> None:
        slider.blockSignals(True)
        slider.setValue(min(value, slider.maximum()))
        slider.blockSignals(False)

    def current_params(self):
        return dataclasses.replace(
            self.host.params.stage_02,
            enabled=self.enabled_box.isChecked(),
            bridge_max_frames=self.bridge_spin.value(),
            min_segment_seconds=self.minimum_spin.value(),
        )

    def _edited(self, *_args) -> None:
        if self.host.pipeline is None or self.host.busy:
            return
        self.host.set_stage_params("02", self.current_params(), refresh_tabs=False)
        self._update_status()
        self._timer.start(self.PREVIEW_DELAY_MS)

    def _update_status(self) -> None:
        pipeline = self.host.pipeline
        if pipeline is not None:
            self.status_label.setText(f"Stage 02: {STATUS_TEXT[self.host.statuses['02']]}")

    def refresh(self) -> None:
        pipeline, busy = self.host.pipeline, self.host.busy
        params = self.host.params.stage_02
        for widget, value in ((self.enabled_box, params.enabled), (self.bridge_spin, params.bridge_max_frames),
                              (self.minimum_spin, params.min_segment_seconds)):
            widget.blockSignals(True)
            widget.setChecked(value) if isinstance(widget, QCheckBox) else widget.setValue(value)
            widget.blockSignals(False)
        self._sync(self.bridge_slider, params.bridge_max_frames)
        self._sync(self.minimum_slider, int(params.min_segment_seconds))
        ready = pipeline is not None and not busy and self.host.statuses["01"] == PL.STATUS_OK
        for widget in (self.enabled_box, self.bridge_spin, self.bridge_slider, self.minimum_spin,
                       self.minimum_slider, self.apply_button):
            widget.setEnabled(ready)
        if pipeline is None:
            self.status_label.setText("Pick a recording on the Project tab.")
        elif self.host.statuses["01"] != PL.STATUS_OK:
            self.status_label.setText("Run Ingest (stages 00 and 01) first.")
            self.figures.set_figure(style.message_figure("Run Ingest (stages 00 and 01) first."))
            self.summary.set_table(None)
            self.summary_label.setText("")
        elif not busy:
            self._update_status()
            self._start_preview()

    def _start_preview(self) -> None:
        pipeline = self.host.pipeline
        if pipeline is None or self.host.busy or self.host.statuses["01"] != PL.STATUS_OK:
            return
        key = (pipeline.recording_id, pipeline.manifest["stages"]["01"]["fingerprint"])
        if key != self._calibrated_for:
            self._calibrated, self._calibrated_for = pipeline.table("01", "calibrated"), key
        params = self.current_params()
        if not params.enabled:
            self.figures.set_figure(style.message_figure(
                "Splitting is off: every track is kept whole if its first-to-last span\n"
                f"reaches {params.min_segment_seconds:g} s, and gaps are ignored, as upstream does."))
            self.summary.set_table(None)
            self.summary_label.setText("")
            return
        self._request += 1
        request = self._request
        self.host.run_in_background(
            segment_tracks,
            lambda result, request=request, params=params: self._show_preview(request, params, result),
            lambda message: self.status_label.setText(f"Preview failed:\n{message.splitlines()[-1]}"),
            self._calibrated, params, self.host.params.stage_01, recording_id=pipeline.recording_id,
        )

    def _show_preview(self, request: int, params, result) -> None:
        if request != self._request or self.host.pipeline is None:
            return   # a newer preview is on its way
        figure = segmentation.segmentation_overview(
            self._calibrated, result.table, result.segments, params, self.host.pipeline.recording_id
        )
        self.figures.set_figure(figure)
        self.figures.canvas.mpl_connect("button_press_event", lambda event: self._panel_clicked(event, result))
        d = result.diagnostics
        self.summary_label.setText(
            f"{d['larvae_in']} larvae → {d['segments_found']} segments found → {d['segments_out']} kept, "
            f"{d['segments_dropped']} dropped; {d['frames_bridged']} frames bridged."
        )
        self.summary.set_table(result.larva_summary, "Per larva")

    def _panel_clicked(self, event, result) -> None:
        source_id = event.inaxes.get_gid() if event.inaxes is not None else None
        if not source_id or self.figures._toolbar.mode:
            return   # not a larva panel, or the user is panning or zooming
        kept = result.segments[(result.segments[S.SOURCE_TRACK_ID] == source_id) & result.segments[S.KEPT]]
        if len(kept):
            self.host.open_in_explorer(kept.iloc[0][S.TRACK_ID])

    def apply(self) -> None:
        self.host.set_stage_params("02", self.current_params(), refresh_tabs=False)
        self.host.run_targets(["02"])


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Explorer
# ──────────────────────────────────────────────────────────────────────────────

class ExplorerTab(QWidget):
    """One track at a time, with layers that switch on and off independently."""

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.data = explorer.ExplorerData()
        self._limits: tuple | None = None
        self._track_drawn: str | None = None

        self.track_selector = QComboBox()
        self.track_selector.currentTextChanged.connect(lambda _text: self._draw(keep_view=False))
        self.layer_boxes: dict[str, QCheckBox] = {}
        layers_box = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers_box)
        for key, text in explorer.LAYERS.items():
            box = QCheckBox(text)
            box.setChecked(key in explorer.DEFAULT_LAYERS)
            box.stateChanged.connect(lambda _state: self._draw(keep_view=True))
            self.layer_boxes[key] = box
            layers_layout.addWidget(box)
        self.time_box = QCheckBox("Show position at time")
        self.time_box.stateChanged.connect(self._time_changed)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.valueChanged.connect(self._time_changed)
        self.time_label = QLabel("")
        self.note = QLabel("")
        self.note.setWordWrap(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Track"))
        left_layout.addWidget(self.track_selector)
        left_layout.addWidget(layers_box)
        left_layout.addWidget(self.time_box)
        left_layout.addWidget(self.time_slider)
        left_layout.addWidget(self.time_label)
        left_layout.addWidget(self.note)
        left_layout.addStretch(1)

        self.figures = FigurePanel()
        self.figures.selector.hide()
        self.figures.export_button.setText("Export current view…")
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.figures)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1000])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _load(self, pipeline) -> explorer.ExplorerData:
        def table(stage: str, name: str) -> pd.DataFrame:
            try:
                return pipeline.load_table(stage, name)
            except PL.PipelineError:
                return pd.DataFrame()

        filtered = table("07", "steps")
        return explorer.ExplorerData(
            trajectories=table("02", "trajectories"),
            smoothed=table("03", "smoothed"),
            rdp_steps=table("05", "steps"),
            hmm_steps=filtered if len(filtered) else table("06", "steps"),
            hmm_segments=table("08", "hmm_segments"),
            events=table("09", "events"),
        )

    def refresh(self) -> None:
        pipeline = self.host.pipeline
        if pipeline is None or self.host.busy:
            if pipeline is None:
                self.data = explorer.ExplorerData()
                self.track_selector.clear()
                self.figures.set_figure(style.message_figure("Pick a recording on the Project tab."))
            return
        self.data = self._load(pipeline)
        tracks = self.data.tracks()
        current = self.track_selector.currentText()
        self.track_selector.blockSignals(True)
        self.track_selector.clear()
        self.track_selector.addItems(tracks)
        if current in tracks:
            self.track_selector.setCurrentText(current)
        self.track_selector.blockSignals(False)
        available = self.data.available_layers()
        for key, box in self.layer_boxes.items():
            box.setEnabled(key in available)
        stale = [stage for stage, status in self.host.statuses.items() if status == PL.STATUS_STALE]
        self.note.setText(f"Stale stages shown as last computed: {', '.join(stale)}" if stale else "")
        self._draw(keep_view=current in tracks)

    def select_track(self, track_id: str) -> None:
        if self.track_selector.findText(track_id) >= 0:
            self.track_selector.setCurrentText(track_id)

    def _time(self) -> float | None:
        path = self.data.path(self.track_selector.currentText())
        if not self.time_box.isChecked() or not len(path):
            return None
        start, end = float(path[S.TIME_S].iloc[0]), float(path[S.TIME_S].iloc[-1])
        return start + (end - start) * self.time_slider.value() / self.time_slider.maximum()

    def _draw(self, keep_view: bool) -> None:
        track_id = self.track_selector.currentText()
        if not track_id:
            self.figures.set_figure(style.message_figure("Run the pipeline up to stage 02 to see trajectories."))
            return
        view = None
        if keep_view and self._track_drawn == track_id and self.figures.figure is not None and self.figures.figure.axes:
            axes = self.figures.figure.axes[0]
            view = (axes.get_xlim(), axes.get_ylim())
        layers = [key for key, box in self.layer_boxes.items() if box.isChecked() and box.isEnabled()]
        time_s = self._time()
        figure = explorer.trajectory_explorer(self.data, track_id, layers, time_s)
        if view is not None and figure.axes:
            figure.axes[0].set_xlim(view[0])
            figure.axes[0].set_ylim(view[1])
        self.figures.set_figure(figure)
        self._track_drawn = track_id
        self.time_label.setText("" if time_s is None else f"t = {time_s:.1f} s")

    def _time_changed(self, *_args) -> None:
        """Move the marker without redrawing the layers."""
        figure, time_s = self.figures.figure, self._time()
        marker = getattr(figure, "time_marker", None)
        if marker is None:
            return
        self.time_label.setText("" if time_s is None else f"t = {time_s:.1f} s")
        marker.set_visible(time_s is not None)
        if time_s is not None:
            x, y = explorer.position_at(self.data.path(self.track_selector.currentText()), time_s)
            marker.set_data([x], [y])
        self.figures.canvas.draw_idle()
