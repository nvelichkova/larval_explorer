"""The Turns tab: stage 12 over the ticked recordings, with its figures and export (D-037)."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
from larval_explorer.core import registry as R
from larval_explorer.core import schema as S
from larval_explorer.core import turn_analysis as TA
from larval_explorer.plots import turns as TP
from larval_explorer.ui.widgets import FigurePanel, TablePreview

# The rows of the summary shown on screen: the primary measures, all turns.
HEADLINE_MEASURES = TA.SIGNED_MEASURES[:2] + (S.BEND_LAG_12_S, S.BEND_LAG_12_SEGLEN)
HEADLINE_COLUMNS = [S.CONDITION, S.MEASURE, S.N_RECORDINGS, S.N_LARVAE, S.N_TURNS, S.TURN_Q25, S.TURN_MEDIAN,
                    S.TURN_Q75, S.SHARE_NEGATIVE, S.LARVA_MEAN, S.LARVA_SD, S.LARVA_SE]


def _run_turn_analysis(project, recording_ids, params, group_keys):
    """Runs in a worker. Brings stages 02, 04 and 09 up to date with the parameters shown, then analyses."""
    inputs = []
    for recording_id in recording_ids:
        pipeline = project.pipeline(recording_id, params)
        pipeline.run(list(TA.INPUT_STAGES))
        inputs.append(TA.load_turn_inputs(pipeline))
    conditions = A.condition_labels(project.registry, recording_ids, group_keys) if group_keys else None
    groups = {rid: {key: R.metadata_for(project.registry, rid)[key] for key in group_keys} for rid in recording_ids}
    # Stage 12 is not a pipeline stage (D-037): it runs with its own defaults, recorded in the export manifest.
    return TA.turn_analysis(inputs, conditions, None, group_keys, groups)


class TurnsTab(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.analysis: TA.TurnAnalysis | None = None
        self._ids: list[str] = []
        self._columns: list[str] = []

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.group_box = QGroupBox("Group recordings by (none: one population)")
        self.group_layout = QVBoxLayout(self.group_box)
        self.group_checks: dict[str, QCheckBox] = {}
        self.run_button = QPushButton("Run the turn analysis on the ticked recordings")
        self.run_button.clicked.connect(self.run)
        self.export_button = QPushButton("Export tables and figures…")
        self.export_button.clicked.connect(self.export_dialog)
        self.export_button.setEnabled(False)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #a15c00;")
        self.summary = TablePreview()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        note = QLabel(
            "Measures each turn found by stage 09: how much of its heading change has happened by the end of the "
            "cast (the sp1 bend peak), and how long the bend takes to move from sp1 to sp2. The end-of-turn "
            "threshold is pooled over the ticked recordings, so results belong to this set of recordings. Stages "
            "02, 04 and 09 are brought up to date first, with the parameters shown in the stage tabs. A condition "
            "with one recording gets numbers but no contrast."
        )
        note.setWordWrap(True)
        left_layout.addWidget(note)
        left_layout.addWidget(QLabel("Recordings"))
        left_layout.addWidget(self.table, 2)
        left_layout.addWidget(self.group_box)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.export_button)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.warning_label)
        left_layout.addWidget(QLabel("Primary measures"))
        left_layout.addWidget(self.summary, 1)

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
        self.run_button.setEnabled(idle)
        self.group_box.setEnabled(idle)
        self.export_button.setEnabled(idle and self.analysis is not None)
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
            for column in columns:
                check = QCheckBox(column)   # unticked: with one recording per condition there is no contrast to make
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

    def run(self) -> None:
        project, recording_ids = self.host.project, self.ticked()
        if project is None or self.host.busy or not recording_ids:
            return
        self.host.begin_batch(f"Turn analysis over {len(recording_ids)} recordings…")
        self.status_label.setText("Working…")
        self.host.run_in_background(_run_turn_analysis, self._done, self._failed,
                                    project, recording_ids, self.host.params, self.group_keys())

    def _done(self, analysis: TA.TurnAnalysis) -> None:
        self.analysis = analysis
        self.host.end_batch("Turn analysis finished.")
        self.status_label.setText(
            f"{len(analysis.recording_ids)} recordings; pooled end-of-turn threshold "
            f"{analysis.params.end_total_bend_deg:.2f}° (per recording: "
            + ", ".join(f"{k} {v:.2f}°" for k, v in analysis.end_threshold_per_recording.items()) + ")."
        )
        self.warning_label.setText("\n".join(analysis.warnings))
        summary = analysis.tables["summary"]
        self.summary.set_table(summary[(summary[S.SUBSET] == TA.SUBSET_ALL)
                                       & summary[S.MEASURE].isin(HEADLINE_MEASURES)][HEADLINE_COLUMNS], "summary")
        self.figures.set_builders({
            "heading_curves_progress": lambda: TP.heading_curves(analysis, S.CURVE_PROGRESS),
            "heading_curves_distance": lambda: TP.heading_curves(analysis, S.CURVE_DISTANCE),
            "signed_fraction": lambda: TP.signed_fraction(analysis),
            "bend_lag": lambda: TP.bend_lag(analysis),
            "smoothing_sensitivity": lambda: TP.smoothing_sensitivity(analysis),
            "censoring": lambda: TP.censoring(analysis),
        })
        self.export_button.setEnabled(True)

    def _failed(self, message: str) -> None:
        self.host.end_batch("The turn analysis stopped.")
        self.status_label.setText(message.strip().splitlines()[-1].split("AnalysisError: ")[-1])

    def export_to(self, folder) -> None:
        if self.analysis is None or self.host.busy:
            return
        analysis = self.analysis
        self.host.begin_batch("Exporting the turn analysis…")

        def export():
            return TA.export_turn_analysis(folder, analysis, TP.turn_figures(analysis), TP.FIGURE_TITLES)

        def done(manifest_path):
            self.host.end_batch("Turn analysis exported.")
            self.status_label.setText(f"Exported to {manifest_path.parent}")

        self.host.run_in_background(export, done, self._failed)

    def export_dialog(self) -> None:
        project = self.host.project
        if project is None:
            return
        suggested = TA.new_turn_analysis_folder(project.output_root)
        suggested.parent.mkdir(parents=True, exist_ok=True)
        parent = QFileDialog.getExistingDirectory(
            self, "Where to put the turn analysis (a dated subfolder is created)", str(suggested.parent))
        if parent:
            self.export_to(Path(parent) / suggested.name)
