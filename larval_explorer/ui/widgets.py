"""Reusable widgets: a parameter form, a figure panel and a table preview."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping

import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from larval_explorer.plots import style

BIG = 1_000_000_000
PREVIEW_ROWS = 500


def _label_for(field_name: str) -> str:
    return field_name.replace("_", " ")


def _number(text: str):
    """An int when written as one, so '1, 99' stays (1, 99) and not (1.0, 99.0)."""
    text = text.strip()
    return int(text) if text.lstrip("+-").isdigit() else float(text)


class ParamsForm(QWidget):
    """Editors for every field of a frozen parameter dataclass.

    ``value()`` returns a new instance of the same dataclass. ``changed`` is
    emitted on every edit that yields a valid parameter set.
    """

    changed = pyqtSignal(object)

    def __init__(self, params: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self._params = params
        self._editors: dict[str, QWidget] = {}
        self._error = QLabel("")
        self._error.setStyleSheet("color: #d03b3b;")
        self._error.setWordWrap(True)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = dataclasses.fields(params)
        for field in fields:
            editor = self._make_editor(field.name, getattr(params, field.name), str(field.type))
            self._editors[field.name] = editor
            layout.addRow(_label_for(field.name), editor)
        if not fields:
            layout.addRow(QLabel("This stage has no parameters."))
        layout.addRow(self._error)

    def _make_editor(self, name: str, value: Any, annotation: str) -> QWidget:
        if isinstance(value, bool):
            editor = QCheckBox()
            editor.setChecked(value)
            editor.stateChanged.connect(self._on_edit)
        elif isinstance(value, int) and "None" not in annotation:
            editor = QSpinBox()
            editor.setRange(-BIG, BIG)
            editor.setValue(value)
            editor.valueChanged.connect(self._on_edit)
        elif isinstance(value, Mapping) or (isinstance(value, tuple) and value and isinstance(value[0], str)):
            # Fixed sets (the kept channels, the KDE bandwidth floors) are shown, not edited.
            text = ", ".join(f"{k} = {v}" for k, v in value.items()) if isinstance(value, Mapping) else f"{len(value)} channels"
            editor = QLabel(text)
            editor.setWordWrap(True)
            if not isinstance(value, Mapping):
                editor.setToolTip(", ".join(value))
        else:
            editor = QLineEdit(self._to_text(value))
            if "None" in annotation:
                editor.setPlaceholderText("none")
            editor.editingFinished.connect(self._on_edit)
        editor.setObjectName(name)
        return editor

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, tuple):
            return ", ".join(repr(item) for item in value)
        # repr keeps every digit: a spin box would round 240/2048 and change the calibration.
        return repr(value) if isinstance(value, float) else str(value)

    def _read(self, name: str) -> Any:
        editor, current = self._editors[name], getattr(self._params, name)
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QSpinBox):
            return editor.value()
        if isinstance(editor, QLabel):
            return current
        text = editor.text().strip()
        annotation = str(next(f.type for f in dataclasses.fields(self._params) if f.name == name))
        if isinstance(current, tuple):
            return tuple(_number(part) for part in text.split(",") if part.strip())
        if "None" in annotation and text.lower() in ("", "none"):
            return None
        if "int" in annotation:
            return int(text)
        if "float" in annotation:
            return float(text)
        return text

    def value(self) -> Any:
        """The edited parameters; raises ``ValueError`` if an entry is not valid."""
        values = {name: self._read(name) for name in self._editors}
        return dataclasses.replace(self._params, **values)

    def set_value(self, params: Any) -> None:
        """Show another parameter set without emitting ``changed``."""
        self._params = params
        for name, editor in self._editors.items():
            value = getattr(params, name)
            editor.blockSignals(True)
            if isinstance(editor, QCheckBox):
                editor.setChecked(value)
            elif isinstance(editor, QSpinBox):
                editor.setValue(value)
            elif isinstance(editor, QLineEdit):
                editor.setText(self._to_text(value))
            editor.blockSignals(False)
        self._error.setText("")

    def _on_edit(self, *_args) -> None:
        try:
            params = self.value()
        except (ValueError, TypeError) as error:
            self._error.setText(str(error))
            return
        self._error.setText("")
        self._params = params
        self.changed.emit(params)


class FigurePanel(QWidget):
    """A list of named figures, drawn one at a time on a canvas with pan and zoom.

    A figure is only built when it is selected, on the GUI thread.
    """

    figure_shown = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._builders: dict[str, Callable[[], Figure]] = {}
        self.figure: Figure | None = None
        self.canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None

        self.selector = QComboBox()
        self.selector.currentTextChanged.connect(self._show)
        self.export_button = QPushButton("Export figure…")
        self.export_button.clicked.connect(self.export_current)
        top = QHBoxLayout()
        top.addWidget(QLabel("Figure"))
        top.addWidget(self.selector, 1)
        top.addWidget(self.export_button)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(top)
        self.set_figure(style.message_figure("No figure yet."))

    def set_builders(self, builders: Mapping[str, Callable[[], Figure]], message: str = "No figure yet.") -> None:
        """Replace the list of figures, keeping the current selection if it still exists."""
        previous = self.selector.currentText()
        self._builders = dict(builders)
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems(list(self._builders))
        if previous in self._builders:
            self.selector.setCurrentText(previous)
        self.selector.blockSignals(False)
        self.selector.setEnabled(bool(self._builders))
        if self._builders:
            self._show(self.selector.currentText())
        else:
            self.set_figure(style.message_figure(message))

    def _show(self, name: str) -> None:
        if name not in self._builders:
            return
        try:
            figure = self._builders[name]()
        except Exception as error:   # a figure must never take the window down
            figure = style.message_figure(f"Could not draw '{name}':\n{error}")
        self.set_figure(figure)
        self.figure_shown.emit(name)

    def set_figure(self, figure: Figure) -> None:
        """Show a figure on a fresh canvas."""
        if self.canvas is not None:
            for old in (self._toolbar, self.canvas):
                self._layout.removeWidget(old)
                old.hide()
                old.setParent(None)   # gone now, not at the next event loop turn
                old.deleteLater()
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self.canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def export_current(self) -> None:
        if self.figure is None:
            return
        name = self.selector.currentText() or "figure"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", f"{name}.png", "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300)
        except Exception as error:
            QMessageBox.warning(self, "Export failed", str(error))


class DataFrameModel(QAbstractTableModel):
    """Read-only view of the first rows of a table."""

    def __init__(self, table: pd.DataFrame | None = None):
        super().__init__()
        self._table = pd.DataFrame() if table is None else table.head(PREVIEW_ROWS)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._table)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._table.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._table.iat[index.row(), index.column()]
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.6g}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self._table.columns[section]) if orientation == Qt.Horizontal else str(section)


class TablePreview(QWidget):
    """A table view with a caption giving the full size of the table."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.caption = QLabel("")
        self.view = QTableView()
        self.view.setAlternatingRowColors(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.caption)
        layout.addWidget(self.view, 1)
        self.set_table(None)

    def set_table(self, table: pd.DataFrame | None, name: str = "") -> None:
        self._model = DataFrameModel(table)
        self.view.setModel(self._model)
        if table is None:
            self.caption.setText("No table yet.")
        else:
            shown = min(len(table), PREVIEW_ROWS)
            suffix = f", first {shown} shown" if shown < len(table) else ""
            self.caption.setText(f"{name}: {len(table)} rows × {len(table.columns)} columns{suffix}")
