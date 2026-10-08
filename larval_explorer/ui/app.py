import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from larval_explorer.ui.main_window import MainWindow


def main() -> int:
    """Launch the GUI. An optional argument is a project folder to open."""
    app = QApplication(sys.argv)
    window = MainWindow()
    if len(sys.argv) > 1:
        window.open_project(Path(sys.argv[1]))
    window.show()
    return app.exec_()
