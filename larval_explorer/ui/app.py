import sys
import traceback
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox

from larval_explorer.ui.main_window import MainWindow


def install_error_dialog(window: MainWindow) -> None:
    """Show unexpected errors instead of letting them close the application.

    PyQt ends the program when an exception escapes a button handler. Results
    on disk are safe either way, but the window should stay open and say what
    went wrong.
    """
    def show(error_type, error, trace) -> None:
        details = "".join(traceback.format_exception(error_type, error, trace))
        sys.stderr.write(details)
        box = QMessageBox(QMessageBox.Critical, "Unexpected error",
                          f"{error_type.__name__}: {error}\n\nThe window stays open; your results on disk are not affected.",
                          parent=window)
        box.setDetailedText(details)
        box.exec_()

    sys.excepthook = show


def main() -> int:
    """Launch the GUI. An optional argument is a project folder to open."""
    app = QApplication(sys.argv)
    window = MainWindow()
    install_error_dialog(window)
    if len(sys.argv) > 1:
        window.open_project(Path(sys.argv[1]))
    window.show()
    return app.exec_()
