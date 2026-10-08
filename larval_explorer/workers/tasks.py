"""Background tasks. Wrappers only: the work itself is in ``core``.

A worker never touches a widget; it talks to the GUI thread through signals.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from larval_explorer.core.pipeline import PipelineCancelled, PipelineError, RecordingPipeline


class FunctionSignals(QObject):
    finished = pyqtSignal(object)   # the function's return value
    failed = pyqtSignal(str)        # the traceback


class FunctionWorker(QRunnable):
    """Run ``function(*args, **kwargs)`` off the GUI thread."""

    def __init__(self, function: Callable, *args, **kwargs):
        super().__init__()
        self.function, self.args, self.kwargs = function, args, kwargs
        self.signals = FunctionSignals()
        self.setAutoDelete(False)   # the window keeps the reference until the result is delivered

    def run(self) -> None:
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(result)


class PipelineSignals(QObject):
    progress = pyqtSignal(str, str, dict)   # stage, event, statuses of all stages
    finished = pyqtSignal(dict, dict)       # outcome per stage visited, final statuses
    failed = pyqtSignal(str, str, dict)     # stage (or ""), message, final statuses
    cancelled = pyqtSignal(dict)            # final statuses


class PipelineWorker(QRunnable):
    """Bring stages of one recording up to date, reporting as it goes.

    While it runs, the pipeline object belongs to this thread; the GUI reads
    stage statuses from the signals instead of from the pipeline.
    """

    def __init__(self, pipeline: RecordingPipeline, targets: list[str] | None = None, force: bool | list[str] = False):
        super().__init__()
        self.pipeline, self.targets, self.force = pipeline, targets, force
        self.signals = PipelineSignals()
        self._cancel = threading.Event()
        self.setAutoDelete(False)   # the window keeps the reference until the result is delivered

    def cancel(self) -> None:
        """Ask the run to stop before its next stage."""
        self._cancel.set()

    def run(self) -> None:
        pipeline = self.pipeline
        try:
            outcome = pipeline.run(
                self.targets,
                force=self.force,
                progress=lambda stage, event: self.signals.progress.emit(stage, event, pipeline.statuses()),
                should_cancel=self._cancel.is_set,
            )
        except PipelineCancelled:
            self.signals.cancelled.emit(pipeline.statuses())
        except PipelineError as error:
            self.signals.failed.emit(error.stage or "", str(error), pipeline.statuses())
        except Exception:
            self.signals.failed.emit("", traceback.format_exc(), pipeline.statuses())
        else:
            self.signals.finished.emit(outcome, pipeline.statuses())


class BatchSignals(QObject):
    progress = pyqtSignal(str, str, str)   # recording, stage, event
    finished = pyqtSignal(object)          # RecordingOutcome


class BatchRecordingWorker(QRunnable):
    """One recording of a batch. Several of these run side by side, one recording each."""

    def __init__(self, project, recording_id: str, options, cancel_event: threading.Event):
        super().__init__()
        self.project, self.recording_id, self.options, self.cancel_event = project, recording_id, options, cancel_event
        self.signals = BatchSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        from larval_explorer.core.batch import run_recording

        outcome = run_recording(
            self.project, self.recording_id, self.options,
            progress=self.signals.progress.emit, should_cancel=self.cancel_event.is_set,
        )
        self.signals.finished.emit(outcome)
