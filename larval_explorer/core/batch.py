"""Run many recordings unattended.

Each recording is processed on its own, end to end, exactly as it would be
alone: nothing is combined across recordings here (D-010). A recording that
fails is recorded and the rest carry on. Every batch writes a run log.

``run_recording`` does one recording and never raises. ``run_batch`` runs
several on a thread pool for console use; the GUI runs the same
``run_recording`` on its own workers.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from larval_explorer.core import manifest as M
from larval_explorer.core import pipeline as PL
from larval_explorer.core.params import PipelineParams
from larval_explorer.core.project import Project

RESULT_OK = "ok"                    # at least one stage ran, all succeeded
RESULT_UP_TO_DATE = "up to date"    # nothing needed running
RESULT_FAILED = "failed"
RESULT_CANCELLED = "cancelled"

LOG_FOLDER = "run_logs"
DEFAULT_WORKERS = 2

# matplotlib shares font and text-layout caches between figures, so figures
# are drawn one at a time even when recordings run side by side.
_FIGURE_LOCK = threading.Lock()


@dataclass(frozen=True)
class BatchOptions:
    """What a batch does to every recording in it."""

    through_stage: str = PL.STAGE_KEYS[-1]
    # Re-run stages even if their results are current.
    force: bool = False
    # None: each recording keeps the parameters it was last run with (defaults
    # if never run). Otherwise every recording is run with these.
    params: PipelineParams | None = None
    # Stage 09 alone draws five figures per track, so figures are opt-in.
    export_figures: bool = False
    figure_formats: tuple[str, ...] = ("png",)

    def __post_init__(self) -> None:
        PL.spec(self.through_stage)


@dataclass
class RecordingOutcome:
    recording_id: str
    result: str
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    message: str = ""
    traceback: str = ""
    figures_written: int = 0
    duration_s: float = 0.0


def run_recording(
    project: Project,
    recording_id: str,
    options: BatchOptions,
    progress: Callable[[str, str, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RecordingOutcome:
    """Process one recording. Failures are returned, never raised.

    ``progress(recording_id, stage, event)`` reports each stage as the
    pipeline does, plus ``("figures", "started")`` when figures are written.
    """
    started = time.perf_counter()
    outcome = RecordingOutcome(recording_id, RESULT_CANCELLED)
    if should_cancel is not None and should_cancel():
        outcome.message = "Cancelled before it started."
        return outcome
    try:
        pipeline = project.pipeline(recording_id, options.params)
        stage_outcomes: dict[str, str] = {}

        def on_progress(stage: str, event: str) -> None:
            stage_outcomes[stage] = event
            if progress is not None:
                progress(recording_id, stage, event)

        try:
            pipeline.run([options.through_stage], force=options.force, progress=on_progress, should_cancel=should_cancel)
            outcome.result = RESULT_OK
        except PL.PipelineCancelled as cancelled:
            outcome.message = str(cancelled)
        except PL.PipelineError as error:
            outcome.result = RESULT_FAILED
            outcome.failed_stage = error.stage
            outcome.message = str(error)
            record = pipeline.manifest["stages"].get(error.stage or "", {})
            outcome.traceback = record.get("error", "") or traceback.format_exc()

        outcome.stages_run = [stage for stage, event in stage_outcomes.items() if event == "finished"]
        outcome.stages_skipped = [stage for stage, event in stage_outcomes.items() if event == "skipped"]
        if outcome.result == RESULT_OK and not outcome.stages_run:
            outcome.result = RESULT_UP_TO_DATE
        # Metadata may have been edited since the last run; keep the manifest in step.
        M.save_manifest(pipeline.folder, pipeline.manifest)

        if options.export_figures and outcome.result in (RESULT_OK, RESULT_UP_TO_DATE):
            if progress is not None:
                progress(recording_id, "figures", "started")
            outcome.figures_written = _export_figures(pipeline, options)
    except Exception as error:   # anything unforeseen still must not stop the batch
        outcome.result = RESULT_FAILED
        outcome.message = f"{type(error).__name__}: {error}"
        outcome.traceback = traceback.format_exc()
    outcome.duration_s = round(time.perf_counter() - started, 3)
    return outcome


def _export_figures(pipeline: PL.RecordingPipeline, options: BatchOptions) -> int:
    from larval_explorer.plots import catalog   # only needed when figures are asked for

    written = 0
    for stage in PL.with_dependencies([options.through_stage]):
        if stage not in catalog.STAGES_WITH_FIGURES or pipeline.status(stage) != PL.STATUS_OK:
            continue
        with _FIGURE_LOCK:
            figures = catalog.stage_figures(pipeline, stage)
            written += len(pipeline.save_figures(stage, figures, formats=options.figure_formats))
    return written


class BatchLog:
    """The run log of one batch: ``<output_root>/run_logs/batch_<time>.json``.

    Rewritten after every recording, so a batch that is killed still leaves a
    log of what finished.
    """

    def __init__(self, output_root: Path, recording_ids: list[str], options: BatchOptions, workers: int):
        self._lock = threading.Lock()
        started = datetime.datetime.now().astimezone()
        folder = Path(output_root) / LOG_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"batch_{started.strftime('%Y%m%d_%H%M%S_%f')}.json"
        self.data = {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": None,
            "recordings_requested": list(recording_ids),
            "options": {
                "through_stage": options.through_stage,
                "force": options.force,
                "export_figures": options.export_figures,
                "figure_formats": list(options.figure_formats),
                "parameters": "same for every recording" if options.params is not None else "each recording's own",
                "params": options.params.to_dict() if options.params is not None else None,
            },
            "workers": workers,
            "environment": M.environment(),
            "summary": {},
            "recordings": [],
        }
        self._write()

    def record(self, outcome: RecordingOutcome) -> None:
        with self._lock:
            self.data["recordings"].append(asdict(outcome))
            self._write()

    def finish(self) -> dict:
        with self._lock:
            results = [entry["result"] for entry in self.data["recordings"]]
            self.data["summary"] = {result: results.count(result) for result in sorted(set(results))}
            self.data["finished_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            self._write()
            return self.data

    def _write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(M.to_json(self.data), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)


def run_batch(
    project: Project,
    recording_ids: list[str] | None = None,
    options: BatchOptions | None = None,
    max_workers: int = DEFAULT_WORKERS,
    progress: Callable[[str, str, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[RecordingOutcome], Path]:
    """Process recordings side by side. Returns the outcomes, in the order asked, and the log path."""
    options = options or BatchOptions()
    recording_ids = list(recording_ids) if recording_ids is not None else project.recording_ids()
    log = BatchLog(project.output_root, recording_ids, options, max_workers)

    def one(recording_id: str) -> RecordingOutcome:
        outcome = run_recording(project, recording_id, options, progress, should_cancel)
        log.record(outcome)
        return outcome

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        outcomes = list(pool.map(one, recording_ids))
    log.finish()
    return outcomes, log.path
