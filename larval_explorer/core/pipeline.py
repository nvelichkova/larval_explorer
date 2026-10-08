"""Run the stages for one recording: order, persistence, resume, stale state.

``stages.py`` computes; this module owns everything about disk (D-007): the
output tree, reading and writing CSVs, and the manifest.

Every table a stage receives has been written to CSV and parsed back exactly
once, whether it was computed a moment ago or loaded from an earlier run.
That is how the upstream scripts hand data on, and it is what makes results
identical to upstream and independent of whether a run was resumed (D-018).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from larval_explorer.core import manifest as M
from larval_explorer.core import schema as S
from larval_explorer.core import stages
from larval_explorer.core.params import PipelineParams, params_to_dict
from larval_explorer.core.result import StageResult

STATUS_MISSING = "missing"
STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_FAILED = "failed"

FIGURES_FOLDER = "figures"


class PipelineError(RuntimeError):
    """A stage failed, or a stage was asked for that cannot be run."""

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message)
        self.stage = stage


class PipelineCancelled(RuntimeError):
    """The run was cancelled between stages. The output tree is consistent."""


@dataclass(frozen=True)
class HmmPool:
    """The recordings whose steps were fitted together, and a hash of what was fitted."""

    recording_ids: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class Output:
    """One table a stage writes: its name in the StageResult, its file, its schema."""

    table: str
    filename: str
    schema: S.TableSchema | None = None
    optional: bool = False


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    folder: str
    depends_on: tuple[str, ...]
    outputs: tuple[Output, ...]
    run: Callable[["RecordingPipeline", PipelineParams], StageResult]


def _run_00(pipeline, params):
    return stages.stage_00(pipeline.read_raw(), params.stage_00)


def _run_01(pipeline, params):
    return stages.stage_01(pipeline.table("00", "wide"), params.stage_01)


def _run_02(pipeline, params):
    return stages.stage_02(
        pipeline.table("01", "calibrated"), params.stage_02, params.stage_01, recording_id=pipeline.recording_id
    )


def _run_03(pipeline, params):
    return stages.stage_03(pipeline.table("02", "trajectories"), params.stage_03)


def _run_04(pipeline, params):
    return stages.stage_04(pipeline.table("03", "smoothed"), params.stage_04)


def _run_05(pipeline, params):
    return stages.stage_05(pipeline.table("03", "smoothed"), pipeline.table("04", "body_length"), params.stage_05)


def _run_06(pipeline, params):
    if pipeline.pool is None:
        return stages.stage_06(pipeline.table("05", "steps"), params.stage_06)
    # A pooled fit needs every recording in the pool; it is supplied by the analysis that made it.
    if pipeline._pooled_result is None:
        raise PipelineError(
            f"Stage 06 of '{pipeline.recording_id}' comes from an HMM fit pooled over "
            f"{len(pipeline.pool.recording_ids)} recordings. Re-fit the pool from the Aggregate tab, "
            "or reset stage 06 to return this recording to a fit of its own.",
            stage="06",
        )
    return pipeline._pooled_result


def _run_07(pipeline, params):
    return stages.stage_07(pipeline.table("06", "steps"), params.stage_07)


def _run_08(pipeline, params):
    return stages.stage_08(pipeline.table("07", "steps"), params.stage_08)


def _run_09(pipeline, params):
    return stages.stage_09(pipeline.table("03", "smoothed"), params.stage_09)


def _run_10(pipeline, params):
    return stages.stage_10(pipeline.table("03", "smoothed"), pipeline.table("09", "events"), params.stage_10)


def _run_11(pipeline, params):
    return stages.stage_11(
        pipeline.table("10", "event_level_steps"), pipeline.table("08", "hmm_segments"), params.stage_11
    )


# File names are upstream's wherever upstream has the table.
STAGES: tuple[StageSpec, ...] = (
    StageSpec("00", "Reorganise", "00_reorganized", (), (
        Output("wide", "merged_tracking_table_unscaled.csv", S.RAW_WIDE),
    ), _run_00),
    StageSpec("01", "Calibrate", "01_calibrated", ("00",), (
        Output("calibrated", "trajectory_timeseries_calibrated.csv", S.TRAJECTORY),
    ), _run_01),
    StageSpec("02", "Track segmentation", "02_segments", ("01",), (
        Output("trajectories", "track_segment_trajectories.csv", S.TRAJECTORY),
        Output("track_segments", "track_segments.csv", optional=True),
        Output("larva_summary", "larva_summary.csv", optional=True),
    ), _run_02),
    StageSpec("03", "Smooth", "03_smoothed", ("02",), (
        Output("smoothed", "trajectory_timeseries.csv", S.TRAJECTORY),
    ), _run_03),
    StageSpec("04", "Body length", "04_body_length", ("03",), (
        Output("body_length", "mean_body_length_by_trajectory.csv", S.BODY_LENGTH),
    ), _run_04),
    StageSpec("05", "RDP steps", "05_rdp_steps", ("03", "04"), (
        Output("steps", "rdp_steps.csv", S.RDP_STEPS),
    ), _run_05),
    StageSpec("06", "HMM", "06_hmm", ("05",), (
        Output("steps", "rdp_steps_with_hmm_states.csv", S.HMM_STEPS),
    ), _run_06),
    StageSpec("07", "Filter HMM runs", "07_hmm_filtered", ("06",), (
        Output("steps", "rdp_steps_with_filtered_hmm_states.csv", S.HMM_STEPS_FILTERED),
    ), _run_07),
    StageSpec("08", "HMM segments", "08_segments", ("07",), (
        Output("hmm_segments", "representative_hmm_segments.csv", S.HMM_SEGMENTS),
    ), _run_08),
    StageSpec("09", "Events", "09_events", ("03",), (
        Output("events", "event_intervals.csv", S.EVENTS),
        Output("signals", "event_signals.csv"),
        Output("thresholds", "event_thresholds.csv"),
    ), _run_09),
    StageSpec("10", "Step models", "10_step_models", ("03", "09"), (
        Output("run_anchor_steps", "run_anchor_steps.csv", S.RUN_ANCHOR_STEPS),
        Output("event_level_steps", "event_level_steps.csv", S.EVENT_LEVEL_STEPS),
    ), _run_10),
    StageSpec("11", "Steering", "11_steering", ("08", "10"), (
        Output("run_fits", "run_fits.csv", S.RUN_FITS),
        Output("segment_fits", "segment_fits.csv", S.SEGMENT_FITS),
        Output("headcast_segment_metrics", "headcast_segment_metrics.csv", S.HEADCAST_SEGMENT_METRICS),
        Output("autocorr_by_state", "autocorr_by_state.csv"),
        Output("state_param_summary", "state_param_summary.csv"),
        Output("simulation_parameters", "simulation_parameters.csv"),
        Output("headcast_directions", "headcast_directions.csv"),
        Output("headcast_thetas", "headcast_thetas.csv"),
        Output("crawl_lengths", "crawl_lengths.csv"),
    ), _run_11),
)

STAGE_KEYS: tuple[str, ...] = tuple(spec.key for spec in STAGES)
_SPEC = {spec.key: spec for spec in STAGES}


def spec(stage: str) -> StageSpec:
    if stage not in _SPEC:
        raise PipelineError(f"Unknown stage '{stage}'. Stages are: {', '.join(STAGE_KEYS)}")
    return _SPEC[stage]


def with_dependencies(targets: list[str] | tuple[str, ...]) -> list[str]:
    """The target stages and everything they depend on, in running order."""
    needed: set[str] = set()

    def visit(stage: str) -> None:
        if stage in needed:
            return
        for dependency in spec(stage).depends_on:
            visit(dependency)
        needed.add(stage)

    for target in targets:
        visit(target)
    return [key for key in STAGE_KEYS if key in needed]


def dependents(stage: str) -> list[str]:
    """Every stage downstream of ``stage``, in running order."""
    result: list[str] = []
    for key in STAGE_KEYS:
        if any(dependency == stage or dependency in result for dependency in spec(key).depends_on):
            result.append(key)
    return result


def read_fimtrack_csv(path: Path) -> pd.DataFrame:
    """A raw FIM-Track export: row labels ``<channel>(<frame>)``, one column per larva."""
    return pd.read_csv(path, index_col=0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RecordingPipeline:
    """The stages of one recording, backed by ``<output_root>/<recording_id>/``."""

    def __init__(
        self,
        recording_id: str,
        source_file: Path,
        output_root: Path,
        params: PipelineParams | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.recording_id = recording_id
        self.source_file = Path(source_file)
        self.folder = Path(output_root) / recording_id
        self.params = params or PipelineParams()
        self.metadata = dict(metadata or {})
        self._tables: dict[tuple[str, str], pd.DataFrame] = {}
        self._source_hash: str | None = None
        self.manifest = M.load_manifest(self.folder) or M.new_manifest(recording_id, self.source_file, self.metadata)
        if self.manifest["recording_id"] != recording_id:
            raise PipelineError(
                f"{self.folder} holds results for '{self.manifest['recording_id']}', not '{recording_id}'."
            )
        if metadata is not None:
            self.manifest["metadata"] = self.metadata
        pool_fingerprint = self.manifest.get("hmm_pool_fingerprint")
        self.pool: HmmPool | None = (
            HmmPool(tuple(self.manifest.get("hmm_pool", [])), pool_fingerprint) if pool_fingerprint else None
        )
        self._pooled_result: StageResult | None = None

    # ── parameters and state ────────────────────────────────────────────────

    def set_params(self, params: PipelineParams) -> None:
        """Change parameters. Stages whose inputs changed become stale; nothing is deleted."""
        self.params = params

    def saved_params(self) -> PipelineParams:
        """The parameters recorded in the manifest, defaults for stages never run."""
        recorded = {key: record["params"] for key, record in self.manifest["stages"].items()}
        return PipelineParams.from_dict(recorded)

    def source_hash(self) -> str:
        if self._source_hash is None:
            if not self.source_file.is_file():
                raise PipelineError(f"Recording file not found: {self.source_file}", stage="00")
            self._source_hash = file_sha256(self.source_file)
        return self._source_hash

    def fingerprint(self, stage: str) -> str:
        """Hash of everything a stage's result depends on.

        The stage's own parameters, the fingerprints of the stages it reads,
        and at the root the content of the raw file.
        """
        stage_spec = spec(stage)
        payload: dict[str, Any] = {
            "stage": stage,
            "params": params_to_dict(getattr(self.params, f"stage_{stage}")),
            "depends_on": [self.fingerprint(dependency) for dependency in stage_spec.depends_on],
        }
        if stage == "00":
            payload["source_sha256"] = self.source_hash()
        if stage == "02":
            payload["recording_id"] = self.recording_id  # segment IDs embed it
        if stage == "06" and self.pool is not None:
            payload["hmm_pool"] = self.pool.fingerprint
        return hashlib.sha256(json.dumps(M.to_json(payload), sort_keys=True).encode("utf-8")).hexdigest()

    def status(self, stage: str) -> str:
        """``missing``, ``ok``, ``stale`` (parameters or inputs changed since) or ``failed``."""
        record = self.manifest["stages"].get(stage)
        if record is None:
            return STATUS_MISSING
        if record["status"] == M.STATUS_FAILED:
            return STATUS_FAILED
        folder = self.stage_folder(stage)
        if not all((folder / name).is_file() for name in record["outputs"]):
            return STATUS_MISSING
        return STATUS_OK if record["fingerprint"] == self.fingerprint(stage) else STATUS_STALE

    def statuses(self) -> dict[str, str]:
        return {key: self.status(key) for key in STAGE_KEYS}

    # ── files ───────────────────────────────────────────────────────────────

    def stage_folder(self, stage: str) -> Path:
        return self.folder / spec(stage).folder

    def figure_folder(self, stage: str) -> Path:
        return self.folder / FIGURES_FOLDER / spec(stage).folder

    def read_raw(self) -> pd.DataFrame:
        return read_fimtrack_csv(self.source_file)

    def table(self, stage: str, name: str) -> pd.DataFrame:
        """A stage's saved table in canonical names, as parsed from its CSV.

        Raises if the stage has no current result; stale tables can be read
        with ``allow_stale`` through ``load_table``.
        """
        return self.load_table(stage, name, allow_stale=False)

    def load_table(self, stage: str, name: str, allow_stale: bool = True) -> pd.DataFrame:
        status = self.status(stage)
        if status in (STATUS_MISSING, STATUS_FAILED) or (status == STATUS_STALE and not allow_stale):
            raise PipelineError(f"Stage {stage} has no usable result (status: {status}).", stage=stage)
        key = (stage, name)
        if key not in self._tables:
            output = self._output(stage, name)
            path = self.stage_folder(stage) / output.filename
            if not path.is_file():
                raise PipelineError(f"Stage {stage} did not produce table '{name}'.", stage=stage)
            self._tables[key] = _read_table(path, output.schema)
        return self._tables[key]

    def _output(self, stage: str, name: str) -> Output:
        for output in spec(stage).outputs:
            if output.table == name:
                return output
        raise PipelineError(f"Stage {stage} has no table '{name}'.", stage=stage)

    # ── running ─────────────────────────────────────────────────────────────

    def run(
        self,
        targets: list[str] | tuple[str, ...] | None = None,
        *,
        force: bool | list[str] | tuple[str, ...] = False,
        progress: Callable[[str, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, str]:
        """Bring the target stages (default: all) up to date.

        Dependencies run first. A stage whose result is current is skipped
        unless forced: ``force=True`` re-runs every stage visited, a list of
        stage keys re-runs just those. ``progress(stage, event)`` is called with events
        ``"skipped"``, ``"started"``, ``"finished"`` and ``"failed"``.
        Cancellation is checked between stages. Returns what happened to each
        stage visited.
        """
        order = with_dependencies(list(targets) if targets else list(STAGE_KEYS))
        forced = set(order) if force is True else set(force or ())
        outcome: dict[str, str] = {}
        for stage in order:
            if should_cancel is not None and should_cancel():
                raise PipelineCancelled(f"Cancelled before stage {stage}.")
            if stage not in forced and self.status(stage) == STATUS_OK:
                outcome[stage] = "skipped"
                if progress:
                    progress(stage, "skipped")
                continue
            if progress:
                progress(stage, "started")
            try:
                self.run_stage(stage)
            except PipelineError:
                outcome[stage] = "failed"
                if progress:
                    progress(stage, "failed")
                raise
            outcome[stage] = "finished"
            if progress:
                progress(stage, "finished")
        return outcome

    def run_stage(self, stage: str) -> StageResult:
        """Run one stage now. Its dependencies must be current."""
        stage_spec = spec(stage)
        for dependency in stage_spec.depends_on:
            if self.status(dependency) != STATUS_OK:
                raise PipelineError(
                    f"Stage {stage} needs stage {dependency}, which is {self.status(dependency)}.", stage=stage
                )
        params_dict = params_to_dict(getattr(self.params, f"stage_{stage}"))
        fingerprint = self.fingerprint(stage)

        # Forget the old result first, so an interruption leaves "missing", never a mixture.
        self.manifest["stages"].pop(stage, None)
        self._forget_tables(stage)
        M.save_manifest(self.folder, self.manifest)
        folder = self.stage_folder(stage)
        if folder.exists():
            shutil.rmtree(folder)

        started = time.perf_counter()
        try:
            result = stage_spec.run(self, self.params)
            written = self._write_tables(stage_spec, result)
        except Exception as error:
            self.manifest["stages"][stage] = M.stage_record(
                status=M.STATUS_FAILED,
                params=params_dict,
                fingerprint=fingerprint,
                duration_s=time.perf_counter() - started,
                error=traceback.format_exc(),
            )
            M.save_manifest(self.folder, self.manifest)
            raise PipelineError(f"Stage {stage} ({stage_spec.label}) failed: {error}", stage=stage) from error

        self.manifest["stages"][stage] = M.stage_record(
            status=M.STATUS_OK,
            params=params_dict,
            fingerprint=fingerprint,
            diagnostics=result.diagnostics,
            warnings=result.warnings,
            duration_s=time.perf_counter() - started,
            outputs=written,
        )
        if stage == "06":
            # Which recordings defined the states: this one alone, or a pool (D-002).
            self.manifest["hmm_pool"] = list(self.pool.recording_ids) if self.pool else [self.recording_id]
            self.manifest["hmm_pool_fingerprint"] = self.pool.fingerprint if self.pool else None
            self.manifest["hmm_state_mean_phi"] = result.diagnostics.get("state_mean_phi", {})
        self.manifest["environment"] = M.environment()
        M.save_manifest(self.folder, self.manifest)
        return result

    def _write_tables(self, stage_spec: StageSpec, result: StageResult) -> list[str]:
        folder = self.stage_folder(stage_spec.key)
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        for output in stage_spec.outputs:
            if output.table not in result.tables:
                if output.optional:
                    continue
                raise PipelineError(
                    f"Stage {stage_spec.key} did not return table '{output.table}'.", stage=stage_spec.key
                )
            table = result.tables[output.table]
            if output.schema is not None:
                # An empty table may lack columns that only rows would have created.
                table = output.schema.to_upstream(table, validate=len(table) > 0)
            table.to_csv(folder / output.filename, index=False)
            written.append(output.filename)
        return written

    def save_figures(
        self, stage: str, figures: dict, formats: tuple[str, ...] = ("png",), dpi: int = 150
    ) -> list[Path]:
        """Write a stage's figures to ``figures/<stage>/``, replacing what was there.

        ``figures`` maps a name to a ``matplotlib.figure.Figure``. Safe to call
        from a worker thread: each figure gets its own Agg canvas and pyplot is
        not involved.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        folder = self.figure_folder(stage)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        written = []
        for name, figure in figures.items():
            FigureCanvasAgg(figure)
            for extension in formats:
                path = folder / f"{_safe_file_name(name)}.{extension}"
                figure.savefig(path, dpi=dpi)
                written.append(path)
        return written

    def _forget_tables(self, stage: str) -> None:
        for key in [key for key in self._tables if key[0] == stage]:
            del self._tables[key]

    def install_pooled_hmm(self, pool: HmmPool, result: StageResult) -> None:
        """Record this recording's share of a pooled HMM fit as its stage 06."""
        self.pool = pool
        self._pooled_result = result
        try:
            self.run_stage("06")
        finally:
            self._pooled_result = None

    def leave_pool(self) -> None:
        """Return to a fit of this recording alone. Stage 06 becomes stale until re-run."""
        self.pool = None
        self.manifest["hmm_pool_fingerprint"] = None
        self.manifest["hmm_pool"] = [self.recording_id] if "06" in self.manifest["stages"] else []
        M.save_manifest(self.folder, self.manifest)

    def reset(self, stage: str) -> None:
        """Delete a stage's result and those of every stage downstream of it.

        Deleting stage 06 also takes the recording out of any pooled fit.
        """
        if stage == "06" or "06" in dependents(stage):
            self.pool = None
            self.manifest["hmm_pool_fingerprint"] = None
            self.manifest["hmm_pool"] = []
        for key in [stage, *dependents(stage)]:
            self.manifest["stages"].pop(key, None)
            self._forget_tables(key)
            folder = self.stage_folder(key)
            if folder.exists():
                shutil.rmtree(folder)
        M.save_manifest(self.folder, self.manifest)


def _safe_file_name(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in name)


def _read_table(path: Path, table_schema: S.TableSchema | None) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if table_schema is not None:
        table = table_schema.to_canonical(table, validate=len(table) > 0)
    return table
