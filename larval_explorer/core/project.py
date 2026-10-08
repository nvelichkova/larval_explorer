"""A project: a folder holding the dataset registry and the project settings.

    <project folder>/
        project.json            output root, filename pattern
        dataset_registry.csv    one row per recording (registry.py)

Results live under the output root, one folder per recording (pipeline.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from larval_explorer.core import registry as R
from larval_explorer.core.params import PipelineParams
from larval_explorer.core.manifest import STATUS_FAILED, load_manifest
from larval_explorer.core.pipeline import STAGE_KEYS, RecordingPipeline

PROJECT_FILE = "project.json"
DEFAULT_OUTPUT_FOLDER = "outputs"
STATE_COMPLETE, STATE_PARTIAL, STATE_FAILED, STATE_NOT_RUN = "complete", "partial", "failed", "not run"
EXAMPLE_FILENAME_PATTERN = r"(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)"


class ProjectError(ValueError):
    """The folder is not a project, or a project cannot be created there."""


@dataclass
class Project:
    folder: Path
    output_root: Path
    filename_pattern: str = ""
    registry: pd.DataFrame = field(default_factory=R.empty_registry)

    @classmethod
    def create(cls, folder: Path) -> "Project":
        folder = Path(folder)
        if (folder / PROJECT_FILE).exists():
            raise ProjectError(f"{folder} already contains a project; open it instead.")
        folder.mkdir(parents=True, exist_ok=True)
        project = cls(folder=folder, output_root=folder / DEFAULT_OUTPUT_FOLDER)
        project.save()
        return project

    @classmethod
    def open(cls, folder: Path) -> "Project":
        folder = Path(folder)
        settings_path = folder / PROJECT_FILE
        if not settings_path.is_file():
            raise ProjectError(f"{folder} is not a Larval Explorer project (no {PROJECT_FILE}).")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        registry_path = folder / R.REGISTRY_NAME
        registry = R.load_registry(registry_path) if registry_path.is_file() else R.empty_registry()
        return cls(
            folder=folder,
            output_root=Path(settings.get("output_root", folder / DEFAULT_OUTPUT_FOLDER)),
            filename_pattern=settings.get("filename_pattern", ""),
            registry=registry,
        )

    def save(self) -> None:
        settings = {"output_root": Path(self.output_root).as_posix(), "filename_pattern": self.filename_pattern}
        (self.folder / PROJECT_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
        R.save_registry(self.registry, self.folder / R.REGISTRY_NAME)

    def recording_ids(self) -> list[str]:
        return self.registry[R.RECORDING_ID].tolist()

    def recording_state(self, recording_id: str) -> str:
        """How far a recording has got, read from its manifest alone.

        ``complete`` (every stage has a result), ``failed`` (a stage failed),
        ``partial`` or ``not run``. Cheap enough to show for every recording;
        it does not tell whether a result is stale, which needs the raw file.
        """
        try:
            manifest = load_manifest(Path(self.output_root) / recording_id)
        except (ValueError, OSError):
            return STATE_NOT_RUN
        if manifest is None or not manifest["stages"]:
            return STATE_NOT_RUN
        statuses = [record["status"] for record in manifest["stages"].values()]
        if STATUS_FAILED in statuses:
            return STATE_FAILED
        return STATE_COMPLETE if len(statuses) == len(STAGE_KEYS) else STATE_PARTIAL

    def pipeline(self, recording_id: str, params: PipelineParams | None = None) -> RecordingPipeline:
        """The pipeline of one recording.

        Without ``params``, a recording that has been run before comes back
        with the parameters it was run with, so its stages show as current.
        """
        row = self.registry[self.registry[R.RECORDING_ID] == recording_id]
        if row.empty:
            raise ProjectError(f"No recording '{recording_id}' in the registry.")
        source = Path(str(row.iloc[0][R.PATH]))
        pipeline = RecordingPipeline(
            recording_id, source, self.output_root, params, metadata=R.metadata_for(self.registry, recording_id)
        )
        if params is None:
            pipeline.set_params(pipeline.saved_params())
        return pipeline
