"""The run manifest: one ``manifest.json`` per recording output folder.

It records, for every stage that has run, the resolved parameters, the
diagnostics, the warnings and a fingerprint of everything the result depends
on. A result without its parameters is not reproducible and does not count as
a result (CLAUDE.md §5).
"""

from __future__ import annotations

import datetime
import json
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from larval_explorer import __version__

SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"

STATUS_OK = "ok"
STATUS_FAILED = "failed"

_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "matplotlib")


def environment() -> dict[str, str]:
    """Versions of everything that can move a number."""
    versions = {"python": platform.python_version(), "app_version": __version__}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def new_manifest(recording_id: str, source_file: str | Path, metadata_values: dict[str, Any] | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "recording_id": recording_id,
        "metadata": dict(metadata_values or {}),
        "source_file": Path(source_file).as_posix(),
        "stages": {},
        "hmm_pool": [],
        # Set when stage 06 came from a fit pooled over several recordings (D-002).
        "hmm_pool_fingerprint": None,
        "hmm_state_mean_phi": {},
        "environment": environment(),
    }


def load_manifest(folder: Path) -> dict | None:
    """The manifest in a recording output folder, or None if there is none."""
    path = Path(folder) / MANIFEST_NAME
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has manifest schema version {version}; this version of the application reads {SCHEMA_VERSION}."
        )
    return manifest


def save_manifest(folder: Path, manifest: dict) -> Path:
    """Write the manifest so that an interrupted write never leaves half a file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MANIFEST_NAME
    temporary = folder / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(to_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def stage_record(
    *,
    status: str,
    params: dict,
    fingerprint: str,
    diagnostics: dict | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    duration_s: float = 0.0,
    outputs: list[str] | None = None,
    error: str | None = None,
) -> dict:
    record = {
        "status": status,
        "params": params,
        "diagnostics": diagnostics or {},
        "duration_s": round(float(duration_s), 3),
        "warnings": list(warnings),
        "fingerprint": fingerprint,
        "outputs": list(outputs or []),
        "completed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if error is not None:
        record["error"] = error
    return record


def to_json(value: Any) -> Any:
    """Plain JSON types from numpy scalars, arrays, tuples and non-finite floats."""
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value
