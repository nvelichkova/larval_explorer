"""The dataset registry: one row per recording, with free condition metadata.

``recording_id`` and ``path`` are required. Every other column is metadata the
user chose (genotype, age, ...) and nothing in the pipeline knows its name
(D-003). Metadata can be pre-filled from file names with a regular expression,
but only through an explicit preview-then-apply step (D-011).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

RECORDING_ID = "recording_id"
PATH = "path"
REQUIRED_COLUMNS = (RECORDING_ID, PATH)

# Columns of a filename-parsing preview that are not metadata.
MATCHED = "matched"

REGISTRY_NAME = "dataset_registry.csv"


class RegistryError(ValueError):
    """The registry is malformed or an edit would make it so."""


def empty_registry() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="string") for column in REQUIRED_COLUMNS})


def load_registry(path: Path) -> pd.DataFrame:
    """Read a registry CSV. Every cell is text; an empty cell is an empty string."""
    registry = pd.read_csv(path, dtype="string", keep_default_na=False)
    validate_registry(registry)
    return registry


def save_registry(registry: pd.DataFrame, path: Path) -> None:
    validate_registry(registry)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False)


def validate_registry(registry: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in registry.columns]
    if missing:
        raise RegistryError(f"Registry is missing required column(s): {missing}")
    ids = registry[RECORDING_ID].astype("string")
    if (ids.str.strip() == "").any() or ids.isna().any():
        raise RegistryError("Every registry row needs a recording_id.")
    duplicated = sorted(ids[ids.duplicated()].unique())
    if duplicated:
        raise RegistryError(f"Duplicate recording_id(s): {duplicated}")


def metadata_columns(registry: pd.DataFrame) -> list[str]:
    """The condition columns: everything except the two required ones."""
    return [column for column in registry.columns if column not in REQUIRED_COLUMNS]


def recording_id_for(path: Path) -> str:
    """Default ID of a recording: its file name without the extension."""
    return Path(path).stem


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    """Recording files named by a selection of files and folders.

    Every CSV is one recording (D-021). A folder stands for the CSVs directly
    inside it, in name order; it is a convenience for adding many recordings
    at once, typically of one condition, and never a recording itself.
    """
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found = sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".csv")
            if not found:
                raise RegistryError(f"No CSV files in folder: {path}")
            files.extend(found)
        else:
            files.append(path)
    return files


def add_recordings(registry: pd.DataFrame, paths: Iterable[Path]) -> pd.DataFrame:
    """Append one row per recording file; folders are expanded to their CSVs.

    Metadata is left blank; nothing is parsed here.
    """
    rows = [{RECORDING_ID: recording_id_for(path), PATH: Path(path).as_posix()} for path in expand_paths(paths)]
    if not rows:
        return registry.copy()
    combined = pd.concat([registry, pd.DataFrame(rows, dtype="string")], ignore_index=True)
    combined = combined.astype("string").fillna("")
    validate_registry(combined)
    return combined


def recording_file(path: Path) -> Path:
    """The raw FIM-Track CSV of a registry row. A row never names a folder (D-021)."""
    path = Path(path)
    if path.is_dir():
        raise RegistryError(
            f"'{path}' is a folder. Each registry row is one CSV; add the folder to get one row per CSV in it."
        )
    if not path.is_file():
        raise RegistryError(f"Recording file not found: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Filename parsing (D-011)
# ──────────────────────────────────────────────────────────────────────────────

def parse_name(pattern: str, name: str) -> dict[str, str] | None:
    """Named groups of ``pattern`` matched against the whole name, or None."""
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise RegistryError(f"Invalid filename pattern: {error}") from error
    if not compiled.groupindex:
        raise RegistryError("The filename pattern has no named groups, e.g. (?P<genotype>.+?)_n(?P<n>\\d+)")
    reserved = [name_ for name_ in compiled.groupindex if name_ in REQUIRED_COLUMNS or name_ == MATCHED]
    if reserved:
        raise RegistryError(f"Pattern group name(s) not allowed: {reserved}")
    match = compiled.fullmatch(name)
    if match is None:
        return None
    return {group: (value or "") for group, value in match.groupdict().items()}


def preview_parse(registry: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """What applying ``pattern`` would fill in, without changing the registry.

    One row per recording: its ID, a ``matched`` flag, and one column per
    named group. Rows whose name does not match are flagged and left blank.
    """
    validate_registry(registry)
    rows = []
    for recording_id, path in zip(registry[RECORDING_ID], registry[PATH]):
        parsed = parse_name(pattern, recording_id_for(Path(path)) if path else recording_id)
        rows.append({RECORDING_ID: recording_id, MATCHED: parsed is not None, **(parsed or {})})
    groups = list(re.compile(pattern).groupindex)
    preview = pd.DataFrame(rows, columns=[RECORDING_ID, MATCHED, *groups])
    preview[groups] = preview[groups].fillna("")
    return preview


def apply_parse(registry: pd.DataFrame, preview: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """Copy previewed values into the registry.

    Only matched rows are touched. A cell the user already filled is kept
    unless ``overwrite`` is set.
    """
    result = registry.copy().astype("string").fillna("")
    groups = [column for column in preview.columns if column not in (RECORDING_ID, MATCHED)]
    for group in groups:
        if group not in result.columns:
            result[group] = pd.Series("", index=result.index, dtype="string")
    values = preview.set_index(RECORDING_ID)
    for index, recording_id in result[RECORDING_ID].items():
        if recording_id not in values.index or not bool(values.loc[recording_id, MATCHED]):
            continue
        for group in groups:
            if overwrite or result.at[index, group] == "":
                result.at[index, group] = str(values.loc[recording_id, group])
    return result


def unlabelled(registry: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Rows with a blank in any of the given metadata columns (default: all).

    An unlabelled recording that reaches aggregation lands in a group of its
    own without anyone noticing, so the UI shows these before a run.
    """
    columns = list(columns) if columns is not None else metadata_columns(registry)
    if not columns or registry.empty:
        return registry.iloc[:0]
    blank = registry[columns].astype("string").fillna("").apply(lambda column: column.str.strip() == "")
    return registry[blank.to_numpy(dtype=bool).any(axis=1)]


def existing_values(registry: pd.DataFrame, column: str) -> list[str]:
    """Distinct non-blank values of a metadata column, for autocomplete."""
    if column not in registry.columns:
        return []
    values = registry[column].astype("string").fillna("").str.strip()
    return sorted(value for value in values.unique() if value)


def metadata_for(registry: pd.DataFrame, recording_id: str) -> dict[str, str]:
    """The metadata of one recording, as stored in its manifest."""
    row = registry[registry[RECORDING_ID] == recording_id]
    if row.empty:
        raise RegistryError(f"No recording '{recording_id}' in the registry.")
    return {column: str(row.iloc[0][column]) for column in metadata_columns(registry)}
