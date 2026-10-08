import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import schema

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPTS = REPO_ROOT / "legacy" / "upstream_scripts"
LEGACY_DATA = REPO_ROOT / "legacy" / "upstream_data"
BASELINE_DIR = REPO_ROOT / "tests" / "data" / "baseline"
VALIDATION_RECORDING = REPO_ROOT / "data" / "amiGA-amRNAi_n1_att2.csv"


def load_legacy(script_name: str):
    """Import an upstream script by file name (the names start with a digit)."""
    path = LEGACY_SCRIPTS / script_name
    spec = importlib.util.spec_from_file_location(f"legacy_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_track(track_id: str, frames, fps: float = 10.0) -> pd.DataFrame:
    """A canonical trajectory table for one track, tracked at the given frames."""
    frames = np.asarray(list(frames), dtype=float)
    table = pd.DataFrame({schema.TRACK_ID: track_id, schema.TIME_S: (frames / fps).round(1)})
    for offset, channel in enumerate(schema.REQUIRED_CHANNELS):
        table[channel] = frames * 0.5 + offset
    table[schema.IS_COILED] = 0.0
    table[schema.IS_WELL_ORIENTED] = (frames % 2 == 0).astype(float)
    return table


@pytest.fixture(scope="session")
def validation_raw() -> pd.DataFrame:
    """The real FIM-Track export used to validate the pipeline (not in the repo)."""
    if not VALIDATION_RECORDING.exists():
        pytest.skip(f"validation recording not present: {VALIDATION_RECORDING}")
    return pd.read_csv(VALIDATION_RECORDING, index_col=0)


@pytest.fixture(scope="session")
def validation_trajectories(tmp_path_factory) -> pd.DataFrame:
    """The validation recording through unmodified upstream 00 and the calibration of 01."""
    if not VALIDATION_RECORDING.exists():
        pytest.skip(f"validation recording not present: {VALIDATION_RECORDING}")
    wide_csv = tmp_path_factory.mktemp("wide") / "wide.csv"
    load_legacy("00_reorganize_raw_tracking_tables.py").raw_to_wide_format(
        VALIDATION_RECORDING, wide_csv, max_frame=10**9
    )
    table = schema.RAW_WIDE.to_canonical(pd.read_csv(wide_csv))
    table[schema.TIME_S] = (table.pop(schema.FRAME) / 10.0).round(1)
    positions = list(schema.POSITION_CHANNELS)
    table[positions] = (table[positions] * (240.0 / 2048.0)).round(3)
    return table
