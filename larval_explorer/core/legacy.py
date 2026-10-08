"""Access to pure helper functions in the read-only upstream scripts (D-013).

Only leaf helpers without I/O, pyplot or module-global tunables are imported;
every stage body is forked into ``stages.py``. The upstream file names start
with a digit, so they are loaded by path.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

UPSTREAM_SCRIPTS = Path(__file__).resolve().parents[2] / "legacy" / "upstream_scripts"

# Scripts that import cleanly: no pyplot, nothing executed at import.
_IMPORTABLE = {
    "04": "04_compute_mean_body_length.py",
    "05": "05_extract_rdp_steps.py",
    "07": "07_filter_short_hmm_runs.py",
    "10": "10_build_event_and_run_steps.py",
}


@lru_cache(maxsize=None)
def upstream(stage: str) -> ModuleType:
    """The upstream script for a stage number, e.g. ``upstream("05").rdp``."""
    if stage not in _IMPORTABLE:
        raise ValueError(f"Upstream script {stage} is not importable; its code is forked in stages.py.")
    path = UPSTREAM_SCRIPTS / _IMPORTABLE[stage]
    spec = importlib.util.spec_from_file_location(f"larval_explorer_upstream_{stage}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
