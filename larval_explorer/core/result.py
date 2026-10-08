"""The value every stage returns."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class StageResult:
    """What a stage produced: named tables, diagnostics for the manifest, warnings."""

    tables: dict[str, pd.DataFrame]
    diagnostics: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
