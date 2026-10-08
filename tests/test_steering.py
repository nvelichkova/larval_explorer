"""Stage 11 against the upstream script itself.

There is no frozen baseline for stage 11: upstream script 11 cannot read
script 08's output as published (D-014). Instead these tests run the upstream
code live. The helpers are compared function by function, and the whole stage
is compared with upstream ``main()`` after setting the two segment-time column
constants to the names D-014 settles on. Nothing else in the script is changed.
"""

import gzip
import io
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # upstream script 11 draws with pyplot

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from larval_explorer.core import params as P
from larval_explorer.core import schema as S
from larval_explorer.core import steering
from tests.conftest import BASELINE_DIR, load_legacy


@pytest.fixture(scope="module")
def upstream_11():
    return load_legacy("11_fit_steering_by_strategy.py")


def test_numerical_helpers_are_identical_to_upstream(upstream_11):
    rng = np.random.default_rng(3)
    for n in (0, 1, 2, 3, 5, 12, 200):
        x = rng.normal(size=n)
        w = rng.uniform(0.0, 5.0, size=n)
        if n > 4:
            x[1], w[2] = np.nan, 0.0
        theta = rng.uniform(-np.pi, np.pi, size=n)
        cases = [
            (steering.weighted_mean, upstream_11.weighted_mean, (x, w)),
            (steering.weighted_std, upstream_11.weighted_std, (x, w)),
            (steering.weighted_quantile, upstream_11.weighted_quantile, (x, w, 0.3)),
            (steering.weighted_kde_peak, upstream_11.weighted_kde_peak, (x, w, 0.02)),
            (steering.fit_vonmises, upstream_11.fit_vonmises, (theta,)),
            (steering.fit_vonmises, upstream_11.fit_vonmises, (theta * 0.1,)),
            (steering.fit_ar2, upstream_11.fit_ar2, (theta,)),
            (steering.circular_autocorr_with_counts, upstream_11.circular_autocorr_with_counts, (theta, 4)),
        ]
        for ours, theirs, args in cases:
            np.testing.assert_equal(_plain(ours(*args)), _plain(theirs(*args)), err_msg=f"{theirs.__name__} n={n}")
        grid = np.linspace(-3, 3, 50)
        np.testing.assert_array_equal(
            steering.weighted_kde_gaussian(grid, x, w, bw_min=0.02), upstream_11.weighted_kde_gaussian(grid, x, w, bw_min=0.02)
        )
        np.testing.assert_array_equal(steering.vonmises_pdf(grid, 0.3, 2.0), upstream_11.vonmises_pdf(grid, 0.3, 2.0))


def _plain(value):
    """Nested tuples/dicts of arrays and floats -> comparable structure."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return np.asarray(value)


def test_ar2_recovers_known_coefficients():
    rng = np.random.default_rng(11)
    a1, a2, c = 0.5, -0.3, 0.02
    theta = np.zeros(20000)
    for i in range(2, len(theta)):
        theta[i] = a1 * theta[i - 1] + a2 * theta[i - 2] + c + rng.normal(scale=0.1)
    fitted_a1, fitted_a2, fitted_c, sigma, _, _ = steering.fit_ar2(theta)
    assert (fitted_a1, fitted_a2, fitted_c, sigma) == pytest.approx((a1, a2, c, 0.1), abs=0.02)
    rho, kappa, mu, _ = steering.ar2_to_params(a1, a2, c, 0.1)
    assert (rho, kappa, mu) == pytest.approx((0.3, 0.8, 0.025))


# ──────────────────────────────────────────────────────────────────────────────
# Whole stage against upstream main()
# ──────────────────────────────────────────────────────────────────────────────

UPSTREAM_OUTPUTS = {
    "run_fits": ("run_fits.csv", S.RUN_FITS),
    "segment_fits": ("segment_fits.csv", S.SEGMENT_FITS),
    "headcast_segment_metrics": ("headcast_segment_metrics.csv", S.HEADCAST_SEGMENT_METRICS),
    "autocorr_by_state": ("autocorr_by_state.csv", S.STATE_TABLE),
    "state_param_summary": ("state_param_summary.csv", S.STATE_TABLE),
    "simulation_parameters": ("simulation_parameters.csv", S.STATE_TABLE),
}

# Upstream defaults, then a looser setting that lets many more runs through the fit.
SETTINGS = {
    "upstream_defaults": {},
    "relaxed": {"min_points_per_run": 4, "require_stable": False, "min_events_per_seg_autocorr": 3, "max_lag": 2},
}
UPSTREAM_CONSTANT = {
    "min_points_per_run": "MIN_POINTS_PER_RUN",
    "require_stable": "REQUIRE_STABLE",
    "min_events_per_seg_autocorr": "MIN_EVENTS_PER_SEG_AUTOCORR",
    "max_lag": "MAX_LAG",
}


def read_csv_or_empty(source) -> pd.DataFrame:
    try:
        return pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays:FutureWarning")
@pytest.mark.parametrize("setting", SETTINGS)
def test_stage_11_matches_upstream_main(upstream_11, tmp_path, monkeypatch, setting):
    inputs = {}
    for name in ("10_event_level_steps", "08_representative_hmm_segments"):
        path = tmp_path / f"{name}.csv"
        path.write_bytes(gzip.decompress((BASELINE_DIR / f"{name}.csv.gz").read_bytes()))
        inputs[name] = path

    overrides = SETTINGS[setting]
    monkeypatch.setattr(upstream_11, "LOW_PATH", str(inputs["10_event_level_steps"]))
    monkeypatch.setattr(upstream_11, "SEG_PATH", str(inputs["08_representative_hmm_segments"]))
    monkeypatch.setattr(upstream_11, "OUTDIR", str(tmp_path / "out"))
    # The D-014 correction: read the columns script 08 writes.
    monkeypatch.setattr(upstream_11, "SEG_T0_COL", "start_time_s")
    monkeypatch.setattr(upstream_11, "SEG_TF_COL", "end_time_s")
    for field, value in overrides.items():
        monkeypatch.setattr(upstream_11, UPSTREAM_CONSTANT[field], value)
    # Figures are not under test, and upstream's boxplot call uses a keyword
    # that current matplotlib no longer accepts. The CSVs do not depend on them.
    for name in dir(upstream_11):
        if name.startswith("plot_"):
            monkeypatch.setattr(upstream_11, name, lambda *args, **kwargs: None)
    upstream_11.main()

    result = steering.stage_11(
        S.EVENT_LEVEL_STEPS.to_canonical(pd.read_csv(inputs["10_event_level_steps"])),
        S.HMM_SEGMENTS.to_canonical(pd.read_csv(inputs["08_representative_hmm_segments"])),
        P.SteeringParams(**overrides),
    )

    compared_rows = 0
    for table_name, (file_name, table_schema) in UPSTREAM_OUTPUTS.items():
        expected = read_csv_or_empty(tmp_path / "out" / file_name)
        ours = result.tables[table_name]
        if expected.empty:
            assert ours.empty, table_name
            continue
        saved = pd.read_csv(io.StringIO(table_schema.to_upstream(ours).to_csv(index=False)))
        assert_frame_equal(saved, expected, check_exact=True, obj=table_name)
        compared_rows += len(expected)
    assert compared_rows > 0
    if setting == "relaxed":
        assert len(result.tables["run_fits"]) > 0 and len(result.tables["segment_fits"]) > 0


def test_stage_11_reads_the_columns_stage_08_writes():
    """The unmodified upstream constants name columns that exist nowhere (D-014)."""
    segments = pd.read_csv(BASELINE_DIR / "08_representative_hmm_segments.csv.gz")
    assert "start_time_s" in segments and "t_inicio (s)" not in segments
    canonical = S.HMM_SEGMENTS.to_canonical(segments)
    assert {S.START_TIME_S, S.END_TIME_S} <= set(canonical.columns)


def test_stage_11_with_no_segments_returns_empty_tables_and_a_warning():
    events = S.EVENT_LEVEL_STEPS.to_canonical(pd.read_csv(BASELINE_DIR / "10_event_level_steps.csv.gz"))
    segments = S.HMM_SEGMENTS.to_canonical(pd.read_csv(BASELINE_DIR / "08_representative_hmm_segments.csv.gz")).iloc[:0]
    result = steering.stage_11(events, segments, P.SteeringParams())
    assert all(table.empty for table in result.tables.values())
    assert result.warnings
