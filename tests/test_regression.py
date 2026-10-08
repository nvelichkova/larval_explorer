"""Upstream parity (CLAUDE.md §6, D-009).

The refactored stages must reproduce ``tests/data/baseline/``: the outputs of
the unmodified upstream scripts on one real recording, with track segmentation
disabled. If one of these fails, the refactor is wrong. Do not edit the
expected values and do not regenerate the baseline.

Each stage is checked in isolation, fed the frozen upstream output of the
stage before it, and compared exactly after the same CSV round trip the
pipeline performs when it saves. A second test chains the stages in memory.

Stage 11 has no frozen baseline (D-014); tests/test_steering.py checks it
against the upstream script run live.
"""

import io

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from larval_explorer.core import params as P
from larval_explorer.core import schema as S
from larval_explorer.core import stages
from tests.conftest import BASELINE_DIR

UPSTREAM_MAX_FRAME = 36000
SEGMENTATION_OFF = P.SegmentationParams(enabled=False)
RECORDING_ID = "amiGA-amRNAi_n1_att2"


def frozen(name: str) -> pd.DataFrame:
    return pd.read_csv(BASELINE_DIR / f"{name}.csv.gz")


def as_saved(table: pd.DataFrame, table_schema: S.TableSchema) -> pd.DataFrame:
    """The table as it reads back after being written with upstream column names."""
    return pd.read_csv(io.StringIO(table_schema.to_upstream(table).to_csv(index=False)))


def assert_matches_baseline(table, table_schema, name, exact=True):
    assert_frame_equal(as_saved(table, table_schema), frozen(name), check_exact=exact)


def test_stage_00(validation_raw):
    result = stages.stage_00(validation_raw, P.ReorganizeParams(max_frame=UPSTREAM_MAX_FRAME))
    assert_matches_baseline(result.tables["wide"], S.RAW_WIDE, "00_merged_tracking_table_unscaled")


def test_stage_01_and_02_disabled_reproduce_upstream_script_01():
    wide = S.RAW_WIDE.to_canonical(frozen("00_merged_tracking_table_unscaled"))
    calibration = P.CalibrationParams()
    calibrated = stages.stage_01(wide, calibration).tables["calibrated"]
    result = stages.stage_02(calibrated, SEGMENTATION_OFF, calibration, recording_id=RECORDING_ID)
    assert_matches_baseline(result.tables["trajectories"], S.TRAJECTORY, "01_trajectory_timeseries_calibrated")


def test_stage_03():
    trajectories = S.TRAJECTORY.to_canonical(frozen("01_trajectory_timeseries_calibrated"))
    result = stages.stage_03(trajectories, P.SmoothingParams())
    assert_matches_baseline(result.tables["smoothed"], S.TRAJECTORY, "03_trajectory_timeseries")


def test_stage_04():
    trajectories = S.TRAJECTORY.to_canonical(frozen("03_trajectory_timeseries"))
    result = stages.stage_04(trajectories, P.BodyLengthParams())
    assert_matches_baseline(result.tables["body_length"], S.BODY_LENGTH, "04_mean_body_length_by_trajectory")


def test_stage_05():
    trajectories = S.TRAJECTORY.to_canonical(frozen("03_trajectory_timeseries"))
    body_lengths = S.BODY_LENGTH.to_canonical(frozen("04_mean_body_length_by_trajectory"))
    result = stages.stage_05(trajectories, body_lengths, P.RdpParams())
    assert_matches_baseline(result.tables["steps"], S.RDP_STEPS, "05_rdp_steps")


def test_stage_06():
    steps = S.RDP_STEPS.to_canonical(frozen("05_rdp_steps"))
    result = stages.stage_06(steps, P.HmmParams())
    assert_matches_baseline(result.tables["steps"], S.HMM_STEPS, "06_rdp_steps_with_hmm_states")
    assert result.diagnostics["converged"]


def test_stage_07():
    steps = S.HMM_STEPS.to_canonical(frozen("06_rdp_steps_with_hmm_states"))
    result = stages.stage_07(steps, P.HmmFilterParams())
    assert_matches_baseline(result.tables["steps"], S.HMM_STEPS_FILTERED, "07_rdp_steps_with_filtered_hmm_states")


def test_stage_08():
    steps = S.HMM_STEPS_FILTERED.to_canonical(frozen("07_rdp_steps_with_filtered_hmm_states"))
    result = stages.stage_08(steps, P.HmmSegmentParams())
    assert_matches_baseline(result.tables["hmm_segments"], S.HMM_SEGMENTS, "08_representative_hmm_segments")


def test_stage_09():
    trajectories = S.TRAJECTORY.to_canonical(frozen("03_trajectory_timeseries"))
    result = stages.stage_09(trajectories, P.EventParams())
    assert_matches_baseline(result.tables["events"], S.EVENTS, "09_event_intervals")
    assert len(result.tables["signals"]) == len(trajectories)
    assert len(result.tables["thresholds"]) == trajectories[S.TRACK_ID].nunique()


def test_stage_10():
    trajectories = S.TRAJECTORY.to_canonical(frozen("03_trajectory_timeseries"))
    events = S.EVENTS.to_canonical(frozen("09_event_intervals"))
    result = stages.stage_10(trajectories, events, P.StepModelParams())
    assert_matches_baseline(result.tables["run_anchor_steps"], S.RUN_ANCHOR_STEPS, "10_run_anchor_steps")
    assert_matches_baseline(result.tables["event_level_steps"], S.EVENT_LEVEL_STEPS, "10_event_level_steps")


def reloaded(table: pd.DataFrame, table_schema: S.TableSchema) -> pd.DataFrame:
    """Save-and-reload between stages, as the upstream scripts do through their CSVs."""
    return table_schema.to_canonical(as_saved(table, table_schema))


def run_chain(raw: pd.DataFrame, between_stages) -> dict:
    """Run 00 -> 08; ``between_stages`` prepares each table for the stage that reads it."""
    params = P.PipelineParams(
        stage_00=P.ReorganizeParams(max_frame=UPSTREAM_MAX_FRAME),
        stage_02=SEGMENTATION_OFF,
    )
    out = {}
    wide = stages.stage_00(raw, params.stage_00).tables["wide"]
    calibrated = stages.stage_01(between_stages(wide, S.RAW_WIDE), params.stage_01).tables["calibrated"]
    trajectories = stages.stage_02(
        calibrated, params.stage_02, params.stage_01, recording_id=RECORDING_ID
    ).tables["trajectories"]
    out["smoothed"] = stages.stage_03(between_stages(trajectories, S.TRAJECTORY), params.stage_03).tables["smoothed"]
    smoothed = between_stages(out["smoothed"], S.TRAJECTORY)
    body_lengths = stages.stage_04(smoothed, params.stage_04).tables["body_length"]
    out["rdp"] = stages.stage_05(smoothed, between_stages(body_lengths, S.BODY_LENGTH), params.stage_05).tables["steps"]
    hmm_steps = stages.stage_06(between_stages(out["rdp"], S.RDP_STEPS), params.stage_06).tables["steps"]
    out["filtered"] = stages.stage_07(between_stages(hmm_steps, S.HMM_STEPS), params.stage_07).tables["steps"]
    out["hmm_segments"] = stages.stage_08(
        between_stages(out["filtered"], S.HMM_STEPS_FILTERED), params.stage_08
    ).tables["hmm_segments"]
    return out


CHAIN_OUTPUTS = [
    ("smoothed", S.TRAJECTORY, "03_trajectory_timeseries"),
    ("rdp", S.RDP_STEPS, "05_rdp_steps"),
    ("filtered", S.HMM_STEPS_FILTERED, "07_rdp_steps_with_filtered_hmm_states"),
    ("hmm_segments", S.HMM_SEGMENTS, "08_representative_hmm_segments"),
]


def test_chain_with_save_and_reload_reproduces_the_baseline_exactly(validation_raw):
    """00 -> 08 from the raw export, each table passed on as it reads back from CSV.

    Each table is written once and parsed once per reader, as upstream does.
    Parsing is lossy in the last bit and not idempotent, so a table must never
    be written, read, re-written and read again on its way to a consumer.
    """
    out = run_chain(validation_raw, between_stages=reloaded)
    for key, table_schema, name in CHAIN_OUTPUTS:
        assert_matches_baseline(out[key], table_schema, name)


def test_chain_in_memory_differs_from_the_baseline_only_by_rounding(validation_raw):
    """Without the reload, floats keep digits a CSV parse loses; nothing else changes.

    Upstream re-reads a CSV between every stage, and pandas' default parser is
    not exact to the last bit, so exact parity needs the reload above.
    """
    out = run_chain(validation_raw, between_stages=lambda table, table_schema: table)
    for key, table_schema, name in CHAIN_OUTPUTS:
        assert_frame_equal(as_saved(out[key], table_schema), frozen(name), check_exact=False, rtol=1e-9)
