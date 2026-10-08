import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import schema
from tests.conftest import LEGACY_DATA


def test_step_table_round_trips_between_upstream_and_canonical_names():
    upstream = pd.DataFrame({
        "animal ID": ["trajectory_0_1"], "tempo_inicial (s)": [0.0], "tempo_final (s)": [1.0],
        "X0 (mm)": [0.0], "Y0 (mm)": [0.0], "X (mm)": [1.0], "Y (mm)": [1.0],
        "dX": [1.0], "dY": [1.0], "theta": [0.1], "phi": [0.1], "extra": ["kept"],
    })
    canonical = schema.RDP_STEPS.to_canonical(upstream)
    assert schema.TRACK_ID in canonical and schema.T_START_S in canonical and "extra" in canonical
    pd.testing.assert_frame_equal(schema.RDP_STEPS.to_upstream(canonical), upstream)


def test_missing_column_raises_schema_error_naming_both_spellings():
    with pytest.raises(schema.SchemaError) as error:
        schema.EVENTS.to_canonical(pd.DataFrame({"ID": ["a"], "type": ["crawl"], "start": [0.0]}))
    assert "'end_s'" in str(error.value) and "'end'" in str(error.value)
    assert not isinstance(error.value, KeyError)


def test_upstream_phi_maps_to_two_different_canonical_columns():
    assert schema.RDP_STEPS.upstream_name(schema.PHI) == schema.RUN_ANCHOR_STEPS.upstream_name(schema.HEADING)
    assert schema.PHI != schema.HEADING


def test_schemas_accept_the_published_upstream_tables():
    segments = schema.HMM_SEGMENTS.to_canonical(pd.read_csv(LEGACY_DATA / "representative_hmm_segments.csv"))
    assert {schema.START_TIME_S, schema.END_TIME_S, schema.STATE} <= set(segments.columns)
    body = schema.BODY_LENGTH.to_canonical(pd.read_csv(LEGACY_DATA / "mean_body_length_by_trajectory.csv"))
    assert body[schema.BODY_LENGTH_MEAN].notna().all()


def test_trajectory_column_sniffer_follows_upstream_precedence():
    table = pd.DataFrame(columns=["Larva", "tempo", "x", "y", "com_x", "com_y"])
    assert schema.detect_trajectory_columns(table) == ("Larva", "tempo", "com_x", "com_y")
    with pytest.raises(schema.SchemaError):
        schema.detect_trajectory_columns(pd.DataFrame(columns=["larva", "time"]))


def test_track_id_helpers():
    assert schema.larva_index("larva(12)") == 12
    assert schema.larva_index("trajectory_0_101") == 101
    assert schema.upstream_track_id("larva(7)") == "trajectory_0_7"
    assert schema.segment_track_id("rec_a", 3, 2) == "rec_a__larva3__seg2"
    with pytest.raises(schema.SchemaError):
        schema.larva_index("no digits")


def make_raw(channels, n_frames=4, larvae=("larva(0)", "larva(1)")):
    index = [f"{channel}({frame})" for channel in channels for frame in range(n_frames)]
    return pd.DataFrame(1.0, index=index, columns=list(larvae))


def test_validator_accepts_required_channels_and_ignores_extras():
    raw = make_raw(schema.REQUIRED_CHANNELS + ("velocity", "bending"))
    raw.iloc[0, 0] = np.nan  # an empty cell is a missing value, not an error
    report = schema.validate_fimtrack(raw)
    assert report.ok, report.summary()
    assert report.n_frames == 4 and report.larvae == ("larva(0)", "larva(1)")
    assert report.extra_channels == ("bending", "velocity")
    assert report.duration_seconds(10.0) == pytest.approx(0.4)


def test_validator_reports_missing_channels_without_key_error():
    raw = make_raw([c for c in schema.REQUIRED_CHANNELS if c not in (schema.TAIL_X, schema.AREA)])
    report = schema.validate_fimtrack(raw)
    assert not report.ok
    assert report.missing_channels == (schema.TAIL_X, schema.AREA)
    with pytest.raises(schema.SchemaError, match="tail_x"):
        report.raise_if_invalid()


def test_validator_reports_malformed_layout():
    raw = make_raw(schema.REQUIRED_CHANNELS, larvae=("animal_a", "larva(1)"))
    raw = pd.concat([raw, pd.DataFrame(1.0, index=["not a label"], columns=raw.columns)])
    raw["larva(1)"] = raw["larva(1)"].astype(object)
    raw.iloc[0, 1] = "x"
    report = schema.validate_fimtrack(raw)
    text = "\n".join(report.errors)
    assert "larva(N)" in text and "row label" in text and "Non-numeric" in text


def test_validator_on_the_real_recording(validation_raw):
    report = schema.validate_fimtrack(validation_raw)
    assert report.ok, report.summary()
    assert len(report.larvae) == 8
    assert (report.n_frames, report.first_frame, report.last_frame) == (9851, 0, 9850)
    assert len(report.channels) == 30 and len(report.extra_channels) == 10
