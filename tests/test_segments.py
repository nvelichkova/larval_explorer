import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import schema
from larval_explorer.core.params import CalibrationParams, SegmentationParams
from larval_explorer.core.segments import SegmentationError, find_runs, segment_tracks
from tests.conftest import make_track

CALIBRATION = CalibrationParams()


def run(table, **overrides):
    params = SegmentationParams(**{"min_segment_seconds": 1.0, **overrides})
    return segment_tracks(table, params, CALIBRATION, recording_id="rec")


def frames_of(table, track_id):
    times = table.loc[table[schema.TRACK_ID] == track_id, schema.TIME_S]
    return np.rint(times.to_numpy() * CALIBRATION.frame_rate_fps).astype(int)


def test_find_runs_splits_only_on_gaps_longer_than_the_tolerance():
    frames = np.array([0, 1, 2, 5, 6, 20, 21])  # 2 missing, then 13 missing
    assert find_runs(frames, 0) == [(0, 3), (3, 5), (5, 7)]
    assert find_runs(frames, 2) == [(0, 5), (5, 7)]
    assert find_runs(frames, 13) == [(0, 7)]
    assert find_runs(np.array([], dtype=int), 3) == []


def test_long_gap_splits_into_separately_named_segments():
    table = make_track("larva(3)", list(range(0, 30)) + list(range(100, 140)))
    result = run(table)
    ids = list(result.table[schema.TRACK_ID].unique())
    assert ids == ["rec__larva3__seg0", "rec__larva3__seg1"]
    assert (result.table[schema.SOURCE_TRACK_ID] == "larva(3)").all()
    assert not result.table[schema.INTERPOLATED].any()
    for track_id in ids:  # no kept segment contains a jump
        assert (np.diff(frames_of(result.table, track_id)) == 1).all()
    assert result.diagnostics["segments_out"] == 2 and result.diagnostics["frames_bridged"] == 0


def test_short_gap_is_bridged_by_linear_interpolation():
    table = make_track("larva(0)", list(range(0, 20)) + list(range(23, 40)))
    result = run(table)
    assert result.diagnostics["segments_out"] == 1 and result.diagnostics["frames_bridged"] == 3
    out = result.table
    assert (np.diff(frames_of(out, "rec__larva0__seg0")) == 1).all()
    filled = out[out[schema.INTERPOLATED]]
    assert filled[schema.TIME_S].tolist() == [2.0, 2.1, 2.2]
    # make_track builds every channel linearly in frame, so interpolation must reproduce it.
    assert filled[schema.MOM_X].tolist() == pytest.approx([10.0, 10.5, 11.0])
    assert not out[list(schema.REQUIRED_CHANNELS)].isna().any().any()
    # Flags carry the last tracked value instead of taking fractional values.
    assert set(filled[schema.IS_WELL_ORIENTED]) == {0.0}  # frame 19 is odd
    assert list(out.columns[-3:]) == [schema.SOURCE_TRACK_ID, schema.SEGMENT_INDEX, schema.INTERPOLATED]


def test_minimum_duration_applies_per_segment_not_per_track():
    # 5 s tracked, 60 s lost, 5 s tracked: the span is 70 s, the longest contiguous run 5 s.
    table = make_track("larva(0)", list(range(0, 51)) + list(range(651, 701)))
    result = run(table, min_segment_seconds=60.0)
    assert result.table.empty
    assert list(result.table.columns[:2]) == [schema.TRACK_ID, schema.TIME_S]
    assert result.diagnostics["segments_found"] == 2 and result.diagnostics["segments_dropped"] == 2
    assert len(result.warnings) == 1

    kept = run(table, min_segment_seconds=5.0)  # exactly 5.0 s: the threshold is inclusive
    assert kept.diagnostics["segments_out"] == 1


def test_segment_numbers_do_not_shift_when_an_earlier_segment_is_dropped():
    table = make_track("larva(0)", list(range(0, 5)) + list(range(100, 200)))
    result = run(table, min_segment_seconds=5.0)
    assert list(result.table[schema.TRACK_ID].unique()) == ["rec__larva0__seg1"]
    assert result.segments[schema.KEPT].tolist() == [False, True]


def test_larva_summary_counts_gaps_against_the_whole_recording():
    table = pd.concat([
        make_track("larva(0)", range(0, 100)),
        make_track("larva(1)", list(range(10, 40)) + list(range(42, 60))),  # late start, 2-frame gap, early end
    ])
    summary = run(table).larva_summary.set_index(schema.SOURCE_TRACK_ID)
    assert summary.loc["larva(0)", schema.GAP_COUNT] == 0
    assert summary.loc["larva(0)", schema.TRACKED_FRACTION] == 1.0
    row = summary.loc["larva(1)"]
    assert (row[schema.GAP_COUNT], row[schema.LONGEST_GAP_FRAMES]) == (3, 40)
    assert row[schema.TRACKED_FRACTION] == pytest.approx(0.48)
    assert (row[schema.SEGMENTS_FOUND], row[schema.SEGMENTS_KEPT]) == (1, 1)
    assert row[schema.KEPT_DURATION_S] == pytest.approx(4.9)


def test_repeated_frames_raise_a_clear_error():
    table = make_track("larva(0)", [0, 1, 1, 2])
    with pytest.raises(SegmentationError, match="repeated or unordered frames"):
        run(table)


def test_missing_columns_raise_schema_error():
    with pytest.raises(schema.SchemaError):
        run(make_track("larva(0)", range(20)).drop(columns=[schema.HEAD_X]))


def test_real_recording_matches_the_reference_figure(validation_trajectories):
    """Segment counts read off docs/track_segmentation_preview.png (bridge 10, min 60 s)."""
    result = segment_tracks(
        validation_trajectories, SegmentationParams(), CALIBRATION, recording_id="amiGA-amRNAi_n1_att2"
    )
    summary = result.larva_summary.sort_values(schema.LARVA_INDEX)
    assert summary[schema.SEGMENTS_FOUND].tolist() == [5, 1, 3, 4, 7, 5, 6, 6]
    assert summary[schema.SEGMENTS_KEPT].tolist() == [4, 1, 3, 3, 2, 3, 2, 3]
    # SPEC §2a table
    assert summary[schema.GAP_COUNT].tolist() == [4, 0, 3, 3, 7, 5, 5, 6]
    assert summary[schema.LONGEST_GAP_FRAMES].tolist() == [2116, 0, 148, 96, 2202, 3354, 4853, 1304]
    tracked_percent = (summary[schema.TRACKED_FRACTION] * 100).round(1).tolist()
    assert tracked_percent == [72.7, 100.0, 97.4, 98.1, 46.3, 55.3, 45.4, 77.2]
    assert result.diagnostics["larvae_in"] == 8 and result.diagnostics["segments_out"] == 21

    table = result.table
    assert table[schema.TRACK_ID].nunique() == 21
    assert not table[list(schema.REQUIRED_CHANNELS)].isna().any().any()
    for track_id, group in table.groupby(schema.TRACK_ID):
        frames = np.rint(group[schema.TIME_S].to_numpy() * 10).astype(int)
        assert (np.diff(frames) == 1).all(), track_id   # no segment contains a gap
        assert (frames[-1] - frames[0]) / 10 >= 60.0, track_id
