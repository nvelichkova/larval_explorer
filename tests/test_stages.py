"""Stage behaviour beyond upstream parity: segmentation on, and guard rails."""

import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import params as P
from larval_explorer.core import schema as S
from larval_explorer.core import stages
from tests.conftest import make_track

RECORDING_ID = "amiGA-amRNAi_n1_att2"


@pytest.fixture(scope="module")
def segmented_run(validation_raw):
    """The validation recording from raw export to HMM segments, segmentation on."""
    params = P.PipelineParams()
    out = {}
    out["wide"] = stages.stage_00(validation_raw, params.stage_00).tables["wide"]
    out["calibrated"] = stages.stage_01(out["wide"], params.stage_01).tables["calibrated"]
    out["stage_02"] = stages.stage_02(out["calibrated"], params.stage_02, params.stage_01, recording_id=RECORDING_ID)
    out["smoothed"] = stages.stage_03(out["stage_02"].tables["trajectories"], params.stage_03).tables["smoothed"]
    out["body_length"] = stages.stage_04(out["smoothed"], params.stage_04).tables["body_length"]
    out["rdp"] = stages.stage_05(out["smoothed"], out["body_length"], params.stage_05).tables["steps"]
    out["stage_06"] = stages.stage_06(out["rdp"], params.stage_06)
    out["filtered"] = stages.stage_07(out["stage_06"].tables["steps"], params.stage_07).tables["steps"]
    out["hmm_segments"] = stages.stage_08(out["filtered"], params.stage_08).tables["hmm_segments"]
    return out


def test_segmented_run_has_no_missing_values(segmented_run):
    assert segmented_run["stage_02"].diagnostics["segments_out"] == 21
    smoothed = segmented_run["smoothed"]
    assert smoothed[S.TRACK_ID].nunique() == 21
    assert not smoothed[list(S.REQUIRED_CHANNELS)].isna().any().any()
    assert segmented_run["body_length"][S.BODY_LENGTH_MEAN].notna().all()
    steps = segmented_run["rdp"]
    assert steps[S.TRACK_ID].nunique() == 21
    geometry = [S.T_START_S, S.T_END_S, S.X0_MM, S.Y0_MM, S.X1_MM, S.Y1_MM, S.DX_MM, S.DY_MM, S.VELOCITY, S.STEP_LENGTH]
    assert not steps[geometry].isna().any().any()


def test_no_rdp_step_spans_a_track_segment_boundary(segmented_run):
    """The corruption stage 02 exists to prevent (CLAUDE.md §6)."""
    track_segments = segmented_run["stage_02"].tables["track_segments"]
    kept = track_segments[track_segments[S.KEPT]].set_index(S.TRACK_ID)
    steps = segmented_run["rdp"]
    assert set(steps[S.TRACK_ID]) == set(kept.index)

    bounds = kept.loc[steps[S.TRACK_ID]]
    assert (steps[S.T_START_S].to_numpy() >= bounds[S.START_S].to_numpy()).all()
    assert (steps[S.T_END_S].to_numpy() <= bounds[S.END_S].to_numpy()).all()

    # Within a segment every frame is present, so a step can never cross lost frames.
    smoothed = segmented_run["smoothed"]
    for track_id, group in smoothed.groupby(S.TRACK_ID):
        frames = np.rint(group[S.TIME_S].to_numpy() * 10).astype(int)
        assert (np.diff(frames) == 1).all(), track_id


def test_upstream_path_does_analyse_steps_across_lost_frames():
    """With segmentation off, the same check fails: this is the upstream behaviour D-008 describes."""
    track = make_track("trajectory_0_0", list(range(0, 400)) + list(range(2000, 2400)))
    track[S.MOM_X] = np.r_[np.linspace(0, 10, 400), np.linspace(100, 110, 400)]
    track[S.MOM_Y] = 0.0
    calibration = P.CalibrationParams()
    off = stages.stage_02(track, P.SegmentationParams(enabled=False, min_segment_seconds=30), calibration, recording_id="r")
    on = stages.stage_02(track, P.SegmentationParams(min_segment_seconds=30), calibration, recording_id="r")

    def longest_step(trajectories):
        body = pd.DataFrame({S.TRACK_ID: trajectories[S.TRACK_ID].unique(), S.BODY_LENGTH_MEAN: 1.0})
        steps = stages.stage_05(trajectories, body, P.RdpParams()).tables["steps"]
        return steps[S.DURATION_S].max()

    assert longest_step(off.tables["trajectories"]) >= 160.0   # one chord over the 160 s gap
    assert longest_step(on.tables["trajectories"]) < 40.0


def test_hmm_states_keep_their_initialised_roles(segmented_run):
    """Stored indices are upstream's: 0 persistent, 1 reversing, 2 mixed (D-016)."""
    result = segmented_run["stage_06"]
    means = result.diagnostics["state_mean_phi"]
    assert means["0"] > means["2"] > means["1"]
    assert set(result.tables["steps"][S.STATE].unique()) <= {0, 1, 2}
    history = result.diagnostics["log_likelihood_history"]
    assert len(history) == result.diagnostics["iterations"] and history[-1] >= history[0]


def test_hmm_segments_stay_inside_their_track_segment(segmented_run):
    hmm_segments = segmented_run["hmm_segments"]
    assert len(hmm_segments) > 0
    track_segments = segmented_run["stage_02"].tables["track_segments"].set_index(S.TRACK_ID)
    bounds = track_segments.loc[hmm_segments[S.TRACK_ID]]
    assert (hmm_segments[S.START_TIME_S].to_numpy() >= bounds[S.START_S].to_numpy()).all()
    assert (hmm_segments[S.END_TIME_S].to_numpy() <= bounds[S.END_S].to_numpy()).all()


def test_pooled_hmm_keeps_recordings_apart_when_track_ids_repeat():
    """Two recordings both contain trajectory_0_1; they must stay two sequences (D-015)."""
    rng = np.random.default_rng(0)
    recording = "recording_id"

    def steps_for(recording_id, n):
        table = pd.DataFrame({
            S.TRACK_ID: "trajectory_0_1", recording: recording_id,
            S.T_START_S: np.arange(n, dtype=float), S.T_END_S: np.arange(n, dtype=float) + 1,
            S.X0_MM: 0.0, S.Y0_MM: 0.0, S.X1_MM: 1.0, S.Y1_MM: 1.0, S.DX_MM: 1.0, S.DY_MM: 1.0,
            S.THETA: 0.1, S.PHI: rng.normal(scale=0.8, size=n),
        })
        return table

    pooled = pd.concat([steps_for("rec_a", 60), steps_for("rec_b", 45)], ignore_index=True)
    keyed = stages.stage_06(pooled, P.HmmParams(), sequence_keys=(recording, S.TRACK_ID))
    assert keyed.diagnostics["sequences"] == 2
    collided = stages.stage_06(pooled, P.HmmParams())
    assert collided.diagnostics["sequences"] == 1   # what keying on the track ID alone would do


def test_smoothing_rejects_tracks_shorter_than_the_window():
    with pytest.raises(ValueError, match="shorter than the smoothing window"):
        stages.stage_03(make_track("larva(0)", range(30)), P.SmoothingParams())


def test_stage_00_rejects_an_export_with_missing_channels():
    raw = pd.DataFrame(1.0, index=[f"{c}({f})" for c in S.REQUIRED_CHANNELS[:-1] for f in range(3)], columns=["larva(0)"])
    with pytest.raises(S.SchemaError, match="is_well_oriented"):
        stages.stage_00(raw, P.ReorganizeParams())


def test_stage_00_max_frame_and_partial_frames():
    index = [f"{c}({f})" for c in S.REQUIRED_CHANNELS for f in range(5)]
    raw = pd.DataFrame(2.0, index=index, columns=["larva(0)", "larva(1)"])
    raw.loc[[f"{c}(1)" for c in S.REQUIRED_CHANNELS], "larva(1)"] = np.nan   # larva 1 lost at frame 1
    raw.loc["tail_x(2)", "larva(0)"] = np.nan                               # one empty cell
    wide = stages.stage_00(raw, P.ReorganizeParams(max_frame=3)).tables["wide"]
    assert wide[S.FRAME].max() == 3
    assert len(wide) == 4 + 3
    assert list(wide[S.TRACK_ID].unique()) == ["trajectory_0_0", "trajectory_0_1"]
    assert wide[S.TAIL_X].isna().sum() == 1 and wide[S.TAIL_X].dtype == float
    assert wide[S.HEAD_X].dtype == np.int64


def test_stage_06_explains_when_there_is_nothing_to_fit():
    steps = pd.DataFrame({column: pd.Series(dtype=float) for column in S.RDP_STEPS.required})
    with pytest.raises(ValueError, match="No RDP step has a defined phi"):
        stages.stage_06(steps, P.HmmParams())


def test_stage_11_with_no_events_is_empty_with_a_warning_not_an_error():
    segments = pd.DataFrame({
        S.TRACK_ID: ["t"], S.STATE: [2], S.START_INDEX: [0], S.END_INDEX: [9],
        S.START_TIME_S: [0.0], S.END_TIME_S: [50.0], S.N_STEPS: [10], S.DURATION_S: [50.0],
    })
    result = stages.stage_11(pd.DataFrame(), segments, P.SteeringParams())
    assert result.tables["segment_fits"].empty and result.tables["run_fits"].empty
    assert "No crawls or head casts" in result.warnings[0]
