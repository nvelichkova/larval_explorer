"""Stage 12 phase T1: per-frame spine geometry on synthetic spines with known answers."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import schema as S
from larval_explorer.core import turns as T
from larval_explorer.core.events import body_angle_series
from larval_explorer.core.params import TurnKinematicsParams

from tests.conftest import make_track

FPS = 10.0
LINKS = (0.8, 0.7, 0.7, 0.6)  # head->sp1 ... sp3->tail, mm


def spine_from_links(head, direction, bends, lengths=LINKS):
    """Five points from the head backward. ``direction`` is the head->sp1 link's angle;
    ``bends[i]`` is the turn from link i to link i+1 (so the bend at boundary i+1)."""
    angles = np.cumsum([direction, *bends])
    points = [np.asarray(head, dtype=float)]
    for angle, length in zip(angles, lengths):
        points.append(points[-1] + length * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(points)


def track_from_points(points, mom=None, track_id="t"):
    """A stage 02-shaped track from an array of shape (frames, 5, 2)."""
    n = len(points)
    track = make_track(track_id, range(n), fps=FPS)
    for k, (x, y) in enumerate(T.SPINE_POINTS):
        track[x] = points[:, k, 0]
        track[y] = points[:, k, 1]
    mom = points.mean(axis=1) if mom is None else mom
    track[S.MOM_X], track[S.MOM_Y] = mom[:, 0], mom[:, 1]
    track[S.IS_WELL_ORIENTED] = 1.0
    return track


def kinematics(points, **kwargs):
    params = TurnKinematicsParams(**kwargs)
    return T.track_kinematics(track_from_points(points), params, FPS)


INTERIOR = slice(10, -10)  # away from the Savitzky-Golay edge fits


def test_rigid_rotation_has_no_bend_and_the_imposed_heading_on_every_axis():
    n = 120
    theta = np.linspace(0, np.deg2rad(400), n)  # beyond 360 degrees: unwrap must hold
    points = np.array([spine_from_links((5 + 0.01 * i, 3), th, (0, 0, 0)) for i, th in enumerate(theta)])
    k = kinematics(points)
    for b in T.BOUNDARIES:
        assert np.abs(k[S.bend_column(b)]).max() < 1e-9
    # The axes point from tail to front, i.e. opposite to the head->tail links.
    expected = np.rad2deg(theta + np.pi)
    for axis in T.AXIS_POINTS:
        heading = k[S.heading_column(axis)].to_numpy()
        offset = heading[60] - expected[60]
        assert abs(offset - 360 * round(offset / 360)) < 1e-6
        assert np.allclose((heading - offset)[INTERIOR], expected[INTERIOR], atol=1e-6), axis


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_a_single_constant_bend_is_measured_at_its_boundary_with_its_sign(boundary):
    bends = [0.0, 0.0, 0.0]
    bends[boundary - 1] = np.deg2rad(35)
    points = np.repeat(spine_from_links((0, 0), 0.3, bends)[None], 30, axis=0)
    k = kinematics(points)
    for b in T.BOUNDARIES:
        # spine_from_links turns walking head->tail; bends are measured walking tail->head (D-032).
        assert np.allclose(k[S.bend_column(b)], -35 if b == boundary else 0, atol=1e-9)
    assert np.allclose(k[S.TOTAL_BEND_DEG], 35)
    assert np.allclose(k[S.BEND_PEAK_BOUNDARY], boundary)
    assert (k[S.BEND_PEAK_AT_EDGE] == (boundary != 2)).all()

    mirrored = kinematics(np.repeat(spine_from_links((0, 0), 0.3, [-x for x in bends])[None], 30, axis=0))
    assert np.allclose(mirrored[S.bend_column(boundary)], 35, atol=1e-9)


def test_bends_walk_tail_to_head_and_share_the_head_angle_sign():
    # Head at the origin, tail at (4, 3). Walking from the tail, the spine turns
    # from direction (-1, -1) to (-1, 0) at sp1: -45 degrees (clockwise in xy).
    points = np.array([[[0, 0], [1, 0], [2, 1], [3, 2], [4, 3]]], dtype=float)
    assert np.rad2deg(T.bend_angles(points))[0, 0] == pytest.approx(-45)
    # A C-shape: every bend has the sign of the head angle.
    c_shape = spine_from_links((0, 0), 0.2, np.deg2rad([25, 30, 20]))[None]
    bends, head = T.bend_angles(c_shape)[0], T.head_angle(c_shape)[0]
    assert np.all(np.sign(bends) == np.sign(head)) and head != 0


def test_head_angle_equals_stage_09_body_angle():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(50, 5, 2))
    a_post = np.arctan2(points[:, 2, 1] - points[:, 4, 1], points[:, 2, 0] - points[:, 4, 0])
    a_ant = np.arctan2(points[:, 0, 1] - points[:, 2, 1], points[:, 0, 0] - points[:, 2, 0])
    assert np.allclose(T.head_angle(points), body_angle_series(a_post, a_ant))


def test_arc_fractions_follow_the_link_lengths():
    points = spine_from_links((0, 0), 0, (0.4, -0.2, 0.3), lengths=(8, 7, 7, 6))[None]
    assert np.allclose(T.boundary_arc_fractions(points)[0], [8 / 28, 15 / 28, 22 / 28])


def test_bend_peak_parabola_and_edges():
    index, edge = T.bend_peak(np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [3.0, 1.0, 0.0], [0.0, 1.0, 5.0], [0.0, 0.0, 0.0]]))
    assert index[0] == pytest.approx(2 - 1 / 6)
    assert index[1] == pytest.approx(2.0)
    assert index[2] == 1 and edge[2]
    assert index[3] == 3 and edge[3]
    assert np.isnan(index[4]) and not edge[4]
    assert not edge[0] and not edge[1]


def test_bend_trail_interpolates_the_20_percent_crossing_and_censors_at_sp3():
    index, censored = T.bend_trail(np.array([[10.0, 10.0, 1.0], [10.0, 1.0, 0.0], [10.0, 5.0, 3.0], [0.0, 0.0, 0.0]]), 0.2)
    assert index[0] == pytest.approx(2 + 8 / 9) and not censored[0]
    assert index[1] == pytest.approx(1 + 8 / 9) and not censored[1]
    assert index[2] == 3 and censored[2]
    assert np.isnan(index[3]) and not censored[3]


def test_bend_trail_takes_the_posterior_most_boundary_even_past_a_dip():
    # An S-shape: the definition is posterior-most, not contiguous with the peak.
    index, censored = T.bend_trail(np.array([[10.0, 1.0, 3.0]]), 0.2)
    assert index[0] == 3 and censored[0]


def test_index_to_arc_interpolates_between_boundaries():
    arcs = np.array([[0.3, 0.5, 0.8], [0.3, 0.5, 0.8]])
    assert np.allclose(T.index_to_arc(np.array([1.5, 2.5]), arcs), [0.4, 0.65])


def test_model_boundary_is_eleven_times_arc_fraction():
    bends = (0.0, np.deg2rad(40), 0.0)
    k = kinematics(np.repeat(spine_from_links((0, 0), 0, bends)[None], 30, axis=0))
    assert np.allclose(k[S.BEND_PEAK_ARC_FRACTION], 1.5 / 2.8)
    assert np.allclose(k[S.BEND_PEAK_MODEL_BOUNDARY], 11 * 1.5 / 2.8)


def test_forward_progression_counts_only_motion_toward_the_front():
    n = 40
    shape = spine_from_links((0, 0), np.pi, (0, 0, 0))  # head at origin, body along -x, facing +x
    forward = np.array([shape + [0.05 * i, 0] for i in range(n)])
    k = kinematics(forward)
    assert np.allclose(k[S.POSTERIOR_DISTANCE_MM].iloc[-1], 0.05 * (n - 1))
    assert np.allclose(k[S.TAIL_FORWARD_SPEED_MM_S].iloc[1:], 0.5)

    backward = kinematics(forward[::-1])
    assert np.allclose(backward[S.POSTERIOR_DISTANCE_MM], 0, atol=1e-12)
    assert np.allclose(backward[S.TAIL_PATH_LENGTH_MM].iloc[-1], 0.05 * (n - 1))

    sideways = kinematics(np.array([shape + [0, 0.05 * i] for i in range(n)]))
    assert np.allclose(sideways[S.POSTERIOR_DISTANCE_MM], 0, atol=1e-12)


def test_a_bend_moving_back_is_tracked_boundary_by_boundary():
    shapes = []
    for boundary in (1, 2, 3):
        bends = [0.0, 0.0, 0.0]
        bends[boundary - 1] = np.deg2rad(40)
        shapes += [spine_from_links((0, 0), 0, bends)] * 20
    k = kinematics(np.array(shapes), spine_sg_window=3, spine_sg_order=1)
    peaks = k[S.BEND_PEAK_BOUNDARY].to_numpy()
    assert np.allclose(peaks[2:18], 1) and np.allclose(peaks[22:38], 2) and np.allclose(peaks[42:58], 3)


def test_head_tail_swap_is_detected_on_unsmoothed_points():
    n = 30
    points = np.array([spine_from_links((0.01 * i, 0), 0, (0, 0, 0)) for i in range(n)])
    points[15:] = points[15:, ::-1]
    k = kinematics(points)
    assert np.flatnonzero(k[S.SPINE_FLIP]).tolist() == [15]


def test_missing_frames_and_short_tracks_are_refused():
    points = np.repeat(spine_from_links((0, 0), 0, (0.1, 0, 0))[None], 30, axis=0)
    track = track_from_points(points).drop(index=12)
    with pytest.raises(ValueError, match="missing frames"):
        T.track_kinematics(track, TurnKinematicsParams(), FPS)
    with pytest.raises(ValueError, match="fewer than"):
        T.track_kinematics(track_from_points(points[:5]), TurnKinematicsParams(), FPS)


def test_noise_floor_uses_clean_crawl_frames_only():
    n = 60
    rng = np.random.default_rng(1)
    points = np.array([spine_from_links((0, 0), 0, rng.normal(0, 0.05, 3)) for _ in range(n)])
    k = kinematics(points)
    k.loc[k.index[25], S.IS_WELL_ORIENTED] = 0
    events = pd.DataFrame({
        S.TRACK_ID: ["t", "t"],
        S.EVENT_TYPE: [S.EVENT_CRAWL, S.EVENT_HEAD_CAST],
        S.START_S: [2.0, 4.0],
        S.END_S: [3.0, 5.0],
    })
    floor = T.noise_floor(k, events)
    row = floor[floor[S.QUANTITY] == S.bend_column(2)].iloc[0]
    inside = k[(k[S.TIME_S] >= 2.0) & (k[S.TIME_S] <= 3.0) & (k[S.IS_WELL_ORIENTED] == 1)]
    assert row[S.N_FRAMES] == len(inside) == 10
    assert row[S.NOISE_SD] == pytest.approx(np.std(inside[S.bend_column(2)]))


def test_larva_index_reads_the_larva_not_the_segment_of_a_segment_id():
    assert S.larva_index(S.segment_track_id("amiRNAi_w1118_n1_att2", 10, 1)) == 10
    assert S.larva_index(S.segment_track_id("rec_n1_att2", 3, 12)) == 3
    assert S.larva_index("larva(7)") == 7
    assert S.larva_index("trajectory_0_101") == 101


def spine_from_tail(tail, phi, beta, link=0.85):
    """Five points from the tail forward: the posterior three links along ``phi``,
    the head link turned by ``beta`` at sp1."""
    u = lambda a: np.array([np.cos(a), np.sin(a)])
    sp3 = tail + link * u(phi)
    sp2 = sp3 + link * u(phi)
    sp1 = sp2 + link * u(phi)
    head = sp1 + link * u(phi + beta)
    return np.array([head, sp1, sp2, sp3, tail])


def synthetic_turn_recording():
    """Crawl at 0.5 mm/s; posterior heading ramps 0 -> 40 deg over 5.0-6.5 s; a head
    bend rises and falls over 5-7 s, peaking at 6.0 s. One head cast, two crawls."""
    t = np.arange(121) / FPS

    def ramp(x):
        x = np.clip(x, 0, 1)
        return 3 * x**2 - 2 * x**3

    phi = np.deg2rad(40) * ramp((t - 5) / 1.5)
    beta = np.deg2rad(100) * np.where((t >= 5) & (t <= 7), np.sin(np.pi * (t - 5) / 2) ** 2, 0)
    tail = np.zeros((len(t), 2))
    for i in range(1, len(t)):
        tail[i] = tail[i - 1] + 0.05 * np.array([np.cos(phi[i - 1]), np.sin(phi[i - 1])])
    points = np.array([spine_from_tail(tail[i], phi[i], beta[i]) for i in range(len(t))])
    track = track_from_points(points, track_id="rec__larva0__seg0")
    track[S.IS_COILED] = 0.0
    events = pd.DataFrame({
        S.TRACK_ID: track[S.TRACK_ID].iloc[0],
        S.EVENT_TYPE: [S.EVENT_CRAWL, S.EVENT_HEAD_CAST, S.EVENT_CRAWL],
        S.START_S: [0.5, 5.2, 8.0], S.END_S: [4.5, 6.8, 11.0],
    })
    body = pd.DataFrame({S.TRACK_ID: [track[S.TRACK_ID].iloc[0]], S.BODY_LENGTH_MEAN: [3.4]})
    return track, events, body, phi


def test_synthetic_turn_has_the_constructed_heading_change_at_sweep_end():
    track, events, body, phi = synthetic_turn_recording()
    params = TurnKinematicsParams(end_total_bend_deg=16.6)
    result = T.stage_12(track, events, body, params, FPS, pd.Series({track[S.TRACK_ID].iloc[0]: 1}))
    turn = result.tables["turns"].iloc[0]
    assert turn[S.INCLUDED], turn[S.EXCLUSION_REASON]
    assert turn[S.T_SWEEP_END_S] == pytest.approx(6.0, abs=0.11)
    i_on, i_sw, i_end = (int(round(turn[c] * FPS)) for c in (S.T_ONSET_S, S.T_SWEEP_END_S, S.T_END_S))
    expected_sweep = np.rad2deg(phi[i_sw] - phi[i_on])
    expected_total = np.rad2deg(phi[i_end] - phi[i_on])
    assert turn[S.T_ONSET_S] == pytest.approx(5.2)  # the cast start: the body is still straight there
    assert turn[S.T_END_S] >= 6.5  # after the posterior has finished turning
    assert expected_total == pytest.approx(40 - np.rad2deg(phi[i_on]), abs=1e-9)
    assert turn[S.DPSI_SWEEP_DEG] == pytest.approx(expected_sweep, abs=1.0)
    assert turn[S.DPSI_TOTAL_DEG] == pytest.approx(expected_total, abs=1.0)
    assert turn[S.FRAC_HEADING_IN_SWEEP] == pytest.approx(expected_sweep / expected_total, abs=0.03)
    assert turn[S.FRAC_HEADING_IN_SWEEP_SIGNED] > 0
    assert turn[S.N_SWEEPS] == 1 and not turn[S.MULTI_SWEEP]
    assert turn[S.at_max_column(S.BEND_PEAK_BOUNDARY)] == 1 and turn[S.at_max_column(S.BEND_PEAK_AT_EDGE)]


def _swap_rows(track, rows):
    return T.swap_labels(track, np.isin(np.arange(len(track)), rows))


def test_a_swap_inside_a_turn_is_corrected_exactly_instead_of_excluding_the_turn():
    track, events, body, _ = synthetic_turn_recording()
    params = TurnKinematicsParams(end_total_bend_deg=16.6)
    clean = T.stage_12(track, events, body, params, FPS).tables["turns"].iloc[0]
    corrupted = _swap_rows(track, range(54, 67))  # 5.4-6.6 s: labels swapped mid-turn

    fixed = T.stage_12(corrupted, events, body, params, FPS)
    turn = fixed.tables["turns"].iloc[0]
    assert turn[S.INCLUDED], turn[S.EXCLUSION_REASON]
    assert turn[S.N_CORRECTED_FRAMES] == 13
    assert fixed.diagnostics["frames_relabelled"] == 13 and fixed.diagnostics["label_jumps_remaining"] == 0
    for column in (S.DPSI_SWEEP_DEG, S.DPSI_TOTAL_DEG, S.FRAC_HEADING_IN_SWEEP_SIGNED, S.T_SWEEP_END_S, S.T_END_S,
                   S.at_max_column(S.BEND_PEAK_ARC_FRACTION)):
        assert turn[column] == pytest.approx(clean[column], abs=1e-9), column

    excluded = T.stage_12(corrupted, events, body, dataclasses.replace(params, correct_swapped_labels=False), FPS)
    assert excluded.tables["turns"].iloc[0][S.EXCLUSION_REASON] == S.EXCLUDE_LABEL_JUMP


def test_two_frame_flicker_is_corrected_by_its_bracketing_jumps():
    track, events, body, _ = synthetic_turn_recording()
    params = TurnKinematicsParams(end_total_bend_deg=16.6)
    clean = T.stage_12(track, events, body, params, FPS).tables["turns"].iloc[0]
    flicker = _swap_rows(track, [58, 59])  # too short to orient by motion
    kinematics, pieces = T.prepare_recording(flicker, events, params, FPS)
    assert kinematics[S.LABELS_CORRECTED].sum() == 2
    fixed = T.stage_12(flicker, events, body, params, FPS).tables["turns"].iloc[0]
    assert fixed[S.INCLUDED], fixed[S.EXCLUSION_REASON]
    assert fixed[S.DPSI_SWEEP_DEG] == pytest.approx(clean[S.DPSI_SWEEP_DEG], abs=1e-9)
    first_pass = T.label_pieces(T.frame_kinematics(flicker, params, FPS), params)
    assert first_pass[S.ORIENTATION_BASIS].tolist() == [S.BASIS_MOTION, S.BASIS_BRACKETING, S.BASIS_MOTION]
    assert first_pass[S.ORIENTATION].tolist() == [S.ORIENTATION_CORRECT, S.ORIENTATION_SWAPPED, S.ORIENTATION_CORRECT]


def test_bracketing_does_not_guess_between_disagreeing_neighbours():
    # Crawl forward for 3 s, a 2-frame flicker, then a genuine reversal with
    # correct labels. Motion calls the reversal "swapped", so the neighbours imply
    # opposite orientations for the flicker and it must stay unresolved. The
    # reversal then has no correct neighbour across its jump, so it must not be
    # relabelled either: a real reversal is never "corrected".
    shape = spine_from_links((0, 0), np.pi, (0, 0, 0))
    x = np.r_[0.05 * np.arange(32), 0.05 * 31 - 0.05 * np.arange(1, 49)]
    points = np.array([shape + [xi, 0] for xi in x])
    points[30:32] = points[30:32, ::-1]
    params = TurnKinematicsParams(spine_sg_window=3, spine_sg_order=1)
    pieces = T.label_pieces(kinematics(points, spine_sg_window=3, spine_sg_order=1), params)
    assert pieces[S.ORIENTATION].tolist() == [S.ORIENTATION_CORRECT, S.ORIENTATION_UNRESOLVED, S.ORIENTATION_UNRESOLVED]
    assert pieces[S.ORIENTATION_BASIS].tolist() == [S.BASIS_MOTION, S.BASIS_UNRESOLVED, S.BASIS_UNANCHORED]


def test_heading_curves_run_from_zero_to_one_and_pass_through_the_signed_fraction():
    track, events, body, _ = synthetic_turn_recording()
    result = T.stage_12(track, events, body, TurnKinematicsParams(end_total_bend_deg=16.6), FPS)
    turn = result.tables["turns"].iloc[0]
    curves = result.tables["heading_curves"]
    mine = curves[curves[S.HEADING_AXIS] == "tail_sp2"]
    progress = mine[mine[S.CURVE] == S.CURVE_PROGRESS]
    assert progress[S.CURVE_X].iloc[0] == 0 and progress[S.CURVE_X].iloc[-1] == 1
    assert progress[S.FRACTION_OF_TOTAL].iloc[0] == pytest.approx(0)
    assert progress[S.FRACTION_OF_TOTAL].iloc[-1] == pytest.approx(1)
    at_sweep = np.interp(turn[S.SWEEP_END_PROGRESS], progress[S.CURVE_X], progress[S.FRACTION_OF_TOTAL])
    assert at_sweep == pytest.approx(turn[S.FRAC_HEADING_IN_SWEEP_SIGNED], abs=0.05)
    time = mine[mine[S.CURVE] == S.CURVE_TIME]
    assert time[S.CURVE_X].iloc[-1] == pytest.approx(turn[S.TURN_DURATION_S])
    assert time[S.DPSI_INTO_TURN_DEG].iloc[-1] == pytest.approx(abs(turn[S.DPSI_TOTAL_DEG]))
    distance = mine[mine[S.CURVE] == S.CURVE_DISTANCE]
    assert distance[S.CURVE_X].is_monotonic_increasing
    assert distance[S.CURVE_X].iloc[-1] <= turn[S.TURN_DISTANCE_SEGLEN]
    summary = T.curve_summary(curves, [S.TRACK_ID])
    assert (summary[S.N_TURNS] == 1).all()


def test_stage_12_requires_the_pooled_end_threshold_and_matching_candidates():
    track, events, body, _ = synthetic_turn_recording()
    with pytest.raises(ValueError, match="end_total_bend_deg"):
        T.stage_12(track, events, body, TurnKinematicsParams(), FPS)
    with pytest.raises(AssertionError, match="differ from stage 09"):
        T.stage_12(track, events, body, TurnKinematicsParams(end_total_bend_deg=16.6), FPS,
                   pd.Series({track[S.TRACK_ID].iloc[0]: 2}))


def test_end_not_found_when_the_body_never_straightens():
    track, events, body, _ = synthetic_turn_recording()
    result = T.stage_12(track, events, body, TurnKinematicsParams(end_total_bend_deg=0.0), FPS)
    turn = result.tables["turns"].iloc[0]
    assert turn[S.EXCLUSION_REASON] == S.EXCLUDE_END_NOT_FOUND and not turn[S.INCLUDED]


def test_sweep_runs_split_on_sign_and_release_with_hysteresis():
    head = np.array([0, 35, 40, 10, 31, -32, -40, 0, 0])
    assert T._sweep_runs(head, 30, 0.6) == [(1, 2), (4, 4), (5, 6)]
    # A dip below the threshold but above 60% of the running maximum does not end the sweep.
    assert T._sweep_runs(np.array([0, 35, 28, 45, 28, 20]), 30, 0.6) == [(1, 4)]


def test_small_turns_keep_their_row_but_not_their_fraction():
    assert np.isnan(T._fraction(5.0, 10.0, 15.0)[0])
    assert T._fraction(-5.0, 20.0, 15.0) == (0.25, -0.25)


def test_censoring_flags_a_filter_that_drops_larger_turns():
    rng = np.random.default_rng(3)
    small, large = rng.uniform(30, 60, 40), rng.uniform(100, 150, 20)
    turns = pd.DataFrame({
        S.CAST_PEAK_HEAD_ANGLE_DEG: np.r_[small, large],
        S.EXCLUSION_REASON: [None] * 40 + [S.EXCLUDE_LABEL_JUMP] * 20,
        S.DPSI_TOTAL_DEG: np.r_[small, large],
        S.FRAC_COMPUTED: [True] * 40 + [False] * 20,
        S.N_FLAGGED_FRAMES: 0,
    })
    report = T.censoring_report(turns).set_index(S.FILTER)
    assert report.loc[S.EXCLUDE_LABEL_JUMP, S.BIASED_TOWARD_LARGE_TURNS]
    assert report.loc[S.EXCLUDE_LABEL_JUMP, S.N_DROPPED] == 20
    assert not report.loc[S.EXCLUDE_COILED, S.BIASED_TOWARD_LARGE_TURNS]


def test_label_pieces_orient_by_motion_between_jumps():
    n = 60
    points = np.array([spine_from_links((0.05 * i, 0), np.pi, (0, 0, 0)) for i in range(n)])  # crawling +x, head-first
    points[20:35] = points[20:35, ::-1]  # labels swapped for 1.5 s
    k = kinematics(points, spine_sg_window=3, spine_sg_order=1)
    pieces = T.label_pieces(k, TurnKinematicsParams(spine_sg_window=3, spine_sg_order=1))
    assert pieces[S.ORIENTATION].tolist() == [S.ORIENTATION_CORRECT, S.ORIENTATION_SWAPPED, S.ORIENTATION_CORRECT]
    assert pieces[S.PIECE_START_S].tolist() == [0.0, 2.0, 3.5]


def test_params_reject_bad_values():
    with pytest.raises(ValueError):
        TurnKinematicsParams(spine_sg_window=6)
    with pytest.raises(ValueError):
        TurnKinematicsParams(heading_axis="head_tail")
    with pytest.raises(ValueError):
        TurnKinematicsParams(trail_fraction_of_peak=1.5)
