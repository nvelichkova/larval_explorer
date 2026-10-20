"""Stage 12 across recordings (T3): pooled threshold, summary, refusals and loud failures."""

import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import schema as S
from larval_explorer.core import turn_analysis as TA
from larval_explorer.core.aggregate import AnalysisError

from tests.test_turns import FPS, spine_from_tail, track_from_points


def recording(recording_id: str, turn_deg: float, larvae=(0,), n_turns: int = 2) -> TA.RecordingTurnInputs:
    """Each larva crawls, then turns ``n_turns`` times by ``turn_deg``, 12 s apart."""
    tracks, events, body = [], [], []
    period = 120
    for larva in larvae:
        n = period * n_turns + 40
        t = np.arange(n) / FPS
        phi = np.zeros(n)
        # A slow crawling wiggle, so the crawl bend distribution (and the pooled
        # end threshold taken from it) is not identically zero.
        beta = np.deg2rad(8) * np.sin(2 * np.pi * t / 4.0)
        for k in range(n_turns):
            t0 = 5 + k * period / FPS
            x = np.clip((t - t0) / 1.5, 0, 1)
            phi += np.deg2rad(turn_deg) * (3 * x**2 - 2 * x**3)
            window = (t >= t0) & (t <= t0 + 2)
            beta += np.deg2rad(100) * np.where(window, np.sin(np.pi * (t - t0) / 2) ** 2, 0)
            events += [(S.EVENT_HEAD_CAST, t0 + 0.2, t0 + 1.8), (S.EVENT_CRAWL, t0 - 4.5, t0 - 0.5), (S.EVENT_CRAWL, t0 + 3, t0 + 6)]
        tail = np.zeros((n, 2))
        for i in range(1, n):
            tail[i] = tail[i - 1] + 0.05 * np.array([np.cos(phi[i - 1]), np.sin(phi[i - 1])])
        points = np.array([spine_from_tail(tail[i], phi[i], beta[i]) for i in range(n)])
        track_id = S.segment_track_id(recording_id, larva, 0)
        track = track_from_points(points, track_id=track_id)
        track[S.IS_COILED] = 0.0
        tracks.append(track)
        events = [(track_id, *e) if len(e) == 3 else e for e in events]
        body.append({S.TRACK_ID: track_id, S.BODY_LENGTH_MEAN: 3.4})
    events = pd.DataFrame(events, columns=[S.TRACK_ID, S.EVENT_TYPE, S.START_S, S.END_S])
    counts = events[events[S.EVENT_TYPE] == S.EVENT_HEAD_CAST].groupby(S.TRACK_ID).size()
    return TA.RecordingTurnInputs(recording_id, pd.concat(tracks, ignore_index=True), events, pd.DataFrame(body), counts, FPS)


def test_single_recording_conditions_get_numbers_but_no_contrast():
    inputs = [recording("a1", 40), recording("b1", 60)]
    analysis = TA.turn_analysis(inputs, {"a1": "A", "b1": "B"}, sensitivity_windows=())
    contrasts = analysis.tables["contrasts"]
    assert (contrasts[S.CONTRAST_STATUS] == S.CONTRAST_REFUSED_SINGLE_RECORDING).all()
    assert contrasts[S.DIFFERENCE_OF_LARVA_MEANS].isna().all()
    assert any("refused" in w for w in analysis.warnings)
    summary = analysis.tables["summary"]
    assert set(summary[S.CONDITION]) == {"A", "B"}
    assert (summary.loc[summary[S.SUBSET] == TA.SUBSET_ALL, S.N_RECORDINGS] == 1).all()


def test_contrast_is_reported_with_two_recordings_per_condition():
    inputs = [recording("a1", 40), recording("a2", 42), recording("b1", 60), recording("b2", 63)]
    conditions = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    analysis = TA.turn_analysis(inputs, conditions, sensitivity_windows=())
    row = analysis.tables["contrasts"].set_index(S.MEASURE).loc[S.ABS_DPSI_TOTAL_DEG]
    assert row[S.CONTRAST_STATUS] == S.CONTRAST_REPORTED
    assert row[S.DIFFERENCE_OF_LARVA_MEANS] == pytest.approx(20.5, abs=1.5)
    assert row[S.WITHIN_LARVA_SPREAD] in (True, False)


def test_identical_conditions_fail_loudly():
    a = recording("a1", 40)
    b = TA.RecordingTurnInputs("b1", a.trajectories.assign(**{S.TRACK_ID: a.trajectories[S.TRACK_ID].str.replace("a1", "b1")}),
                               a.events.assign(**{S.TRACK_ID: a.events[S.TRACK_ID].str.replace("a1", "b1")}),
                               a.body_length.assign(**{S.TRACK_ID: a.body_length[S.TRACK_ID].str.replace("a1", "b1")}),
                               a.head_cast_counts.rename(index=lambda i: i.replace("a1", "b1")), FPS)
    with pytest.raises(AnalysisError, match="identical statistics on every measure"):
        TA.turn_analysis([a, b], {"a1": "A", "b1": "B"}, sensitivity_windows=())


def test_larvae_are_keyed_on_recording_and_larva_and_summarised_at_both_levels():
    # Both recordings contain a larva 0: they are two larvae, not one.
    inputs = [recording("a1", 40, larvae=(0, 1)), recording("a2", 50, larvae=(0,))]
    analysis = TA.turn_analysis(inputs, sensitivity_windows=())
    turns = TA._headline_set(analysis.tables["turns"])
    assert turns[S.LARVA_ID].nunique() == 3
    row = analysis.tables["summary"].set_index([S.SUBSET, S.MEASURE]).loc[(TA.SUBSET_ALL, S.ABS_DPSI_TOTAL_DEG)]
    assert row[S.N_LARVAE] == 3 and row[S.N_RECORDINGS] == 2 and row[S.N_TURNS] == len(turns)
    assert row[S.LARVA_MEAN] == pytest.approx(turns.groupby(S.LARVA_ID)[S.ABS_DPSI_TOTAL_DEG].mean().mean())
    assert row[S.TURN_MEAN] == pytest.approx(turns[S.ABS_DPSI_TOTAL_DEG].mean())
    grouping = analysis.tables["grouping"]
    assert grouping.set_index(S.VARIABLE).loc[S.CONDITION, S.N_LARVAE] == 3


def test_pooled_threshold_is_recorded_with_each_recordings_own_value(tmp_path):
    inputs = [recording("a1", 40), recording("a2", 50)]
    analysis = TA.turn_analysis(inputs, sensitivity_windows=(5,))
    assert set(analysis.end_threshold_per_recording) == {"a1", "a2"}
    assert analysis.params.end_total_bend_deg is not None
    windows = set(analysis.tables["smoothing_sensitivity"][S.SPINE_SG_WINDOW])
    assert windows == {7, 5}
    manifest = TA.export_turn_analysis(tmp_path / "out", analysis)
    text = manifest.read_text(encoding="utf-8")
    assert "end_total_bend_deg_pooled" in text and "summary.csv" in text
