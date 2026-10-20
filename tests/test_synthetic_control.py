"""Stage 12 against the synthetic recordings with a known answer (D-034).

Pins stage 12's definitions qualitatively, not to frozen values: with the true
sweeps given as head casts, stage 12 must find the onset and sweep end close to
the truth, and must recover the sign of the heading change at the sweep end on
the model-matched axis. The magnitude is not asserted: the reduction to five
spine points biases it differently in each condition (SYNTHETIC.md, D-034).
"""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from larval_explorer.core import schema as S
from larval_explorer.core import turns as T
from larval_explorer.core.params import TurnKinematicsParams

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "data" / "synthetic"


def load_tool():
    spec = importlib.util.spec_from_file_location("synthetic_control", REPO_ROOT / "tools" / "synthetic_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seeded():
    tool = load_tool()
    data = {c: tool.run_condition(FIXTURE, c) for c in ("slow_bend", "fast_bend")}
    base = TurnKinematicsParams()
    bends = []
    for d in data.values():
        k, _ = T.prepare_recording(d["trajectories"], d["events"], base, d["fps"])
        bends.append(T.crawl_total_bends(k, d["events"], k[S.ORIENTATION].to_numpy()))
    params = dataclasses.replace(base, end_total_bend_deg=float(np.percentile(np.concatenate(bends), 25)))
    out = {}
    for c, d in data.items():
        truth = d["truth"]
        casts = pd.DataFrame({S.TRACK_ID: truth[S.TRACK_ID], S.EVENT_TYPE: S.EVENT_HEAD_CAST,
                              S.START_S: truth.true_onset_frame / d["fps"], S.END_S: truth.true_sweep_end_frame / d["fps"]})
        events = pd.concat([d["events"][d["events"][S.EVENT_TYPE] == S.EVENT_CRAWL], casts], ignore_index=True)
        turns = T.stage_12(d["trajectories"], events, d["body"], params, d["fps"]).tables["turns"]
        turns = turns.sort_values([S.TRACK_ID, S.HEAD_CAST_START_S]).reset_index(drop=True)
        truth = truth.sort_values([S.TRACK_ID, "true_onset_frame"]).reset_index(drop=True)
        big = (truth.ideal_dpsi_total_deg.abs() >= 15).to_numpy()
        out[c] = (turns[big].reset_index(drop=True), truth[big].reset_index(drop=True), d["fps"])
    return out


@pytest.mark.parametrize("condition", ["slow_bend", "fast_bend"])
def test_onset_and_head_angle_peak_are_found_close_to_the_truth(seeded, condition):
    # The truth file's "sweep end" is the |head angle| maximum (D-035).
    turns, truth, fps = seeded[condition]
    onset_err = turns[S.T_ONSET_S] - truth.true_onset_frame / fps
    peak_err = turns[S.T_HEAD_ANGLE_PEAK_S] - truth.true_sweep_end_frame / fps
    # In frames: slow sweeps have a broad peak, and the median error there is 2 frames.
    assert (onset_err * fps).abs().median() <= 2.5
    assert (peak_err * fps).abs().median() <= 2.5
    assert turns[S.INCLUDED].mean() >= 0.6


# The user's independent measurement through the pixel data (2026-10-21): median
# time from onset to the sp1 and sp2 bend peaks.
BEND_PEAKS_MEASURED = {"slow_bend": (2.60, 4.30), "fast_bend": (1.70, 2.40)}


@pytest.mark.parametrize("condition", ["slow_bend", "fast_bend"])
def test_bend_peaks_reproduce_the_independent_measurement(seeded, condition):
    turns, _, _ = seeded[condition]
    kept = turns[turns[S.INCLUDED]]
    sp1, sp2 = BEND_PEAKS_MEASURED[condition]
    assert (kept[S.T_BEND_1_PEAK_S] - kept[S.T_ONSET_S]).median() == pytest.approx(sp1, abs=0.3)
    assert (kept[S.T_BEND_2_PEAK_S] - kept[S.T_ONSET_S]).median() == pytest.approx(sp2, abs=0.3)
    # The cast ends at the sp1 peak (D-036).
    assert (kept[S.T_SWEEP_END_S] == kept[S.T_BEND_1_PEAK_S]).all()


def test_faster_migration_gives_a_shorter_sp1_to_sp2_lag(seeded):
    slow = seeded["slow_bend"][0]
    fast = seeded["fast_bend"][0]
    slow, fast = slow.loc[slow[S.INCLUDED], S.BEND_LAG_12_S], fast.loc[fast[S.INCLUDED], S.BEND_LAG_12_S]
    assert slow.median() > fast.median() > 0
    assert (slow > 0).mean() > 0.9 and (fast > 0).mean() > 0.9


def test_the_sign_at_the_head_angle_peak_is_recovered_on_the_model_axis(seeded):
    # The truth defines its fraction at the |head angle| maximum, so the sign is scored there.
    column = S.model_axis_column(S.FRAC_SIGNED_AT_HEAD_ANGLE_PEAK)
    slow = seeded["slow_bend"][0]
    fast = seeded["fast_bend"][0]
    slow, fast = slow.loc[slow[S.INCLUDED], column].dropna(), fast.loc[fast[S.INCLUDED], column].dropna()
    # Truth: slow_bend never counter-rotates (0%), fast_bend nearly always (97%).
    assert (slow < 0).mean() < 0.1 and slow.median() > 0
    assert (fast < 0).mean() > 0.85 and fast.median() < 0


# The regenerated truth files (D-039) define the cast end (sp1 bend peak) and the
# sp1 -> sp2 lag the same way stage 12 does, so both can be scored per turn.

@pytest.mark.parametrize("condition", ["slow_bend", "fast_bend"])
def test_cast_end_matches_the_truth(seeded, condition):
    turns, truth, fps = seeded[condition]
    kept = turns[S.INCLUDED].to_numpy()
    error = (turns[S.T_SWEEP_END_S] - truth.t_cast_end_s).to_numpy()[kept] * fps
    assert np.median(np.abs(error)) <= 1.0


def _lag_error(seeded, condition):
    turns, truth, _ = seeded[condition]
    kept = turns[S.INCLUDED].to_numpy()
    return (turns[S.BEND_LAG_12_S] - truth.lag_sp1_sp2_s).to_numpy()[kept]


def test_lag_matches_the_truth_on_fast_bend(seeded):
    assert np.median(np.abs(_lag_error(seeded, "fast_bend"))) <= 0.25


@pytest.mark.xfail(strict=True, reason="D-039: on slow_bend the sp2 bend peak is mislocated in ~38% of turns "
                   "(broad plateau; cast sign taken from a small first sweep). Remove this mark once fixed.")
def test_lag_matches_the_truth_on_slow_bend(seeded):
    error = _lag_error(seeded, "slow_bend")
    assert np.median(np.abs(error)) <= 0.25 and np.mean(np.abs(error) > 0.5) <= 0.1
