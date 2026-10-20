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
def test_onset_and_sweep_end_are_found_close_to_the_truth(seeded, condition):
    turns, truth, fps = seeded[condition]
    onset_err = turns[S.T_ONSET_S] - truth.true_onset_frame / fps
    sweep_err = turns[S.T_SWEEP_END_S] - truth.true_sweep_end_frame / fps
    # In frames: slow sweeps have a broad peak, and the median sweep-end error there is 2 frames.
    assert (onset_err * fps).abs().median() <= 2.5
    assert (sweep_err * fps).abs().median() <= 2.5
    assert turns[S.INCLUDED].mean() >= 0.6


def test_the_sign_at_sweep_end_is_recovered_on_the_model_axis(seeded):
    column = S.model_axis_column(S.FRAC_HEADING_IN_SWEEP_SIGNED)
    slow = seeded["slow_bend"][0]
    fast = seeded["fast_bend"][0]
    slow, fast = slow.loc[slow[S.INCLUDED], column].dropna(), fast.loc[fast[S.INCLUDED], column].dropna()
    # Truth: slow_bend never counter-rotates (0%), fast_bend nearly always (97%).
    assert (slow < 0).mean() < 0.1 and slow.median() > 0
    assert (fast < 0).mean() > 0.85 and fast.median() < 0
