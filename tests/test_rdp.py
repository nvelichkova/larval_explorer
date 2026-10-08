"""The iterative RDP returns exactly the vertices of upstream's recursive RDP (D-006).

If this fails the rewrite is wrong; the expected values are never adjusted.
"""

import numpy as np
import pandas as pd
import pytest

from larval_explorer.core.stages import rdp_iterative
from tests.conftest import BASELINE_DIR, load_legacy

recursive_rdp = load_legacy("05_extract_rdp_steps.py").rdp


def assert_identical(x, y, t, epsilon):
    assert rdp_iterative(x, y, t, epsilon) == recursive_rdp(x, y, t, epsilon)


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.3, 1.5, 50.0])
def test_identical_on_random_walks(epsilon):
    rng = np.random.default_rng(7)
    for n in (0, 1, 2, 3, 4, 17, 400):
        steps = rng.normal(size=(n, 2))
        x, y = np.cumsum(steps[:, 0]).tolist(), np.cumsum(steps[:, 1]).tolist()
        assert_identical(x, y, list(range(n)), epsilon)


@pytest.mark.parametrize("epsilon", [0.0, 0.5])
def test_identical_on_degenerate_shapes(epsilon):
    t = list(range(9))
    assert_identical([1.0] * 9, [2.0] * 9, t, epsilon)                           # a stationary animal
    assert_identical([float(i) for i in range(9)], [0.0] * 9, t, epsilon)        # a straight line
    assert_identical([0, 1, 2, 3, 0, 1, 2, 3, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0], t, epsilon)  # closed loop, tied distances
    assert_identical([0, 1, 1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0], t, epsilon)  # repeated points
    assert_identical([0.0, float("nan"), 2.0, 3.0, 4.0], [0.0, 1.0, float("nan"), 0.5, 0.0], t[:5], epsilon)


def test_identical_on_every_baseline_trajectory():
    trajectories = pd.read_csv(BASELINE_DIR / "03_trajectory_timeseries.csv.gz")
    body_lengths = pd.read_csv(BASELINE_DIR / "04_mean_body_length_by_trajectory.csv.gz").set_index("larva")
    for larva, track in trajectories.groupby("larva"):
        x, y, t = track["mom_x"].tolist(), track["mom_y"].tolist(), track["time"].tolist()
        for factor in (0.05, 0.5, 2.0):
            assert_identical(x, y, t, factor * body_lengths.loc[larva, "average_body_length"])


def test_long_straight_track_does_not_exhaust_the_stack():
    # A zig-zag forces one split per point: recursion depth ~n in the upstream version.
    n = 20000
    x = [float(i) for i in range(n)]
    y = [float(i % 2) * (1 + i * 1e-6) for i in range(n)]
    sx, _, _ = rdp_iterative(x, y, x, 0.01)
    assert len(sx) == n


def test_negative_epsilon_is_rejected():
    with pytest.raises(ValueError):
        rdp_iterative([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [0, 1, 2], -1.0)
