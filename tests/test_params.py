import dataclasses
import json

import pytest

from larval_explorer.core import params as P
from tests.conftest import load_legacy

# SPEC §7 and the constants in legacy/upstream_scripts. A change here changes the science.
UPSTREAM_DEFAULTS = {
    "01": {"frame_rate_fps": 10.0, "millimetres_per_pixel": 240.0 / 2048.0, "time_decimals": 1, "position_decimals": 3},
    "02": {"enabled": True, "bridge_max_frames": 10, "min_segment_seconds": 60.0},
    "03": {"window_size": 41, "polynomial_order": 3},
    "05": {"epsilon_factor": 0.5},
    "06": {"n_states": 3, "n_iter": 1000, "random_state": 42, "covariance_type": "diag",
           "mu_phi": 0.8, "sigma_phi_fast": 0.15, "sigma_phi_mixed": 1.0},
    "07": {"min_run": 3},
    "08": {"min_steps": 9},
    "09": {"hc_bodyamp_min_deg": 30.0, "sg_window_position": 21, "sg_window_spine_length": 7,
           "sg_window_angle": 7, "sg_window_angular_velocity": 11, "sg_polynomial_order": 3,
           "spine_peak_prominence": 0.05, "angular_velocity_peak_prominence": 0.05,
           "threshold_histogram_bins": 100, "threshold_split_margin_bins": 5, "threshold_scale": 1.1,
           "event_extent_fraction": 0.9, "merge_gap_seconds": 1.0, "min_crawl_seconds": 0.5},
    "11": {"min_events_per_segment": 5, "min_points_per_run": 8, "clip_outliers": True, "clip_pcts": [1, 99],
           "clip_min_points": 10, "require_stable": True, "rho_min": 0.0, "rho_max": 0.999,
           "kappa_min": 0.05, "mu_max": 0.5, "max_lag": 4, "min_events_per_seg_autocorr": 4,
           "kde_bw_min": {"rho": 0.015, "kappa": 0.020, "mu": 0.020, "abs_mu": 0.020, "sigma": 0.010},
           "crawl_length_kde_bw_min": 1e-3},
}


def test_defaults_match_upstream():
    resolved = P.PipelineParams().to_dict()
    for stage, expected in UPSTREAM_DEFAULTS.items():
        assert resolved[stage] == expected, stage
    assert len(resolved["00"]["measurements_to_keep"]) == 20
    assert resolved["00"]["max_frame"] is None


def test_measurements_match_upstream_script_00_order():
    upstream = load_legacy("00_reorganize_raw_tracking_tables.py").MEASUREMENTS_TO_KEEP
    assert list(P.ReorganizeParams().measurements_to_keep) == upstream


def test_params_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        P.SmoothingParams().window_size = 5


def test_manifest_round_trip_through_json():
    original = P.PipelineParams(
        stage_02=P.SegmentationParams(bridge_max_frames=4),
        stage_11=P.SteeringParams(clip_pcts=(2, 98), mu_max=None),
    )
    restored = P.PipelineParams.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="Unknown parameter"):
        P.params_from_dict(P.RdpParams, {"epsilon": 1.0})


def test_hmm_is_fixed_at_three_states():
    with pytest.raises(ValueError, match="n_states must be 3"):
        P.HmmParams(n_states=4)
