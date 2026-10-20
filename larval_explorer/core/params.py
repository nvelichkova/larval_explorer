"""Stage parameters: one frozen dataclass per stage, upstream values as defaults.

A changed default silently changes the science (CLAUDE.md §5). Every default
here is the value the upstream script uses, including constants upstream wrote
inline in function bodies. The full resolved set is written to the run manifest.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

from larval_explorer.core import schema


@dataclass(frozen=True)
class ReorganizeParams:
    """Stage 00 - reorganise a raw FIM-Track export into a wide table."""

    measurements_to_keep: tuple[str, ...] = schema.REQUIRED_CHANNELS
    # Last frame index to keep. None keeps every frame in the file; upstream
    # hardcoded 36000 at the call site.
    max_frame: int | None = None


@dataclass(frozen=True)
class CalibrationParams:
    """Stage 01 - frames to seconds, pixels to millimetres."""

    frame_rate_fps: float = 10.0
    millimetres_per_pixel: float = 240.0 / 2048.0
    # Upstream rounds time to 0.1 s and positions to 0.001 mm.
    time_decimals: int = 1
    position_decimals: int = 3


@dataclass(frozen=True)
class SegmentationParams:
    """Stage 02 - split tracks into contiguous segments (D-008).

    With ``enabled=False`` the stage reproduces upstream script 01's filter:
    tracks are kept whole if their first-to-last time span reaches
    ``min_segment_seconds``, and gaps are ignored.
    """

    enabled: bool = True
    # Gaps of at most this many missing frames are interpolated; longer ones split.
    bridge_max_frames: int = 10
    min_segment_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.bridge_max_frames < 0:
            raise ValueError("bridge_max_frames must be zero or positive.")
        if self.min_segment_seconds < 0:
            raise ValueError("min_segment_seconds must be zero or positive.")


@dataclass(frozen=True)
class SmoothingParams:
    """Stage 03 - Savitzky-Golay smoothing of the centre of mass."""

    window_size: int = 41  # forced odd, as upstream does
    polynomial_order: int = 3


@dataclass(frozen=True)
class BodyLengthParams:
    """Stage 04 - mean body length per track. Upstream has no tunables."""


@dataclass(frozen=True)
class RdpParams:
    """Stage 05 - Ramer-Douglas-Peucker steps."""

    # epsilon = epsilon_factor * mean body length of the track.
    epsilon_factor: float = 0.5


@dataclass(frozen=True)
class HmmParams:
    """Stage 06 - 3-state Gaussian HMM on phi."""

    n_states: int = 3
    n_iter: int = 1000
    random_state: int = 42
    covariance_type: str = "diag"
    # Manual initialisation, in radians before standardisation. Upstream passes
    # the two "sigma" values to the model as variances; the names are kept.
    mu_phi: float = 0.8
    sigma_phi_fast: float = 0.15
    sigma_phi_mixed: float = 1.00

    def __post_init__(self) -> None:
        if self.n_states != 3:
            raise ValueError(
                "n_states must be 3: upstream initialises exactly three states "
                "(+mu_phi, -mu_phi, 0) and writes three posterior columns."
            )


@dataclass(frozen=True)
class HmmFilterParams:
    """Stage 07 - replace short HMM state runs."""

    min_run: int = 3


@dataclass(frozen=True)
class HmmSegmentParams:
    """Stage 08 - representative HMM segments."""

    min_steps: int = 9


@dataclass(frozen=True)
class EventParams:
    """Stage 09 - crawl and head-cast detection."""

    hc_bodyamp_min_deg: float = 30.0
    # Savitzky-Golay windows (frames) and polynomial order.
    sg_window_position: int = 21
    sg_window_spine_length: int = 7
    sg_window_angle: int = 7
    sg_window_angular_velocity: int = 11
    sg_polynomial_order: int = 3
    # scipy.signal.find_peaks prominence.
    spine_peak_prominence: float = 0.05
    angular_velocity_peak_prominence: float = 0.05
    # Two-line fit to the |angular velocity| histogram that sets the threshold.
    threshold_histogram_bins: int = 100
    threshold_split_margin_bins: int = 5
    threshold_scale: float = 1.1
    # A head cast spans this fraction of the way to the neighbouring opposite peaks.
    event_extent_fraction: float = 0.9
    merge_gap_seconds: float = 1.0
    min_crawl_seconds: float = 0.5


@dataclass(frozen=True)
class StepModelParams:
    """Stage 10 - event-level and run-anchor steps. Upstream has no tunables."""


def _default_kde_bw_min() -> dict[str, float]:
    return {"rho": 0.015, "kappa": 0.020, "mu": 0.020, "abs_mu": 0.020, "sigma": 0.010}


@dataclass(frozen=True)
class SteeringParams:
    """Stage 11 - AR(2) steering fit by HMM state."""

    min_events_per_segment: int = 5
    min_points_per_run: int = 8
    clip_outliers: bool = True
    clip_pcts: tuple[float, float] = (1, 99)
    clip_min_points: int = 10
    require_stable: bool = True
    rho_min: float = 0.0
    rho_max: float = 0.999
    kappa_min: float = 0.05
    mu_max: float | None = 0.5
    max_lag: int = 4
    min_events_per_seg_autocorr: int = 4
    # Bandwidth floors for the weighted KDE peaks in the state summary.
    kde_bw_min: Mapping[str, float] = field(default_factory=_default_kde_bw_min)
    crawl_length_kde_bw_min: float = 1e-3


HEADING_AXES = ("tail_sp3", "tail_sp2", "tail_sp1", "tail_mom", "tail_head")


@dataclass(frozen=True)
class TurnKinematicsParams:
    """Stage 12 - turn kinematics (docs/TURN_KINEMATICS.md, D-027).

    No upstream equivalent, so no default here is an upstream value; each was
    agreed with the user. Phase T1 holds the per-frame geometry only; the turn
    segmentation thresholds are added in T2, after the noise floor is reviewed.
    """

    # Temporal Savitzky-Golay smoothing of the integer-pixel spine points.
    spine_sg_window: int = 7
    spine_sg_order: int = 2
    # Primary biological axis, and the axis matching the model's heading (tail
    # to arc 3/11, D-027 Q3).
    heading_axis: str = "tail_sp2"
    model_heading_axis: str = "tail_sp1"
    sensitivity_axes: tuple[str, ...] = HEADING_AXES
    # Model body: 11 segments T1...A8; model boundary i sits at arc i/11.
    n_body_segments: int = 11
    # Bend trailing edge: posterior-most position carrying this fraction of the peak.
    trail_fraction_of_peak: float = 0.2
    # Head/tail labels: a track is split where the labels jump (no threshold,
    # D-030); each piece is ``correct`` if at least this share of the tail's
    # motion along the posterior axis is head-first, ``swapped`` if at most
    # 1 minus this share, otherwise ``unresolved``.
    piece_orientation_share: float = 0.7
    # Motion decides a piece's orientation only with at least this many usable
    # frames; shorter pieces (tracker flicker) are decided by their bracketing
    # jumps (D-033). One smoothing window.
    piece_min_motion_frames: int = 7
    # Relabel frames in swapped pieces instead of excluding the turns around
    # them (D-033). False gives the exclusion variant, for sensitivity.
    correct_swapped_labels: bool = True

    # ── Turn segmentation (phase T2; thresholds from the T1 noise floor, D-031) ──
    # Onset: the last frame at or before the cast start with |head angle| below this.
    onset_head_angle_deg: float = 15.0
    max_onset_lookback_s: float = 2.0
    # A sweep is a run of frames with |head angle| at or above this. 30 deg is the
    # 95th percentile of crawling |head angle|: about 5% of crawl frames cross it,
    # tolerable only because stage 09 also requires amplitude and duration. Do not
    # lower it.
    sweep_head_angle_deg: float = 30.0
    # A sweep continues while |head angle| stays above this fraction of its
    # running maximum (hysteresis, D-034); 0.6 is the synthetic generator's
    # own reader's value.
    sweep_release_fraction: float = 0.6
    # End: `end_hold_frames` consecutive frames with total bend below
    # `end_total_bend_deg` and forward tail speed above the speed threshold.
    # `end_total_bend_deg` is the pooled crawl percentile below, computed across
    # every recording in the analysis and passed in explicitly; None means "not
    # yet set" and the stage refuses to run.
    end_total_bend_deg: float | None = None
    end_total_bend_crawl_percentile: float = 25.0
    # Speed threshold relative to the larva's own median crawl forward speed, so
    # "has resumed crawling" means the same in fast and slow animals. The
    # absolute variant (body lengths per second) is the sensitivity check.
    end_speed_mode: str = "relative"
    end_speed_fraction_of_crawl: float = 0.25
    end_speed_absolute_bl_s: float = 0.1
    end_hold_frames: int = 3
    max_turn_duration_s: float = 15.0
    frame_padding_s: float = 1.0
    min_abs_dpsi_total_deg: float = 15.0
    # Bend-travel fit: frames from maximum bend to turn end with total bend at
    # least this (crawl 95th percentile, 51-59 deg).
    travel_min_total_bend_deg: float = 55.0
    travel_min_frames: int = 4
    travel_min_distance_seglen: float = 0.5
    # Heading curves (D-034): grids on which each turn's cumulative heading
    # change is sampled. Distance and time curves stop at the turn's end.
    curve_progress_step: float = 0.05
    curve_distance_step_seglen: float = 0.25
    curve_max_distance_seglen: float = 10.0

    def __post_init__(self) -> None:
        if self.spine_sg_window % 2 == 0 or self.spine_sg_window <= self.spine_sg_order:
            raise ValueError("spine_sg_window must be odd and larger than spine_sg_order.")
        for axis in (self.heading_axis, self.model_heading_axis, *self.sensitivity_axes):
            if axis not in HEADING_AXES:
                raise ValueError(f"Unknown heading axis '{axis}'. Axes are: {', '.join(HEADING_AXES)}")
        if self.n_body_segments < 2:
            raise ValueError("n_body_segments must be at least 2.")
        if not 0 < self.trail_fraction_of_peak < 1:
            raise ValueError("trail_fraction_of_peak must be between 0 and 1.")
        if self.end_speed_mode not in ("relative", "absolute"):
            raise ValueError("end_speed_mode must be 'relative' or 'absolute'.")
        if not 0.5 < self.piece_orientation_share <= 1:
            raise ValueError("piece_orientation_share must be above 0.5 and at most 1.")


@dataclass(frozen=True)
class PipelineParams:
    """The complete parameter set of one run, keyed by stage number."""

    stage_00: ReorganizeParams = field(default_factory=ReorganizeParams)
    stage_01: CalibrationParams = field(default_factory=CalibrationParams)
    stage_02: SegmentationParams = field(default_factory=SegmentationParams)
    stage_03: SmoothingParams = field(default_factory=SmoothingParams)
    stage_04: BodyLengthParams = field(default_factory=BodyLengthParams)
    stage_05: RdpParams = field(default_factory=RdpParams)
    stage_06: HmmParams = field(default_factory=HmmParams)
    stage_07: HmmFilterParams = field(default_factory=HmmFilterParams)
    stage_08: HmmSegmentParams = field(default_factory=HmmSegmentParams)
    stage_09: EventParams = field(default_factory=EventParams)
    stage_10: StepModelParams = field(default_factory=StepModelParams)
    stage_11: SteeringParams = field(default_factory=SteeringParams)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """JSON-ready ``{"00": {...}, "01": {...}, ...}`` for the manifest."""
        return {f.name.removeprefix("stage_"): params_to_dict(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Any]]) -> "PipelineParams":
        stages = {}
        for f in dataclasses.fields(cls):
            key = f.name.removeprefix("stage_")
            if key in data:
                stages[f.name] = params_from_dict(type(f.default_factory()), data[key])
        return cls(**stages)


def params_to_dict(params: Any) -> dict[str, Any]:
    """One stage's parameters as plain JSON types (tuples become lists)."""
    return {
        f.name: _to_plain(getattr(params, f.name))
        for f in dataclasses.fields(params)
    }


def params_from_dict(params_type: type, data: Mapping[str, Any]):
    """Rebuild a stage's parameters from ``params_to_dict`` output.

    Unknown keys raise, so a manifest written by a different version is not
    silently half-applied.
    """
    known = {f.name: f for f in dataclasses.fields(params_type)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown parameter(s) for {params_type.__name__}: {unknown}")
    defaults = params_type()
    values = {}
    for name, value in data.items():
        if isinstance(getattr(defaults, name), tuple) and value is not None:
            value = tuple(value)
        values[name] = value
    return params_type(**values)


def _to_plain(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    return value
