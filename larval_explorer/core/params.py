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
