"""Canonical column names, the upstream-name adapter, and FIM-Track validation.

This is the only module allowed to spell a column name as a string literal
(CLAUDE.md §4, D-004). Everything else references the constants defined here.

Inside the package tables use canonical snake_case names. CSVs written to the
user's output tree keep the original upstream names, some of them Portuguese,
because those names are tied to the manuscript. Each table kind has a
``TableSchema`` that converts in both directions and validates on the way in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import pandas as pd


class SchemaError(ValueError):
    """A table does not have the columns, labels or values its schema requires."""


# ──────────────────────────────────────────────────────────────────────────────
# Canonical column names
# ──────────────────────────────────────────────────────────────────────────────

# Keys
TRACK_ID = "track_id"
FRAME = "frame"
TIME_S = "time_s"

# FIM-Track measurement channels. The names are the tracker's own and are
# already snake_case, so canonical and upstream spellings coincide.
MOM_X, MOM_Y = "mom_x", "mom_y"
HEAD_X, HEAD_Y = "head_x", "head_y"
SPINEPOINT_1_X, SPINEPOINT_1_Y = "spinepoint_1_x", "spinepoint_1_y"
SPINEPOINT_2_X, SPINEPOINT_2_Y = "spinepoint_2_x", "spinepoint_2_y"
SPINEPOINT_3_X, SPINEPOINT_3_Y = "spinepoint_3_x", "spinepoint_3_y"
TAIL_X, TAIL_Y = "tail_x", "tail_y"
PERIMETER, AREA, SPINE_LENGTH = "perimeter", "area", "spine_length"
RADIUS_1, RADIUS_2, RADIUS_3 = "radius_1", "radius_2", "radius_3"
IS_COILED, IS_WELL_ORIENTED = "is_coiled", "is_well_oriented"

# Pixel/millimetre coordinates, in upstream script 01's order.
POSITION_CHANNELS = (
    MOM_X, MOM_Y,
    HEAD_X, HEAD_Y,
    SPINEPOINT_1_X, SPINEPOINT_1_Y,
    SPINEPOINT_2_X, SPINEPOINT_2_Y,
    SPINEPOINT_3_X, SPINEPOINT_3_Y,
    TAIL_X, TAIL_Y,
)

# The 20 channels the pipeline keeps, in upstream script 00's order.
REQUIRED_CHANNELS = POSITION_CHANNELS + (
    PERIMETER, AREA, SPINE_LENGTH,
    RADIUS_1, RADIUS_2, RADIUS_3,
    IS_COILED, IS_WELL_ORIENTED,
)

# Binary tracker flags: never interpolated linearly.
FLAG_CHANNELS = (IS_COILED, IS_WELL_ORIENTED)

# Added by stage 02 (track segmentation). No upstream equivalent.
SOURCE_TRACK_ID = "source_track_id"
SEGMENT_INDEX = "segment_index"
INTERPOLATED = "interpolated"

# Stage 02 summary tables (one row per segment, one row per larva).
LARVA_INDEX = "larva_index"
FIRST_FRAME = "first_frame"
LAST_FRAME = "last_frame"
N_FRAMES_TRACKED = "n_frames_tracked"
N_FRAMES_BRIDGED = "n_frames_bridged"
KEPT = "kept"
TRACKED_FRACTION = "tracked_fraction"
GAP_COUNT = "gap_count"
LONGEST_GAP_FRAMES = "longest_gap_frames"
SEGMENTS_FOUND = "segments_found"
SEGMENTS_KEPT = "segments_kept"
KEPT_DURATION_S = "kept_duration_s"

# Aggregation across recordings (Phase 5). No upstream equivalent.
RECORDING_ID = "recording_id"
LARVA_ID = "larva_id"
UNIT_ID = "unit_id"
CONDITION = "condition"
MEASURE = "measure"
VALUE = "value"
WEIGHT = "weight"
PARAMETER = "parameter"
TRACKED_S = "tracked_s"
SECONDS = "seconds"
DWELL_S = "dwell_s"
N_RECORDINGS = "n_recordings"
N_LARVAE = "n_larvae"
N_TRACK_SEGMENTS = "n_track_segments"
N_UNITS = "n_units"
FIT = "fit"
MEAN_PHI = "mean_phi"
CONVERGED = "converged"

# Stage 04
BODY_LENGTH_MEAN = "body_length_mean"
BODY_LENGTH_STD = "body_length_std"
SOURCE_TABLE = "source_table"

# Step tables (stages 05-07 and 10)
T_START_S = "t_start_s"
T_END_S = "t_end_s"
X0_MM, Y0_MM = "x0_mm", "y0_mm"
X1_MM, Y1_MM = "x1_mm", "y1_mm"
DX_MM, DY_MM = "dx_mm", "dy_mm"
DURATION_S = "duration_s"
VELOCITY = "velocity"
STEP_LENGTH = "step_length"
THETA = "theta"
# Upstream uses the name "phi" for two different quantities. In the RDP step
# table (05-07) it is the signed turning persistence the HMM is fitted on. In
# the stage 10 step tables it is the absolute heading atan2(dY, dX).
PHI = "phi"
HEADING = "heading"
EPSILON_VALUE_MM = "epsilon_value_mm"
EPSILON_FACTOR = "epsilon_factor"

# HMM (stages 06-08)
STATE = "state"
STATE_FILTERED = "state_filtered"
START_INDEX = "start_index"
END_INDEX = "end_index"
START_TIME_S = "start_time_s"
END_TIME_S = "end_time_s"
N_STEPS = "n_steps"

# Events (stages 09-10)
EVENT_TYPE = "event_type"
START_S = "start_s"
END_S = "end_s"
CAST_THETA = "cast_theta"

EVENT_CRAWL = "crawl"
EVENT_HEAD_CAST = "head_cast"


# Stage 09 per-frame signals and per-track thresholds. No upstream table; these
# feed the diagnostic figures upstream drew inside its detection loop.
COM_X_SMOOTH, COM_Y_SMOOTH = "com_x_smooth", "com_y_smooth"
ANTERIOR_ANGULAR_VELOCITY = "anterior_angular_velocity"
POSTERIOR_ANGULAR_VELOCITY = "posterior_angular_velocity"
HEAD_SPEED, TAIL_SPEED, COM_SPEED = "head_speed", "tail_speed", "com_speed"
BODY_ANGLE = "body_angle"
SPINE_LENGTH_SMOOTH = "spine_length_smooth"
MEAN_SPINE_LENGTH = "mean_spine_length"
ANTERIOR_THRESHOLD = "anterior_threshold"
POSTERIOR_THRESHOLD = "posterior_threshold"
N_CRAWLS = "n_crawls"
N_HEAD_CASTS = "n_head_casts"


def threshold_fit_column(side: str, part: str) -> str:
    """Column holding one number of the two-line threshold fit.

    ``side`` is "anterior" or "posterior"; ``part`` is one of "split_bin",
    "low_slope", "low_intercept", "high_slope", "high_intercept".
    """
    return f"{side}_fit_{part}"


# Stage 11 outputs. Upstream already uses these names, apart from the track ID.
SEGMENT_ID = "segment_id"
T0_SEG, TF_SEG = "t0_seg", "tf_seg"
RUN_ID_IN_SEG = "run_id_in_seg"
N_POINTS = "n_points"
RHO, KAPPA, MU, SIGMA, ABS_MU = "rho", "kappa", "mu", "sigma", "abs_mu"
AR_A1, AR_A2, AR_C = "a1", "a2", "c"
THETA_MEAN, THETA_STD = "theta_mean", "theta_std"
T0_RUN, TF_RUN = "t0_run", "tf_run"
N_EVENTS_IN_SEG = "n_events_in_seg"
N_RUNS_FIT = "n_runs_fit"
N_POINTS_TOTAL = "n_points_total"
SEG_DURATION_S = "seg_duration_s"
N_HEADCASTS = "n_headcasts"
HC_RATE_HZ = "hc_rate_hz"
N_HC_DIR_VALID = "n_hc_dir_valid"
N_THETA_HC_CSV = "n_theta_hc_csv"
KIND = "kind"
LAG = "lag"
MEAN, STD = "mean", "std"
N_ANIMALS = "n_animals"
N_SEGMENTS = "n_segments"
WEIGHT_SUM = "weight_sum"
SEG_CORR, SEG_CNT = "seg_corr", "seg_cnt"
ANIMAL = "animal"
CRAWL_LENGTH = "crawl_length"
CRAWL_LENGTH_MEAN = "crawl_length_mean"
CRAWL_LENGTH_STD = "crawl_length_std"
CRAWL_LENGTH_PEAK = "crawl_length_peak"
P_HEADCAST_GIVEN_CRAWL = "p_headcast_given_crawl"
HEADCAST_CORR_LAG1_MEAN = "headcast_corr_lag1_mean"
HEADCAST_CORR_LAG1_STD = "headcast_corr_lag1_std"
HEADCAST_THETA_MU = "headcast_theta_mu"
HEADCAST_THETA_PEAK = "headcast_theta_peak"
HEADCAST_THETA_KAPPA = "headcast_theta_kappa"
HEADCAST_THETA_R = "headcast_theta_R"
N_CRAWLS_IN_SEGMENTS = "n_crawls_in_segments"
N_HEADCASTS_IN_SEGMENTS = "n_headcasts_in_segments"
N_HEADCAST_THETA_POINTS = "n_headcast_theta_points"

STEERING_PARAMETERS = (RHO, KAPPA, MU, SIGMA)
STEERING_SUMMARY_PARAMETERS = (RHO, KAPPA, MU, ABS_MU, SIGMA)

AUTOCORR_STEERING = "steering"
AUTOCORR_HEADCAST = "headcast_theta_csv"


def within_segment_std(parameter: str) -> str:
    return f"{parameter}_std_withinseg"


def weighted_mean_column(parameter: str) -> str:
    return f"{parameter}_mean_w"


def weighted_std_column(parameter: str) -> str:
    return f"{parameter}_std_w"


def weighted_peak_column(parameter: str) -> str:
    return f"{parameter}_peak_w"


def state_probability(state_index: int) -> str:
    """Canonical name of the posterior-probability column for one HMM state."""
    return f"p_state{state_index}"


def _upstream_state_probability(state_index: int) -> str:
    return f"p_est{state_index}"


# ──────────────────────────────────────────────────────────────────────────────
# Table schemas
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TableSchema:
    """Canonical <-> upstream column mapping for one kind of table.

    ``columns`` maps canonical name to upstream name. Columns not listed pass
    through both directions unchanged, so extra metadata is never dropped.
    ``required`` lists the canonical columns that must be present.
    """

    name: str
    columns: Mapping[str, str]
    required: tuple[str, ...] = ()
    _to_canonical: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        reverse = {upstream: canonical for canonical, upstream in self.columns.items()}
        if len(reverse) != len(self.columns):
            raise ValueError(f"Schema '{self.name}' maps two canonical columns to one upstream name.")
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        object.__setattr__(self, "_to_canonical", MappingProxyType(reverse))

    def upstream_name(self, canonical: str) -> str:
        return self.columns.get(canonical, canonical)

    def to_canonical(self, table: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Rename upstream columns to canonical names and validate."""
        renamed = table.rename(columns=dict(self._to_canonical))
        if validate:
            self.validate(renamed)
        return renamed

    def to_upstream(self, table: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Rename canonical columns back to the names upstream CSVs use."""
        if validate:
            self.validate(table)
        return table.rename(columns=dict(self.columns))

    def missing_columns(self, table: pd.DataFrame) -> list[str]:
        return [column for column in self.required if column not in table.columns]

    def validate(self, table: pd.DataFrame) -> None:
        """Raise ``SchemaError`` naming every required column that is absent."""
        missing = self.missing_columns(table)
        if missing:
            wanted = ", ".join(f"'{column}' (upstream '{self.upstream_name(column)}')" for column in missing)
            raise SchemaError(
                f"Table '{self.name}' is missing required column(s): {wanted}. "
                f"Columns present: {list(table.columns)}"
            )


_CHANNELS = {channel: channel for channel in REQUIRED_CHANNELS}

# Stage 00 output: one row per (track, frame), pixels, frame index in "time".
RAW_WIDE = TableSchema(
    name="raw_wide",
    columns={TRACK_ID: "larva", FRAME: "time", **_CHANNELS},
    required=(TRACK_ID, FRAME) + REQUIRED_CHANNELS,
)

# Stages 01-03 output: one row per (track, time), millimetres and seconds.
TRAJECTORY = TableSchema(
    name="trajectory",
    columns={TRACK_ID: "larva", TIME_S: "time", **_CHANNELS},
    required=(TRACK_ID, TIME_S) + REQUIRED_CHANNELS,
)

BODY_LENGTH = TableSchema(
    name="body_length",
    columns={
        TRACK_ID: "larva",
        BODY_LENGTH_MEAN: "average_body_length",
        BODY_LENGTH_STD: "std_dev_body_length",
        SOURCE_TABLE: "file_name",
    },
    required=(TRACK_ID, BODY_LENGTH_MEAN),
)

_STEP_COLUMNS = {
    TRACK_ID: "animal ID",
    T_START_S: "tempo_inicial (s)",
    T_END_S: "tempo_final (s)",
    X0_MM: "X0 (mm)",
    Y0_MM: "Y0 (mm)",
    X1_MM: "X (mm)",
    Y1_MM: "Y (mm)",
    DX_MM: "dX",
    DY_MM: "dY",
    DURATION_S: "duracao (s)",
    VELOCITY: "velocity",
    STEP_LENGTH: "step_length",
    THETA: "theta",
    EPSILON_VALUE_MM: "epsilon_value_mm",
    EPSILON_FACTOR: "epsilon_factor",
}
_STEP_REQUIRED = (TRACK_ID, T_START_S, T_END_S, X0_MM, Y0_MM, X1_MM, Y1_MM, DX_MM, DY_MM, THETA)

RDP_STEPS = TableSchema(
    name="rdp_steps",
    columns={**_STEP_COLUMNS, PHI: "phi"},
    required=_STEP_REQUIRED + (PHI,),
)

# Upstream script 06 writes exactly three posterior columns.
_HMM_COLUMNS = {
    **_STEP_COLUMNS,
    PHI: "phi",
    STATE: "estado",
    **{state_probability(k): _upstream_state_probability(k) for k in range(3)},
}

HMM_STEPS = TableSchema(
    name="hmm_steps",
    columns=_HMM_COLUMNS,
    required=_STEP_REQUIRED + (PHI, STATE),
)

HMM_STEPS_FILTERED = TableSchema(
    name="hmm_steps_filtered",
    columns={**_HMM_COLUMNS, STATE_FILTERED: "estado_filtrado"},
    required=_STEP_REQUIRED + (PHI, STATE, STATE_FILTERED),
)

# Stage 08 output. Apart from the ID, upstream already uses these names; they
# are the names stage 11 reads (D-014).
HMM_SEGMENTS = TableSchema(
    name="hmm_segments",
    columns={TRACK_ID: "animal ID"},
    required=(TRACK_ID, STATE, START_INDEX, END_INDEX, START_TIME_S, END_TIME_S, N_STEPS, DURATION_S),
)

EVENTS = TableSchema(
    name="events",
    columns={TRACK_ID: "ID", EVENT_TYPE: "type", START_S: "start", END_S: "end"},
    required=(TRACK_ID, EVENT_TYPE, START_S, END_S),
)

RUN_ANCHOR_STEPS = TableSchema(
    name="run_anchor_steps",
    columns={**_STEP_COLUMNS, HEADING: "phi"},
    required=_STEP_REQUIRED + (HEADING,),
)

EVENT_LEVEL_STEPS = TableSchema(
    name="event_level_steps",
    columns={**_STEP_COLUMNS, HEADING: "phi", EVENT_TYPE: "tipo"},
    required=_STEP_REQUIRED + (HEADING, EVENT_TYPE),
)


# Stage 11 tables that carry a track ID. The state-level tables
# (autocorrelation, state summary, simulation parameters) need no renaming.
RUN_FITS = TableSchema(
    name="run_fits",
    columns={TRACK_ID: "animal ID"},
    required=(SEGMENT_ID, TRACK_ID, STATE, RUN_ID_IN_SEG, N_POINTS) + STEERING_PARAMETERS,
)

SEGMENT_FITS = TableSchema(
    name="segment_fits",
    columns={TRACK_ID: "animal ID"},
    required=(SEGMENT_ID, TRACK_ID, STATE, N_POINTS_TOTAL) + STEERING_PARAMETERS,
)

HEADCAST_SEGMENT_METRICS = TableSchema(
    name="headcast_segment_metrics",
    columns={TRACK_ID: "animal ID"},
    required=(SEGMENT_ID, TRACK_ID, STATE, N_HEADCASTS, HC_RATE_HZ),
)

STATE_TABLE = TableSchema(name="state_table", columns={}, required=(STATE,))


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory column sniffer (folded in from upstream script 10, D-004)
# ──────────────────────────────────────────────────────────────────────────────

_ID_CANDIDATES = ["larva", "larva_id", "id", "track", "track_id"]
_TIME_CANDIDATES = ["time", "t", "timestamp", "tempo", "tempo_s"]
_POSITION_PAIRS = [("mom_x", "mom_y"), ("com_x", "com_y"), ("x", "y"), ("X", "Y")]


def detect_trajectory_columns(table: pd.DataFrame) -> tuple[str, str, str, str]:
    """Find the id, time, x and y columns of a per-frame trajectory table.

    Same candidates and precedence as upstream ``detect_columns_traj``; raises
    ``SchemaError`` where upstream raised ``ValueError``.
    """
    id_col = next((c for c in table.columns if c.lower() in _ID_CANDIDATES), None)
    if id_col is None:
        raise SchemaError(f"Could not find larva id column in trajectory (tried: {_ID_CANDIDATES})")
    time_col = next((c for c in table.columns if c.lower() in _TIME_CANDIDATES), None)
    if time_col is None:
        raise SchemaError(f"Could not find time column in trajectory (tried: {_TIME_CANDIDATES})")
    cols_lower = {c.lower(): c for c in table.columns}
    for a, b in _POSITION_PAIRS:
        if a in cols_lower and b in cols_lower:
            return id_col, time_col, cols_lower[a], cols_lower[b]
    raise SchemaError("Could not find position columns in trajectory (looked for mom/com/x,y).")


# ──────────────────────────────────────────────────────────────────────────────
# Track identifiers
# ──────────────────────────────────────────────────────────────────────────────

# Under D-010 a recording is never concatenated with another, so the recording
# index upstream script 00 embeds in its track IDs is always 0 (D-015).
UPSTREAM_VIDEO_INDEX = 0

_LARVA_LABEL = re.compile(r"^larva\((\d+)\)$")
_TRAILING_INDEX = re.compile(r"(\d+)\)?$")


def larva_index(track_id: str) -> int:
    """Larva number from ``larva(7)``, ``trajectory_0_7`` or any ID ending in digits."""
    match = _TRAILING_INDEX.search(str(track_id))
    if match is None:
        raise SchemaError(f"Cannot read a larva number from track ID '{track_id}'.")
    return int(match.group(1))


def upstream_track_id(larva_label: str) -> str:
    """``larva(N)`` -> ``trajectory_0_N``: the ID scheme of the published data (D-015)."""
    return f"trajectory_{UPSTREAM_VIDEO_INDEX}_{larva_index(larva_label)}"


def segment_track_id(recording_id: str, larva: int, segment: int) -> str:
    """ID of one contiguous track segment from stage 02 (D-008)."""
    return f"{recording_id}__larva{larva}__seg{segment}"


# ──────────────────────────────────────────────────────────────────────────────
# FIM-Track export validation
# ──────────────────────────────────────────────────────────────────────────────

_ROW_LABEL = re.compile(r"^(?P<channel>.+)\((?P<frame>\d+)\)$")


@dataclass(frozen=True)
class FimTrackReport:
    """What a raw FIM-Track export contains, and whether the pipeline can use it."""

    larvae: tuple[str, ...]
    n_frames: int
    first_frame: int | None
    last_frame: int | None
    channels: tuple[str, ...]
    missing_channels: tuple[str, ...]
    extra_channels: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def duration_seconds(self, frame_rate_fps: float) -> float:
        return self.n_frames / frame_rate_fps

    def summary(self) -> str:
        lines = [
            f"{len(self.larvae)} larvae, {self.n_frames} frames "
            f"(frames {self.first_frame}-{self.last_frame}), {len(self.channels)} channels",
        ]
        lines += [f"ERROR: {message}" for message in self.errors]
        lines += [f"warning: {message}" for message in self.warnings]
        if self.extra_channels:
            lines.append(f"ignored channels: {', '.join(self.extra_channels)}")
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise SchemaError("Not a usable FIM-Track export:\n" + "\n".join(self.errors))


def parse_fimtrack_labels(labels: pd.Index) -> pd.DataFrame:
    """Split row labels ``<channel>(<frame>)`` into channel and frame columns.

    Labels that do not match get a missing channel and frame.
    """
    parsed = pd.Series(labels, dtype="string").str.extract(_ROW_LABEL)
    parsed[FRAME] = pd.to_numeric(parsed.pop("frame"), errors="coerce").astype("Int64")
    return parsed


def validate_fimtrack(raw: pd.DataFrame) -> FimTrackReport:
    """Check a raw FIM-Track export read with its first column as the index.

    ``raw`` has one column per larva (``larva(0)`` ...) and one row per
    ``<channel>(<frame>)``. Missing values are empty cells (NaN once read).
    Extra channels are accepted silently; a missing required channel, a
    malformed layout or non-numeric data is an error. Never raises ``KeyError``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    larvae = tuple(str(column) for column in raw.columns)
    if not larvae:
        errors.append("No larva columns found.")
    bad_columns = [column for column in larvae if not _LARVA_LABEL.match(column)]
    if bad_columns:
        errors.append(f"Column headers are not of the form 'larva(N)': {bad_columns[:5]}")
    if len(set(larvae)) != len(larvae):
        errors.append("Duplicate larva columns.")

    non_numeric = [str(c) for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
    if non_numeric:
        errors.append(f"Non-numeric values in column(s): {non_numeric[:5]}")

    labels = parse_fimtrack_labels(raw.index)
    malformed = labels["channel"].isna()
    if malformed.any():
        examples = [str(label) for label in raw.index[malformed.to_numpy()][:5]]
        errors.append(
            f"{int(malformed.sum())} row label(s) are not of the form '<channel>(<frame>)', e.g. {examples}"
        )
    labels = labels[~malformed]

    channels = tuple(sorted(labels["channel"].unique()))
    missing = tuple(channel for channel in REQUIRED_CHANNELS if channel not in channels)
    extra = tuple(channel for channel in channels if channel not in REQUIRED_CHANNELS)
    if missing:
        errors.append(f"Missing required channel(s): {', '.join(missing)}")

    if labels.duplicated().any():
        errors.append(f"{int(labels.duplicated().sum())} duplicated '<channel>(<frame>)' row label(s).")

    required_rows = labels[labels["channel"].isin(REQUIRED_CHANNELS)]
    frames = required_rows[FRAME]
    first_frame = int(frames.min()) if len(frames) else None
    last_frame = int(frames.max()) if len(frames) else None
    n_frames = int(frames.nunique())
    if n_frames == 0:
        errors.append("No frames found for the required channels.")
    else:
        if n_frames != last_frame - first_frame + 1:
            warnings.append(
                f"Frame numbering has holes: {n_frames} distinct frames between {first_frame} and {last_frame}."
            )
        per_channel = required_rows.groupby("channel")[FRAME].nunique()
        uneven = per_channel[per_channel != n_frames]
        if len(uneven):
            errors.append(
                "Required channels cover different numbers of frames: "
                + ", ".join(f"{channel}={count}" for channel, count in uneven.items())
                + f" (expected {n_frames})"
            )

    if not errors:
        required_values = raw[(labels["channel"].isin(REQUIRED_CHANNELS)).to_numpy()]
        empty = [column for column in larvae if required_values[column].isna().all()]
        if empty:
            warnings.append(f"Larva column(s) with no tracked frames: {empty}")

    return FimTrackReport(
        larvae=larvae,
        n_frames=n_frames,
        first_frame=first_frame,
        last_frame=last_frame,
        channels=channels,
        missing_channels=missing,
        extra_channels=extra,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
