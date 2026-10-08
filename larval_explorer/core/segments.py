"""Stage 02 - split each larva's track into contiguous tracked segments (D-008).

A lost track leaves no rows for the missing frames, so a gap shows up as a jump
in the frame sequence, not as NaN. Gaps of at most ``bridge_max_frames`` missing
frames are filled by interpolation; longer gaps split the track. Each run that
lasts at least ``min_segment_seconds`` becomes an independent track with its
own ID; shorter runs are dropped.

These are *track* segments. They are unrelated to the *HMM* segments of
stage 08.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from larval_explorer.core import schema
from larval_explorer.core.params import CalibrationParams, SegmentationParams


class SegmentationError(ValueError):
    """The trajectory table cannot be segmented as given."""


@dataclass(frozen=True)
class SegmentationResult:
    """Output of track segmentation.

    ``table`` is the trajectory table restricted to kept segments, with the
    track ID replaced by the segment ID and three added columns: the source
    track ID, the segment index, and whether the row was interpolated.
    ``segments`` has one row per segment found, kept or not. ``larva_summary``
    has one row per source track.
    """

    table: pd.DataFrame
    segments: pd.DataFrame
    larva_summary: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def find_runs(frames: np.ndarray, bridge_max_frames: int) -> list[tuple[int, int]]:
    """Positions ``(start, stop)`` of contiguous runs in a sorted frame array.

    Two consecutive tracked frames belong to the same run when at most
    ``bridge_max_frames`` frames are missing between them. ``stop`` is exclusive.
    """
    if len(frames) == 0:
        return []
    missing = np.diff(frames) - 1
    breaks = np.flatnonzero(missing > bridge_max_frames) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [len(frames)]])
    return list(zip(starts.tolist(), stops.tolist()))


def frame_numbers(time_s: pd.Series, frame_rate_fps: float, track_id: str) -> np.ndarray:
    """Integer frame numbers of one track's rows, which must be in time order.

    Raises if two rows land on the same frame. That happens when the time
    column was rounded more coarsely than one frame, e.g. upstream's 0.1 s
    rounding at more than 10 fps.
    """
    frames = np.rint(time_s.to_numpy(float) * frame_rate_fps).astype(np.int64)
    if len(frames) > 1 and not (np.diff(frames) > 0).all():
        raise SegmentationError(
            f"Track '{track_id}' has repeated or unordered frames at {frame_rate_fps} fps. "
            "Check the frame rate and that time is not rounded more coarsely than one frame."
        )
    return frames


def segment_tracks(
    trajectories: pd.DataFrame,
    params: SegmentationParams,
    calibration: CalibrationParams,
    *,
    recording_id: str,
) -> SegmentationResult:
    """Split a calibrated trajectory table into contiguous track segments.

    ``trajectories`` is the canonical per-frame table from stage 01. The frame
    rate and time rounding come from ``calibration`` so that times of
    interpolated rows are built exactly as stage 01 builds them.
    """
    schema.TRAJECTORY.validate(trajectories)
    fps = calibration.frame_rate_fps

    numeric = [c for c in trajectories.columns if pd.api.types.is_numeric_dtype(trajectories[c])]
    interpolate_columns = [c for c in numeric if c not in schema.FLAG_CHANNELS and c != schema.TIME_S]
    out_columns = list(trajectories.columns) + [schema.SOURCE_TRACK_ID, schema.SEGMENT_INDEX, schema.INTERPOLATED]

    # A row counts as tracked only if every position channel is present.
    tracked = trajectories[trajectories[list(schema.POSITION_CHANNELS)].notna().all(axis=1)]
    partial_rows_dropped = len(trajectories) - len(tracked)

    all_frames = np.rint(trajectories[schema.TIME_S].to_numpy(float) * fps).astype(np.int64)
    recording_first = int(all_frames.min()) if len(all_frames) else 0
    recording_last = int(all_frames.max()) if len(all_frames) else -1
    recording_frames = recording_last - recording_first + 1

    source_ids = sorted(trajectories[schema.TRACK_ID].unique(), key=lambda t: (schema.larva_index(t), str(t)))

    pieces: list[pd.DataFrame] = []
    segment_rows: list[dict] = []
    larva_rows: list[dict] = []
    warnings: list[str] = []

    for source_id in source_ids:
        larva = schema.larva_index(source_id)
        track = tracked[tracked[schema.TRACK_ID] == source_id].sort_values(schema.TIME_S, kind="mergesort")
        frames = frame_numbers(track[schema.TIME_S], fps, source_id)

        found = kept = 0
        kept_duration = 0.0
        for segment, (start, stop) in enumerate(find_runs(frames, params.bridge_max_frames)):
            run_frames = frames[start:stop]
            first, last = int(run_frames[0]), int(run_frames[-1])
            duration = (last - first) / fps
            is_kept = duration >= params.min_segment_seconds
            track_id = schema.segment_track_id(recording_id, larva, segment)
            n_bridged = (last - first + 1) - len(run_frames)

            found += 1
            segment_rows.append({
                schema.TRACK_ID: track_id,
                schema.SOURCE_TRACK_ID: source_id,
                schema.LARVA_INDEX: larva,
                schema.SEGMENT_INDEX: segment,
                schema.FIRST_FRAME: first,
                schema.LAST_FRAME: last,
                schema.START_S: first / fps,
                schema.END_S: last / fps,
                schema.DURATION_S: duration,
                schema.N_FRAMES_TRACKED: len(run_frames),
                schema.N_FRAMES_BRIDGED: n_bridged,
                schema.KEPT: is_kept,
            })
            if not is_kept:
                continue

            kept += 1
            kept_duration += duration
            piece = track.iloc[start:stop].set_axis(run_frames, axis=0)
            piece = piece.reindex(np.arange(first, last + 1))
            interpolated = ~np.isin(piece.index.to_numpy(), run_frames)
            if n_bridged:
                piece[interpolate_columns] = piece[interpolate_columns].interpolate(method="linear")
                piece[schema.TIME_S] = piece[schema.TIME_S].fillna(
                    pd.Series(piece.index / fps, index=piece.index).round(calibration.time_decimals)
                )
                # Flags and any non-numeric metadata carry the last tracked value.
                piece = piece.ffill()
            piece[schema.TRACK_ID] = track_id
            piece[schema.SOURCE_TRACK_ID] = source_id
            piece[schema.SEGMENT_INDEX] = segment
            piece[schema.INTERPOLATED] = interpolated
            pieces.append(piece[out_columns])

        # Gaps are counted against the whole recording, so a track that starts
        # late or ends early has a leading or trailing gap.
        edges = np.concatenate([[recording_first - 1], frames, [recording_last + 1]])
        gaps = np.diff(edges) - 1
        gaps = gaps[gaps > 0]
        larva_rows.append({
            schema.SOURCE_TRACK_ID: source_id,
            schema.LARVA_INDEX: larva,
            schema.TRACKED_FRACTION: len(frames) / recording_frames if recording_frames else 0.0,
            schema.GAP_COUNT: int(len(gaps)),
            schema.LONGEST_GAP_FRAMES: int(gaps.max()) if len(gaps) else 0,
            schema.SEGMENTS_FOUND: found,
            schema.SEGMENTS_KEPT: kept,
            schema.KEPT_DURATION_S: kept_duration,
        })
        if kept == 0:
            warnings.append(f"Track '{source_id}' has no segment of at least {params.min_segment_seconds} s.")

    if pieces:
        table = pd.concat(pieces, ignore_index=True)
    else:
        table = pd.DataFrame(columns=out_columns)
    segments = pd.DataFrame(segment_rows)
    larva_summary = pd.DataFrame(larva_rows)

    segments_out = int(segments[schema.KEPT].sum()) if len(segments) else 0
    diagnostics = {
        "larvae_in": len(source_ids),
        "segments_found": len(segments),
        "segments_out": segments_out,
        "segments_dropped": len(segments) - segments_out,
        "recording_frames": recording_frames,
        "frames_in": len(tracked),
        "frames_out": len(table),
        "frames_bridged": int(table[schema.INTERPOLATED].sum()) if len(table) else 0,
        "partial_rows_dropped": partial_rows_dropped,
    }
    return SegmentationResult(
        table=table,
        segments=segments,
        larva_summary=larva_summary,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )
