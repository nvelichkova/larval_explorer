"""Which figures each stage has, built from a recording's saved results.

``stage_figures(pipeline, "06")`` returns ``{name: Figure}``. The GUI shows
them on canvases; batch export hands them to ``pipeline.save_figures``. Nothing
here writes a file.

Figures that exist once per track are named ``<figure>__<track_id>``.
"""

from __future__ import annotations

from typing import Callable, Iterable

from matplotlib.figure import Figure

from larval_explorer.core import schema as S
from larval_explorer.plots import events, hmm, segmentation, steering, style, trajectory

PER_TRACK_SEPARATOR = "__"


def _tracks(table, limit: int | None) -> list[str]:
    tracks = sorted(table[S.TRACK_ID].unique(), key=style.natural_key)
    return tracks if limit is None else tracks[:limit]


def _per_track(name: str, tracks: Iterable[str], build: Callable[[str], Figure]) -> dict[str, Callable[[], Figure]]:
    return {f"{name}{PER_TRACK_SEPARATOR}{track}": (lambda track=track: build(track)) for track in tracks}


def _stage_01(p, limit):
    return {"tracks_overview": lambda: segmentation.tracks_overview(p.load_table("01", "calibrated"), p.params.stage_01)}


def _stage_02(p, limit):
    if "track_segments.csv" not in p.manifest["stages"]["02"]["outputs"]:
        return {"segmentation_overview": lambda: style.message_figure(
            "Track segmentation is disabled: tracks are kept whole, as upstream does.")}
    return {"segmentation_overview": lambda: segmentation.segmentation_overview(
        p.load_table("01", "calibrated"), p.load_table("02", "trajectories"), p.load_table("02", "track_segments"),
        p.saved_params().stage_02, p.recording_id,
    )}


def _stage_03(p, limit):
    before, after = p.load_table("02", "trajectories"), p.load_table("03", "smoothed")
    return _per_track("smoothing", _tracks(after, limit), lambda track: trajectory.smoothing_overlay(before, after, track))


def _stage_04(p, limit):
    return {"body_length": lambda: trajectory.body_length(p.load_table("04", "body_length"))}


def _stage_05(p, limit):
    smoothed, steps = p.load_table("03", "smoothed"), p.load_table("05", "steps")
    figures = {"step_distributions": lambda: trajectory.step_distributions(steps)}
    figures.update(_per_track("rdp", _tracks(smoothed, limit), lambda track: trajectory.rdp_overlay(smoothed, steps, track)))
    return figures


def _stage_06(p, limit):
    smoothed, steps = p.load_table("03", "smoothed"), p.load_table("06", "steps")
    diagnostics = p.manifest["stages"]["06"]["diagnostics"]
    figures = {
        "phi_distribution_by_state": lambda: hmm.phi_by_state(steps),
        "phi_distribution_overlay": lambda: hmm.phi_overlay(steps),
        "transition_matrix": lambda: hmm.transition_matrix(diagnostics.get("transition_matrix", [])),
        "state_raster": lambda: hmm.state_raster(steps),
        "log_likelihood": lambda: hmm.log_likelihood_trace(diagnostics.get("log_likelihood_history", [])),
    }
    tracks = _tracks(steps, limit)
    figures.update(_per_track("posterior", tracks, lambda track: hmm.posterior_ribbons(steps, track)))
    figures.update(_per_track("state_trajectory", tracks, lambda track: hmm.state_trajectory(smoothed, steps, track)))
    return figures


def _stage_07(p, limit):
    steps, min_run = p.load_table("07", "steps"), p.saved_params().stage_07.min_run
    return {
        "filter_comparison": lambda: hmm.filter_comparison(steps, min_run),
        "run_length_histogram": lambda: hmm.run_length_histogram(steps, min_run),
    }


def _stage_08(p, limit):
    min_steps = p.saved_params().stage_08.min_steps
    return {"hmm_segment_lengths": lambda: hmm.hmm_segment_lengths(p.load_table("07", "steps"), min_steps)}


def _stage_09(p, limit):
    ev, sig, thr = (p.load_table("09", name) for name in ("events", "signals", "thresholds"))
    params = p.saved_params().stage_09
    figures = {"ethogram": lambda: events.ethogram(ev)}
    tracks = _tracks(sig, limit)
    figures.update(_per_track("anterior_velocity", tracks, lambda t: events.anterior_velocity(sig, ev, thr, t)))
    figures.update(_per_track("linear_speed", tracks, lambda t: events.linear_speed(sig, ev, t)))
    figures.update(_per_track("thresholds", tracks, lambda t: events.threshold_distributions(sig, thr, t, params)))
    figures.update(_per_track("head_cast_trajectory", tracks, lambda t: events.head_cast_trajectory(sig, ev, thr, t)))
    figures.update(_per_track("head_cast_circles", tracks, lambda t: events.head_cast_circles(sig, ev, thr, t)))
    return figures


def _stage_10(p, limit):
    smoothed = p.load_table("03", "smoothed")
    runs, steps = p.load_table("10", "run_anchor_steps"), p.load_table("10", "event_level_steps")
    return _per_track("step_models", _tracks(smoothed, limit), lambda t: trajectory.step_models(smoothed, runs, steps, t))


def _stage_11(p, limit):
    params = p.saved_params().stage_11
    table = p.load_table
    return {
        "segment_gallery": lambda: steering.segment_gallery(table("10", "event_level_steps"), table("08", "hmm_segments")),
        "param_box_weighted": lambda: steering.parameter_boxes(table("11", "segment_fits")),
        "param_kde_weighted": lambda: steering.parameter_kdes(table("11", "segment_fits"), params),
        "headcast_rate_boxplot": lambda: steering.headcast_rate(table("11", "headcast_segment_metrics")),
        "headcast_direction_polar": lambda: steering.headcast_directions(table("11", "headcast_directions")),
        "headcast_theta_fit": lambda: steering.headcast_theta_fit(table("11", "headcast_thetas")),
        "autocorr_by_state": lambda: steering.autocorrelation(table("11", "autocorr_by_state")),
        "ar2_diagnostics": lambda: steering.ar2_diagnostics(table("11", "run_fits"), params),
    }


_BUILDERS = {
    "01": _stage_01, "02": _stage_02, "03": _stage_03, "04": _stage_04, "05": _stage_05, "06": _stage_06,
    "07": _stage_07, "08": _stage_08, "09": _stage_09, "10": _stage_10, "11": _stage_11,
}

STAGES_WITH_FIGURES = tuple(_BUILDERS)


def figure_builders(pipeline, stage: str, max_tracks: int | None = None) -> dict[str, Callable[[], Figure]]:
    """Name -> function that draws that figure, for one stage of one recording.

    Nothing is drawn until a function is called, so a screen can list every
    figure and draw only the one being looked at. Figures describe the saved
    result, so they use the parameters the result was computed with, not
    parameters edited since. ``max_tracks`` limits the per-track figures to the
    first tracks; ``None`` lists every track. Stage 00 has no figure: its
    content is the validation report and a table.
    """
    if stage not in _BUILDERS:
        return {}
    return _BUILDERS[stage](pipeline, max_tracks)


def stage_figures(pipeline, stage: str, max_tracks: int | None = None) -> dict[str, Figure]:
    """All figures of one stage for one recording, drawn."""
    return {name: build() for name, build in figure_builders(pipeline, stage, max_tracks).items()}
