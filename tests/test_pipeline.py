import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from larval_explorer.core import manifest as M
from larval_explorer.core import params as P
from larval_explorer.core import pipeline as PL
from larval_explorer.core import schema as S
from tests.conftest import BASELINE_DIR, VALIDATION_RECORDING

RECORDING_ID = "amiGA-amRNAi_n1_att2"


# ──────────────────────────────────────────────────────────────────────────────
# Mechanics, on a small synthetic recording
# ──────────────────────────────────────────────────────────────────────────────

def write_synthetic_recording(path, n_frames=900, drop_channel=None, seed=5):
    """Two larva-like tracks: smooth paths, a peristaltic length wave, a head sweep now and then.

    Larva 1 is lost for 100 frames.
    """
    rng = np.random.default_rng(seed)
    frames = np.arange(n_frames)
    columns = {}
    for larva in range(2):
        heading = np.cumsum(rng.normal(scale=0.03, size=n_frames)) + 1.5 * np.sin(2 * np.pi * frames / 140.0)
        x = 1000 + np.cumsum(0.8 * np.cos(heading))
        y = 1000 + np.cumsum(0.8 * np.sin(heading))
        link = 6.0 * (1.0 + 0.12 * np.sin(2 * np.pi * frames / 10.0))        # one contraction per second
        sweep = sum(((-1) ** k) * 1.2 * np.exp(-0.5 * ((frames - centre) / 4.0) ** 2)
                    for k, centre in enumerate(range(80 + 37 * larva, n_frames, 130)))
        values = {}
        for k, part in enumerate(("spinepoint_1", "spinepoint_2", "spinepoint_3", "tail"), start=1):
            values[f"{part}_x"] = x - link * k * np.cos(heading)
            values[f"{part}_y"] = y - link * k * np.sin(heading)
        values["head_x"] = values["spinepoint_1_x"] + link * np.cos(heading + sweep)
        values["head_y"] = values["spinepoint_1_y"] + link * np.sin(heading + sweep)
        values["mom_x"], values["mom_y"] = values["spinepoint_2_x"], values["spinepoint_2_y"]
        for channel in ("perimeter", "area", "spine_length", "radius_1", "radius_2", "radius_3"):
            values[channel] = np.full(n_frames, 10.5)
        values["is_coiled"] = np.zeros(n_frames)
        values["is_well_oriented"] = np.ones(n_frames)
        columns[f"larva({larva})"] = values
    lines = [",larva(0),larva(1)"]
    for channel in list(S.REQUIRED_CHANNELS) + ["velocity"]:
        if channel == drop_channel:
            continue
        for frame in frames:
            cells = []
            for larva in range(2):
                lost = larva == 1 and 300 <= frame < 400
                value = columns[f"larva({larva})"].get(channel, np.zeros(n_frames))[frame]
                cells.append("" if lost else f"{value:.4f}")
            lines.append(f"{channel}({frame})," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


SHORT_SEGMENTS = P.PipelineParams(stage_02=P.SegmentationParams(min_segment_seconds=20.0))


@pytest.fixture
def synthetic(tmp_path):
    source = write_synthetic_recording(tmp_path / "synthetic.csv")
    return PL.RecordingPipeline("synthetic", source, tmp_path / "out", SHORT_SEGMENTS, metadata={"genotype": "w1118"})


def test_dependency_resolution():
    assert PL.with_dependencies(["05"]) == ["00", "01", "02", "03", "04", "05"]
    assert PL.with_dependencies(["10"]) == ["00", "01", "02", "03", "09", "10"]
    assert PL.with_dependencies(["11"]) == list(PL.STAGE_KEYS)
    assert PL.dependents("03") == ["04", "05", "06", "07", "08", "09", "10", "11"]
    assert PL.dependents("09") == ["10", "11"]
    with pytest.raises(PL.PipelineError, match="Unknown stage"):
        PL.with_dependencies(["12"])


def test_run_writes_the_output_tree_and_a_complete_manifest(synthetic):
    events = []
    outcome = synthetic.run(["05"], progress=lambda stage, event: events.append((stage, event)))
    assert outcome == {key: "finished" for key in ("00", "01", "02", "03", "04", "05")}
    assert events[:2] == [("00", "started"), ("00", "finished")]

    folder = synthetic.folder
    assert (folder / "03_smoothed" / "trajectory_timeseries.csv").is_file()
    assert (folder / "02_segments" / "track_segments.csv").is_file()
    saved = pd.read_csv(folder / "05_rdp_steps" / "rdp_steps.csv")
    assert "animal ID" in saved.columns and "tempo_inicial (s)" in saved.columns   # upstream names on disk
    assert S.T_START_S in synthetic.table("05", "steps").columns                   # canonical in memory

    manifest = json.loads((folder / M.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2 and manifest["recording_id"] == "synthetic"
    assert manifest["metadata"] == {"genotype": "w1118"}
    assert set(manifest["stages"]) == {"00", "01", "02", "03", "04", "05"}
    record = manifest["stages"]["02"]
    assert record["status"] == "ok" and record["params"]["min_segment_seconds"] == 20.0
    assert record["diagnostics"]["larvae_in"] == 2 and record["diagnostics"]["segments_out"] == 3
    assert record["outputs"] == ["track_segment_trajectories.csv", "track_segments.csv", "larva_summary.csv"]
    assert {"python", "pandas", "hmmlearn", "app_version"} <= set(manifest["environment"])


def test_reopening_restores_state_and_skips_finished_stages(synthetic):
    synthetic.run(["04"])
    reopened = PL.RecordingPipeline("synthetic", synthetic.source_file, synthetic.folder.parent, SHORT_SEGMENTS)
    statuses = reopened.statuses()
    assert [statuses[key] for key in ("00", "01", "02", "03", "04")] == ["ok"] * 5
    assert statuses["05"] == "missing"
    assert reopened.saved_params().stage_02 == SHORT_SEGMENTS.stage_02
    assert reopened.manifest["metadata"] == {"genotype": "w1118"}
    outcome = reopened.run(["05"])
    assert outcome == {"00": "skipped", "01": "skipped", "02": "skipped", "03": "skipped", "04": "skipped", "05": "finished"}


def test_changing_a_parameter_marks_that_stage_and_everything_downstream_stale(synthetic):
    synthetic.run(["05"])
    synthetic.set_params(dataclasses.replace(SHORT_SEGMENTS, stage_03=P.SmoothingParams(window_size=31)))
    statuses = synthetic.statuses()
    assert [statuses[key] for key in ("00", "01", "02")] == ["ok"] * 3
    assert [statuses[key] for key in ("03", "04", "05")] == ["stale"] * 3

    with pytest.raises(PL.PipelineError, match="no usable result"):
        synthetic.table("03", "smoothed")                      # a stale table is never fed to a stage
    assert len(synthetic.load_table("03", "smoothed")) > 0     # but can still be looked at

    outcome = synthetic.run(["05"])
    assert outcome == {"00": "skipped", "01": "skipped", "02": "skipped", "03": "finished", "04": "finished", "05": "finished"}
    assert synthetic.manifest["stages"]["03"]["params"]["window_size"] == 31

    synthetic.set_params(SHORT_SEGMENTS)                        # changing it back is stale again
    assert synthetic.status("03") == "stale"


def test_editing_the_raw_file_makes_everything_stale(synthetic):
    synthetic.run(["01"])
    write_synthetic_recording(synthetic.source_file, n_frames=901)
    reopened = PL.RecordingPipeline("synthetic", synthetic.source_file, synthetic.folder.parent, SHORT_SEGMENTS)
    assert reopened.status("00") == "stale" and reopened.status("01") == "stale"


def test_force_reruns_and_reset_removes_downstream_results(synthetic):
    synthetic.run(["04"])
    assert set(synthetic.run(["01"], force=True).values()) == {"finished"}
    assert synthetic.status("04") == "ok"       # same inputs, same fingerprint

    synthetic.reset("02")
    statuses = synthetic.statuses()
    assert statuses["01"] == "ok" and statuses["02"] == "missing" and statuses["04"] == "missing"
    assert not (synthetic.folder / "03_smoothed").exists()


def test_cancel_between_stages_leaves_a_consistent_tree(synthetic):
    finished = []
    with pytest.raises(PL.PipelineCancelled):
        synthetic.run(
            ["05"],
            progress=lambda stage, event: finished.append(stage) if event == "finished" else None,
            should_cancel=lambda: len(finished) >= 2,
        )
    statuses = synthetic.statuses()
    assert statuses["00"] == "ok" and statuses["01"] == "ok" and statuses["02"] == "missing"
    assert synthetic.run(["02"])["02"] == "finished"


def test_a_failing_stage_is_recorded_with_its_traceback_and_can_be_rerun(tmp_path):
    source = write_synthetic_recording(tmp_path / "broken.csv", drop_channel="tail_y")
    pipeline = PL.RecordingPipeline("broken", source, tmp_path / "out")
    with pytest.raises(PL.PipelineError, match="tail_y") as error:
        pipeline.run(["01"])
    assert error.value.stage == "00"
    record = pipeline.manifest["stages"]["00"]
    assert record["status"] == "failed" and "SchemaError" in record["error"]
    assert pipeline.status("00") == "failed" and pipeline.status("01") == "missing"

    write_synthetic_recording(source)
    repaired = PL.RecordingPipeline("broken", source, tmp_path / "out")
    assert repaired.run(["01"]) == {"00": "finished", "01": "finished"}


def test_a_stage_cannot_run_on_missing_dependencies(synthetic):
    with pytest.raises(PL.PipelineError, match="needs stage 02"):
        synthetic.run_stage("03")


def test_an_output_folder_belongs_to_one_recording(synthetic):
    synthetic.run(["00"])
    (synthetic.folder.parent / "other").mkdir()
    (synthetic.folder / M.MANIFEST_NAME).replace(synthetic.folder.parent / "other" / M.MANIFEST_NAME)
    with pytest.raises(PL.PipelineError, match="holds results for 'synthetic'"):
        PL.RecordingPipeline("other", synthetic.source_file, synthetic.folder.parent)


def test_manifest_handles_numpy_values_and_rejects_other_versions(tmp_path):
    manifest = M.new_manifest("r", tmp_path / "r.csv", {"n": np.int64(3)})
    manifest["stages"]["00"] = M.stage_record(
        status="ok", params={"a": (1, 2)}, fingerprint="f",
        diagnostics={"x": np.float64(1.5), "bad": float("nan"), "arr": np.arange(2)},
    )
    M.save_manifest(tmp_path, manifest)
    loaded = M.load_manifest(tmp_path)
    assert loaded["metadata"] == {"n": 3}
    assert loaded["stages"]["00"]["diagnostics"] == {"x": 1.5, "bad": None, "arr": [0, 1]}
    assert not (tmp_path / (M.MANIFEST_NAME + ".tmp")).exists()

    loaded["schema_version"] = 1
    (tmp_path / M.MANIFEST_NAME).write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version 1"):
        M.load_manifest(tmp_path)
    assert M.load_manifest(tmp_path / "nowhere") is None


# ──────────────────────────────────────────────────────────────────────────────
# The real recording, raw file to steering fit
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_FILES = {
    "00_merged_tracking_table_unscaled": ("00", "merged_tracking_table_unscaled.csv"),
    "01_trajectory_timeseries_calibrated": ("02", "track_segment_trajectories.csv"),
    "03_trajectory_timeseries": ("03", "trajectory_timeseries.csv"),
    "04_mean_body_length_by_trajectory": ("04", "mean_body_length_by_trajectory.csv"),
    "05_rdp_steps": ("05", "rdp_steps.csv"),
    "06_rdp_steps_with_hmm_states": ("06", "rdp_steps_with_hmm_states.csv"),
    "07_rdp_steps_with_filtered_hmm_states": ("07", "rdp_steps_with_filtered_hmm_states.csv"),
    "08_representative_hmm_segments": ("08", "representative_hmm_segments.csv"),
    "09_event_intervals": ("09", "event_intervals.csv"),
    "10_run_anchor_steps": ("10", "run_anchor_steps.csv"),
    "10_event_level_steps": ("10", "event_level_steps.csv"),
}


@pytest.fixture(scope="module")
def upstream_equivalent_run(tmp_path_factory):
    """The whole pipeline on the validation recording, configured as upstream ran."""
    if not VALIDATION_RECORDING.exists():
        pytest.skip(f"validation recording not present: {VALIDATION_RECORDING}")
    params = P.PipelineParams(
        stage_00=P.ReorganizeParams(max_frame=36000),
        stage_02=P.SegmentationParams(enabled=False),
    )
    pipeline = PL.RecordingPipeline(RECORDING_ID, VALIDATION_RECORDING, tmp_path_factory.mktemp("upstream_like"), params)
    pipeline.run()
    return pipeline


@pytest.mark.parametrize("baseline_name", BASELINE_FILES)
def test_pipeline_files_equal_the_upstream_baseline(upstream_equivalent_run, baseline_name):
    """Raw file in, CSVs out: every file the upstream scripts wrote, reproduced exactly."""
    stage, filename = BASELINE_FILES[baseline_name]
    ours = pd.read_csv(upstream_equivalent_run.stage_folder(stage) / filename)
    expected = pd.read_csv(BASELINE_DIR / f"{baseline_name}.csv.gz")
    assert_frame_equal(ours, expected, check_exact=True)


def test_full_run_manifest(upstream_equivalent_run):
    pipeline = upstream_equivalent_run
    assert set(pipeline.statuses().values()) == {"ok"}
    manifest = M.load_manifest(pipeline.folder)
    assert set(manifest["stages"]) == set(PL.STAGE_KEYS)
    assert manifest["hmm_pool"] == [RECORDING_ID]
    means = manifest["hmm_state_mean_phi"]
    assert set(means) == {"0", "1", "2"} and means["0"] > means["2"] > means["1"]
    assert manifest["stages"]["02"]["diagnostics"]["segmentation"] == "disabled"
    assert manifest["stages"]["11"]["diagnostics"]["runs_fitted"] == 63
    assert len(pipeline.table("11", "segment_fits")) == 10


def test_full_run_with_segmentation_on(tmp_path_factory):
    """Defaults, as a user would run it: 21 track segments, every stage completes."""
    if not VALIDATION_RECORDING.exists():
        pytest.skip(f"validation recording not present: {VALIDATION_RECORDING}")
    pipeline = PL.RecordingPipeline(RECORDING_ID, VALIDATION_RECORDING, tmp_path_factory.mktemp("segmented"))
    outcome = pipeline.run()
    assert set(outcome.values()) == {"finished"}
    assert pipeline.manifest["stages"]["02"]["diagnostics"]["segments_out"] == 21

    events = pipeline.table("09", "events")
    track_segments = pipeline.table("02", "track_segments")
    kept = track_segments[track_segments[S.KEPT]].set_index(S.TRACK_ID)
    assert set(events[S.TRACK_ID]) <= set(kept.index)
    bounds = kept.loc[events[S.TRACK_ID]]
    assert (events[S.START_S].to_numpy() >= bounds[S.START_S].to_numpy()).all()   # no event outside its segment
    assert (events[S.END_S].to_numpy() <= bounds[S.END_S].to_numpy()).all()
    assert "state_param_summary" in {output.table for output in PL.spec("11").outputs}
    pipeline.table("11", "state_param_summary")   # loads, even if few segments were fitted
