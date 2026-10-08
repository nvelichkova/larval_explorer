import pandas as pd
import pytest

from larval_explorer.core import registry as R

PATTERN = r"(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)"


def make_registry():
    return R.add_recordings(R.empty_registry(), [
        "D:/data/amiGA-amRNAi_n1_att2.csv",
        "D:/data/w1118_n01_att1.csv",
        "D:/data/pilot recording.csv",
    ])


def test_add_recordings_uses_the_file_stem_and_leaves_metadata_blank():
    registry = make_registry()
    assert registry[R.RECORDING_ID].tolist() == ["amiGA-amRNAi_n1_att2", "w1118_n01_att1", "pilot recording"]
    assert R.metadata_columns(registry) == []
    with pytest.raises(R.RegistryError, match="Duplicate"):
        R.add_recordings(registry, ["E:/elsewhere/w1118_n01_att1.csv"])


def test_preview_does_not_change_the_registry_and_flags_non_matching_names():
    registry = make_registry()
    preview = R.preview_parse(registry, PATTERN)
    assert R.metadata_columns(registry) == []
    assert preview[R.MATCHED].tolist() == [True, True, False]
    assert preview.loc[0, ["genotype", "n", "attempt"]].tolist() == ["amiGA-amRNAi", "1", "2"]
    assert preview.loc[2, ["genotype", "n", "attempt"]].tolist() == ["", "", ""]


def test_apply_fills_blanks_keeps_user_edits_and_skips_non_matching_rows():
    registry = make_registry()
    registry["genotype"] = ["", "control", ""]
    preview = R.preview_parse(registry, PATTERN)

    applied = R.apply_parse(registry, preview)
    assert applied["genotype"].tolist() == ["amiGA-amRNAi", "control", ""]   # the edit survives
    assert applied["n"].tolist() == ["1", "01", ""]                          # text, not numbers
    assert R.unlabelled(applied)[R.RECORDING_ID].tolist() == ["pilot recording"]

    overwritten = R.apply_parse(registry, preview, overwrite=True)
    assert overwritten["genotype"].tolist() == ["amiGA-amRNAi", "w1118", ""]


def test_pattern_must_match_the_whole_name():
    assert R.parse_name(PATTERN, "w1118_n1_att1_rerun") is None
    assert R.parse_name(PATTERN, "w1118_n1_att1") == {"genotype": "w1118", "n": "1", "attempt": "1"}


def test_bad_patterns_are_rejected_with_a_clear_message():
    with pytest.raises(R.RegistryError, match="Invalid filename pattern"):
        R.parse_name("(?P<genotype", "x")
    with pytest.raises(R.RegistryError, match="no named groups"):
        R.parse_name(r".+_n\d+", "x_n1")
    with pytest.raises(R.RegistryError, match="not allowed"):
        R.parse_name(r"(?P<path>.+)", "x")


def test_round_trip_keeps_unknown_columns_and_text_values(tmp_path):
    registry = R.apply_parse(make_registry(), R.preview_parse(make_registry(), PATTERN))
    registry["age"] = ["5", "", "7"]
    registry["notes"] = ["", "control, repeat", ""]
    path = tmp_path / "project" / R.REGISTRY_NAME
    R.save_registry(registry, path)
    loaded = R.load_registry(path)
    pd.testing.assert_frame_equal(loaded, registry.astype("string"))
    assert loaded.loc[1, "n"] == "01" and loaded.loc[1, "age"] == ""
    assert R.metadata_columns(loaded) == ["genotype", "n", "attempt", "age", "notes"]
    assert R.metadata_for(loaded, "w1118_n01_att1")["genotype"] == "w1118"


def test_existing_values_for_autocomplete():
    registry = make_registry()
    registry["genotype"] = ["w1118", " w1118 ", ""]
    assert R.existing_values(registry, "genotype") == ["w1118"]
    assert R.existing_values(registry, "age") == []


def test_registry_without_required_columns_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("recording_id,genotype\na,w1118\n", encoding="utf-8")
    with pytest.raises(R.RegistryError, match="path"):
        R.load_registry(path)


def test_a_folder_adds_one_recording_per_csv(tmp_path):
    folder = tmp_path / "w1118"
    folder.mkdir()
    for name in ("w1118_n2_att1.csv", "w1118_n1_att1.csv", "notes.txt"):
        (folder / name).write_text("x", encoding="utf-8")
    single = tmp_path / "amiGA_n1_att1.csv"
    single.write_text("x", encoding="utf-8")

    registry = R.add_recordings(R.empty_registry(), [folder, single])
    assert registry[R.RECORDING_ID].tolist() == ["w1118_n1_att1", "w1118_n2_att1", "amiGA_n1_att1"]
    assert all(path.endswith(".csv") for path in registry[R.PATH])

    (tmp_path / "empty").mkdir()
    with pytest.raises(R.RegistryError, match="No CSV files"):
        R.add_recordings(registry, [tmp_path / "empty"])


def test_a_registry_row_is_always_one_file(tmp_path):
    with pytest.raises(R.RegistryError, match="Each registry row is one CSV"):
        R.recording_file(tmp_path)
    with pytest.raises(R.RegistryError, match="not found"):
        R.recording_file(tmp_path / "missing.csv")
    recording = tmp_path / "a.csv"
    recording.write_text("x", encoding="utf-8")
    assert R.recording_file(recording) == recording


def test_an_emptied_registry_with_metadata_columns_has_no_unlabelled_rows():
    """Removing every recording must leave a usable, empty registry."""
    registry = R.apply_parse(make_registry(), R.preview_parse(make_registry(), PATTERN)).iloc[:0]
    assert R.metadata_columns(registry) == ["genotype", "n", "attempt"]
    assert R.unlabelled(registry).empty
    assert R.existing_values(registry, "genotype") == []
    assert R.preview_parse(registry, PATTERN).empty
