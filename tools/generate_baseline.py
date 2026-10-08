"""Generate the frozen parity baseline from the unmodified upstream scripts (D-009).

    python tools/generate_baseline.py data/amiGA-amRNAi_n1_att2.csv

The upstream scripts resolve paths as ``parents[1] / "data" | "outputs"``, so
they are copied byte for byte into ``<work>/scripts`` next to ``data`` and
``outputs`` folders and run there, in order, with segmentation absent (it has
no upstream equivalent). The harness only does what a user following the
upstream README would have to do by hand:

- script 00 is called through ``process_recordings`` with a list of one
  recording, which yields track IDs ``trajectory_0_N`` (D-015);
- outputs that later scripts expect under ``data/`` are copied there;
- output folders that scripts 09 and 10 assume to exist are created.

Script 11 is not run: it cannot read script 08's output (D-014).

The baseline is regenerated only when the user says so (CLAUDE.md §6). This
script refuses to overwrite an existing baseline unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPTS = REPO_ROOT / "legacy" / "upstream_scripts"
BASELINE_DIR = REPO_ROOT / "tests" / "data" / "baseline"

MAX_FRAME = 36000  # the value in upstream script 00's main()

# frozen name -> path of the upstream output inside the work tree
FROZEN = {
    "00_merged_tracking_table_unscaled.csv": "outputs/00_reorganized_tables/merged_tracking_table_unscaled.csv",
    "01_trajectory_timeseries_calibrated.csv": "outputs/01_calibrated_tables/trajectory_timeseries_calibrated.csv",
    "03_trajectory_timeseries.csv": "outputs/03_smoothed_tables/trajectory_timeseries.csv",
    "04_mean_body_length_by_trajectory.csv": "outputs/04_body_length/mean_body_length_by_trajectory.csv",
    "05_rdp_steps.csv": "outputs/05_rdp_steps/rdp_steps.csv",
    "06_rdp_steps_with_hmm_states.csv": "outputs/06_hmm/rdp_steps_with_hmm_states.csv",
    "07_rdp_steps_with_filtered_hmm_states.csv": "outputs/07_hmm_filtered/rdp_steps_with_filtered_hmm_states.csv",
    "08_representative_hmm_segments.csv": "outputs/08_representative_segments/representative_hmm_segments.csv",
    "09_event_intervals.csv": "outputs/09_event_detection/event_intervals.csv",
    "10_run_anchor_steps.csv": "outputs/10_event_and_run_steps/run_anchor_steps.csv",
    "10_event_level_steps.csv": "outputs/10_event_and_run_steps/event_level_steps.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(f"upstream_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(work: Path, name: str) -> None:
    print(f"  running {name}", flush=True)
    subprocess.run([sys.executable, str(work / "scripts" / name)], cwd=work, check=True)


def generate(recording: Path, work: Path) -> None:
    scripts = work / "scripts"
    shutil.copytree(LEGACY_SCRIPTS, scripts)
    (work / "data").mkdir()
    (work / "outputs").mkdir()

    print("  running 00 (process_recordings on one file)", flush=True)
    script_00 = load_script(scripts / "00_reorganize_raw_tracking_tables.py")
    script_00.process_recordings([recording], max_frame=MAX_FRAME, output_dir=work / "outputs" / "00_reorganized_tables")

    for name in [
        "01_calibrate_and_filter_trajectories.py",
        "03_smooth_center_of_mass_trajectories.py",
        "04_compute_mean_body_length.py",
        "05_extract_rdp_steps.py",
        "06_fit_hmm_on_phi.py",
        "07_filter_short_hmm_runs.py",
        "08_extract_representative_hmm_segments.py",
    ]:
        run_script(work, name)

    # Scripts 09 and 10 read from data/, and assume their output folders exist.
    shutil.copy(work / "outputs" / "03_smoothed_tables" / "trajectory_timeseries.csv", work / "data")
    (work / "outputs" / "09_event_detection").mkdir()
    run_script(work, "09_detect_crawls_and_head_casts.py")
    shutil.copy(work / "outputs" / "09_event_detection" / "event_intervals.csv", work / "data")
    (work / "outputs" / "10_event_and_run_steps").mkdir()
    run_script(work, "10_build_event_and_run_steps.py")


def freeze(recording: Path, work: Path) -> None:
    import hmmlearn, numpy, pandas, scipy, sklearn

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for frozen_name, relative in FROZEN.items():
        source = work / relative
        target = BASELINE_DIR / (frozen_name + ".gz")
        with open(target, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(source.read_bytes())
        lines.append(f"| `{target.name}` | `{relative}` | `{sha256(source)}` |")

    readme = f"""# Parity baseline

Outputs of the **unmodified** upstream scripts in `legacy/upstream_scripts/`,
frozen as the regression reference (docs/DECISIONS.md D-009). Generated by
`tools/generate_baseline.py`. Do not edit, and do not regenerate to make a
failing test pass (CLAUDE.md §6).

## Source recording

- File: `{recording.name}` (not in the repository)
- SHA-256: `{sha256(recording)}`
- Track segmentation: absent (stage 02 has no upstream equivalent)
- Track IDs: `trajectory_0_N`, from `process_recordings` on a list of one (D-015)
- `max_frame`: {MAX_FRAME}

## Known gap

**Stage 11 has no baseline.** Upstream script 11 requires columns
`t_inicio (s)` and `t_fim (s)` that script 08 does not write, so it cannot run
on these outputs (D-014). The gap stays until the upstream author resolves the
schema.

## Environment

Python {platform.python_version()}, numpy {numpy.__version__}, pandas {pandas.__version__},
scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, hmmlearn {hmmlearn.__version__}.

## Files

Each file is the upstream CSV, gzip-compressed and otherwise untouched. The
hash is of the uncompressed CSV.

| Frozen file | Upstream output | SHA-256 |
|---|---|---|
""" + "\n".join(lines) + "\n"
    (BASELINE_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path, help="raw FIM-Track export of one recording")
    parser.add_argument("--force", action="store_true", help="overwrite an existing baseline")
    parser.add_argument("--keep", type=Path, default=None, help="keep the work tree in this (new) folder")
    args = parser.parse_args()

    if BASELINE_DIR.exists() and any(BASELINE_DIR.iterdir()) and not args.force:
        raise SystemExit(f"A baseline already exists in {BASELINE_DIR}. Use --force to replace it.")

    recording = args.recording.resolve()
    if args.keep is not None:
        args.keep.mkdir(parents=True)
        generate(recording, args.keep.resolve())
        freeze(recording, args.keep.resolve())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            generate(recording, Path(tmp))
            freeze(recording, Path(tmp))
    print(f"Baseline written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
