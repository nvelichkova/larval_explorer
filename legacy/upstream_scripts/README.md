# Hierarchical dynamics of *Drosophila* larval exploratory behavior

This repository contains the analysis scripts and example data used to study the multi-scale organization of larval exploratory behavior. The code is organized as an ordered pipeline, from preprocessing of trajectory time series to geometric coarse-graining, HMM-based segmentation, event detection, and steering-model fitting.

## Repository structure

```text
larval_exploration_repository/
├── data/
├── docs/
├── outputs/
├── scripts/
├── requirements.txt
└── README.md
```

- `data/` contains the renamed CSV files distributed with the repository.
- `docs/` contains a machine-readable inventory of the data files.
- `outputs/` is the default location for all newly generated outputs.
- `scripts/` contains the numbered analysis scripts.

## Python dependencies

The scripts were written for Python 3 and rely on the following packages:

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn
- hmmlearn

Install them with:

```bash
pip install -r requirements.txt
```

## Numbered analysis pipeline

The scripts are intended to be run in numerical order.

### Preprocessing and geometric coarse-graining

1. `scripts/00_reorganize_raw_tracking_tables.py`
   - Reorganizes raw per-recording FIM-Track tables into wide trajectory tables.
   - **Important:** this script expects raw files `table_1.csv` ... `table_10.csv` inside `data/raw_recordings/`.
   - Those raw per-recording exports are **not included** in this repository.

2. `scripts/01_calibrate_and_filter_trajectories.py`
   - Converts frames to seconds and pixels to millimetres.
   - Removes trajectory segments shorter than 60 s.

3. `scripts/03_smooth_center_of_mass_trajectories.py`
   - Smooths the centre-of-mass coordinates with a Savitzky–Golay filter.

4. `scripts/04_compute_mean_body_length.py`
   - Computes mean and standard deviation of body length for each trajectory.

5. `scripts/05_extract_rdp_steps.py`
   - Applies the Ramer–Douglas–Peucker algorithm to generate the step representation.
   - Uses body length to set the simplification scale for each larva.

### HMM-based segmentation of geometric modes

6. `scripts/06_fit_hmm_on_phi.py`
   - Fits a 3-state Gaussian HMM using the variable `phi` as the only observable.

7. `scripts/07_filter_short_hmm_runs.py`
   - Removes very short isolated HMM runs with a one-pass temporal filter.

8. `scripts/08_extract_representative_hmm_segments.py`
   - Extracts sustained filtered HMM segments with at least 9 steps.

### Event detection and steering analysis

9. `scripts/09_detect_crawls_and_head_casts.py`
   - Detects crawls and head casts directly from the per-frame trajectory table.
   - Exports an event table and diagnostic plots.

10. `scripts/10_build_event_and_run_steps.py`
    - Builds two derived representations from trajectories and events:
      - event-level steps
      - run-anchor steps

11. `scripts/11_fit_steering_by_strategy.py`
    - Fits the steering model to representative HMM segments.
    - Computes state-wise parameter summaries and diagnostic plots.

## Quick start with the included data

If you want to run the repository starting from the processed data included here, you can skip script `00` and start from the distributed trajectory table:

```bash
python scripts/04_compute_mean_body_length.py
python scripts/05_extract_rdp_steps.py
python scripts/06_fit_hmm_on_phi.py
python scripts/07_filter_short_hmm_runs.py
python scripts/08_extract_representative_hmm_segments.py
python scripts/09_detect_crawls_and_head_casts.py
python scripts/10_build_event_and_run_steps.py
python scripts/11_fit_steering_by_strategy.py
```

If you want to reproduce the earlier preprocessing stages from raw FIM-Track exports, place the per-recording raw tables in `data/raw_recordings/` and run scripts `00`, `01`, and `03` first.

## Included CSV files

The repository includes the following renamed CSV files in `data/`.

### Core trajectory tables

- `all_preprocessed_trajectory_samples.csv`
  - A large per-frame tracking table representing an earlier, broad preprocessing stage.
  - Useful as an archive/reference table.

- `trajectory_timeseries.csv`
  - Main per-frame trajectory table used throughout the analysis.
  - This is the most important input for event detection and downstream geometric analyses.

- `trajectory_timeseries_with_state_labels.csv`
  - Per-frame trajectory table with state labels projected back onto the time series for visualization purposes.

- `mean_body_length_by_trajectory.csv`
  - Trajectory-level mean and standard deviation of body length.
  - Used by the RDP step-extraction script to set the geometric tolerance.

### Event- and step-level derived tables

- `event_intervals.csv`
  - Validated event table containing crawl and head-cast intervals.

- `event_level_steps.csv`
  - Event-level step representation built from the ordered crawl/head-cast sequence.
  - Corresponds to the "low model" used in the original working directory.

- `run_anchor_steps.csv`
  - Run-level step representation built from head-cast anchors plus track start/end.
  - Corresponds to the "high model" used in the original working directory.

### HMM and segment tables

- `rdp_steps_with_filtered_hmm_states.csv`
  - RDP step table containing both the raw HMM state labels and the filtered state labels.

- `representative_hmm_segments.csv`
  - Contiguous filtered HMM segments with at least 9 steps.
  - Used by the steering-fitting script.

### Steering summary table

- `steering_fit_summary_by_state.csv`
  - Summary statistics of steering-model parameters and related state-wise metrics.

A machine-readable version of this inventory is available in `docs/data_inventory.csv`.

## Original-to-renamed data mapping

The repository renames the working CSV files to make their role clearer and to keep naming consistent with the output names used by the scripts.

| Original file | Repository file |
|---|---|
| `input_data.csv` | `all_preprocessed_trajectory_samples.csv` |
| `trajectories.csv` | `trajectory_timeseries.csv` |
| `trajectories_com_estado.csv` | `trajectory_timeseries_with_state_labels.csv` |
| `events.csv` | `event_intervals.csv` |
| `low_model.csv` | `event_level_steps.csv` |
| `high_model.csv` | `run_anchor_steps.csv` |
| `passos_estado_filtrado.csv` | `rdp_steps_with_filtered_hmm_states.csv` |
| `fit_statistics_parameters.csv` | `steering_fit_summary_by_state.csv` |

## Notes on consistency with the manuscript

The repository names are chosen to be descriptive and consistent, but the original column names inside the CSV files have been preserved wherever they are required for compatibility with the analysis code and with the manuscript. In particular, some column names remain in Portuguese because they are part of the original working data schema.

## Recommended citation/use note

If you use this repository in a public archive, it is a good idea to keep the original manuscript title in the repository title or description so the code and data remain unambiguously linked to the associated paper.
