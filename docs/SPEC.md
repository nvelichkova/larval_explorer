# SPEC.md — Larval Explorer

A PyQt5 application wrapping the hierarchical-dynamics larval exploratory
behaviour pipeline, for new FIM-Track recordings, with per-stage figures and
condition-aware batch processing.

Standing engineering rules: `CLAUDE.md`. Decisions and their reasoning:
`DECISIONS.md`. This file is the plan.

> **Revision 2 (2026-10-08).** Adds stage 02 (track segmentation) after
> validating a real recording against the pipeline's assumptions; changes the
> parity baseline to own-data (the upstream repo does not publish the CSVs its
> README describes); changes the batch unit to one CSV per recording with no
> cross-recording pooling. See D-008 … D-011.
>
> **Revision 3 (2026-10-08).** After reading the upstream function bodies:
> every stage body is forked, only pure leaf helpers are imported (D-013);
> stage 11's segment-time column constants are corrected (D-014); the track-ID
> scheme with segmentation off is fixed and the pooled fit keys on
> `(recording_id, track_id)` (D-015); stored HMM state indices are never
> reordered (D-016). The two published upstream CSVs now live in
> `legacy/upstream_data/`.

---

## 1. Goals

1. Process new FIM-Track recordings end to end without editing Python.
2. Expose every pipeline parameter, record every parameter used.
3. Produce an inspectable figure at every stage — trajectories above all.
4. Batch many recordings, each tagged with experimental conditions
   (genotype now, age later), and compare across them.
5. Preserve the upstream method, diverging only where upstream is demonstrably
   wrong for this data — explicitly, with reasoning recorded.

Non-goals (for now): re-tracking video, editing trajectories by hand, altering
the scientific method beyond §5a, cross-platform packaging beyond
Windows/Anaconda.

---

## 2. Source material audit

Upstream repo files are flat at the repository root, but every script resolves
paths as `Path(__file__).resolve().parents[1] / "data" | "outputs"` — i.e. they
expect to live in a `scripts/` subdirectory alongside `data/` and `outputs/`.
A plain clone therefore does not run. Vendor the scripts into
`legacy/upstream_scripts/` and reconstruct the expected layout in fixtures.

**The upstream repo does not contain the data its README describes.** The
README documents a `data/` folder of ten CSVs plus `docs/`, `outputs/` and
`requirements.txt`. The single published commit contains eleven `.py` files,
the README, and two CSVs — `mean_body_length_by_trajectory.csv` and
`representative_hmm_segments.csv` — both of which are mid-branch intermediates
whose own inputs are absent. **No upstream stage can be run end to end from
what is published.** This changes the parity strategy; see D-009 and §5
Phase 1. The two published CSVs are vendored read-only in
`legacy/upstream_data/`.

Every upstream stage function reads its input CSV and writes its output CSV
itself, so none can be imported as a pure stage. The body of each is forked
into `core/stages.py`, ported verbatim; only pure leaf helpers are imported
(D-013).

| Script | Shape | Port strategy |
|---|---|---|
| `00_reorganize_raw_tracking_tables.py` | `raw_to_wide_format(input_csv, output_csv, max_frame)` + `process_recordings` + `main()`; all read and write files | Fork bodies; single-file relabel to `trajectory_0_N` kept (D-015); concatenation removed (D-010) |
| `01_calibrate_and_filter_trajectories.py` | `calibrate_and_filter(...)` reads and writes CSV | Fork body; duration filter moves to stage 02 (D-008) |
| *(new)* **stage 02** | — | **New. Track segmentation. See §2a** |
| `03_smooth_center_of_mass_trajectories.py` | `smooth_trajectory_timeseries(...)` reads and writes CSV | Fork body; now operates per segment |
| `04_compute_mean_body_length.py` | `compute_mean_body_length(...)` reads and writes CSV | Fork body; import `euclidean_distance` |
| `05_extract_rdp_steps.py` | `extract_rdp_steps(...)` reads and writes CSV; RDP is recursive | Fork body; import `perpendicular_distance`, `build_steps_and_angles`; iterative RDP rewrite, with recursive `rdp` imported as the test reference only (D-006) |
| `06_fit_hmm_on_phi.py` | One function fits, writes CSV and writes figures; imports pyplot | Fork body, split figures out; nothing importable |
| `07_filter_short_hmm_runs.py` | `filter_hmm_runs(...)` reads and writes CSV | Fork body; import `run_length_encode`, `filter_state_sequence_one_pass` |
| `08_extract_representative_hmm_segments.py` | `extract_representative_segments(...)` reads and writes CSV; helper reads module-global `MIN_STEPS` | Fork both functions; nothing importable |
| `09_detect_crawls_and_head_casts.py` | **Module-level script.** `df = pd.read_csv(INPUT_CSV)` at import; top-level loop; 7 PDFs per animal | **Fork and refactor** (D-013); nothing importable |
| `10_build_event_and_run_steps.py` | All logic inside an argparse `main()`; has `detect_columns_traj` sniffer | Fork `main()` body; import `angle_wrap_pi`, `interp_pos`, `build_step_rows`; fold sniffer into `schema.py` |
| `11_fit_steering_by_strategy.py` | ~90 lines of module-level config, a 430-line `main()`, 44 KB; imports pyplot | **Fork and refactor** (D-013); segment-time column constants corrected (D-014); nothing importable |

Upstream left `02_` vacant. Stage 02 takes that slot so numbering stays aligned
with the manuscript.

Roughly 60% of the porting effort is scripts 09 and 11.

---

## 2a. Stage 02 — track segmentation (new)

**Why it exists.** Real recordings lose tracks. A validation run on
`amiGA-amRNAi_n1_att2.csv` (8 larvae, 9851 frames, 16.4 min at 10 fps) found
missing values — encoded as **empty cells**, not `-1` sentinels — in 7 of 8
larvae, with individual gaps up to 4853 frames (≈ 8 min):

| larva | tracked | gaps | longest gap |
|---|---|---|---|
| 0 | 72.7% | 4 | 2116 fr |
| 1 | 100% | 0 | — |
| 2 | 97.4% | 3 | 148 fr |
| 3 | 98.1% | 3 | 96 fr |
| 4 | 46.3% | 7 | 2202 fr |
| 5 | 55.3% | 5 | 3354 fr |
| 6 | 45.4% | 5 | 4853 fr |
| 7 | 77.2% | 6 | 1304 fr |

Upstream has no defence against this: scripts 00's relabeling, gap-interpolation
and static-removal helpers exist but are **disabled** in the published path.

**Four consequences, all silent:**

1. Stage 01's filter does not do what its docstring says. The docstring reads
   "removes trajectory *segments* shorter than 60 s"; the implementation
   computes `groupby("larva")["time"].max() - min()`, which is **span, not
   contiguous tracked time**. larva(6) above is tracked 45% of the window and
   passes as a 985-second track.
2. Savitzky–Golay smoothing over a NaN run propagates NaN or raises.
3. RDP draws a straight chord across the gap, producing one enormous spurious
   step with nonsense length, speed and turning angle — which then enters the
   HMM as a genuine observation.
4. Crawl/head-cast detection invents events at gap seams.

**What stage 02 does.** Split each larva into contiguous tracked segments;
bridge gaps shorter than a tolerance by interpolation; split on anything
longer; assign each surviving segment its own track ID; apply the ≥ 60 s
minimum **per segment** rather than per larva. This is what the upstream
docstring describes, and it makes 2–4 disappear as a side effect.

On the validation recording this turns 8 nominal tracks into **21 real
tracks**, and discards material that upstream would have analysed as
continuous.

**Track IDs.** Segment IDs are `<recording_id>__larva<N>__seg<K>`, stable
across re-runs, so a segment is traceable back to its source. Every downstream
stage treats a segment as an independent animal track; `04` computes body
length per segment; `06`'s HMM sequences are per segment.

**Interaction with D-010.** Because segments are per recording and recordings
are never concatenated, segment IDs are globally unique without the recording
index trick upstream script 00 uses.

---

## 3. Architecture

```
core/schema.py      canonical column names; adapter on load/save; FIM-Track validation
core/params.py      frozen dataclasses, one per stage, upstream defaults
core/segments.py    gap detection, bridging, splitting, segment IDs  (stage 02)
core/stages.py      stage_00 … stage_11, each (inputs, params) -> StageResult
core/pipeline.py    stage DAG, dependency resolution, dirty-state, resume
core/registry.py    dataset registry: recording -> condition metadata
core/manifest.py    run manifest read/write
plots/*.py          (data, params) -> matplotlib.figure.Figure
workers/*.py        QThread/QRunnable, progress signals, cancellation
ui/*.py             PyQt5 widgets
```

`StageResult` carries: output dataframes (keyed by name), a diagnostics dict
(counts kept/dropped, convergence info, timings), and any warnings raised.
Stages are pure: no disk I/O, no figure creation, no Qt.

---

## 4. Data model

### 4.1 Dataset registry

The batch unit is **one recording**, and every recording is exactly one
FIM-Track CSV (D-021). Adding a folder to the registry adds one row per CSV in
it; files are never merged. Conditions are generic key-value metadata so that
`age` slots in later without migration (D-003).

`dataset_registry.csv`:

```
recording_id,path,genotype,n,attempt,age,notes
amiGA-amRNAi_n1_att2,D:/data/amiGA-amRNAi_n1_att2.csv,amiGA-amRNAi,1,2,,
w1118_n1_att1,D:/data/w1118_n1_att1.csv,w1118,1,1,,control
```

- `recording_id` and `path` are required; everything else is free metadata.
- Unknown columns are carried through and offered as grouping keys in the UI.
- **Filename metadata parsing** (D-011): an optional user-editable regex
  pre-fills metadata columns from the filename — e.g.
  `(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)` against
  `amiGA-amRNAi_n1_att2`. Always previewed before applying, always overridable.
- The registry editor offers existing values as autocomplete, so typos do not
  create phantom groups.
- Phase 5 aggregation groups by any selected subset of metadata columns.

### 4.2 Run manifest

One `manifest.json` per recording output folder:

```json
{
  "schema_version": 2,
  "recording_id": "amiGA-amRNAi_n1_att2",
  "metadata": {"genotype": "amiGA-amRNAi", "n": 1, "attempt": 2},
  "source_file": "D:/data/amiGA-amRNAi_n1_att2.csv",
  "stages": {
    "02": {"status": "ok", "params": {"bridge_max_frames": 10,
           "min_segment_seconds": 60.0},
           "diagnostics": {"larvae_in": 8, "segments_out": 21,
                           "segments_dropped": 15},
           "duration_s": 3.1, "warnings": []}
  },
  "hmm_pool": ["amiGA-amRNAi_n1_att2", "w1118_n1_att1"],
  "hmm_state_mean_phi": {"0": 0.81, "1": -0.79, "2": 0.03},
  "environment": {"python": "3.11.x", "hmmlearn": "0.3.x", "app_version": "0.5.0"}
}
```

`hmm_pool` records which recordings were in the pooled HMM fit (D-002) — the
result is analysis-scoped and not reproducible without it.

`hmm_state_mean_phi` records the fitted mean φ (radians, un-standardised) for
each stored state index (D-016; the values above are illustrative). Stored
indices are upstream's and are never reordered: 0 is initialised as
persistent-turning, 1 as reversing, 2 as mixed. Display and aggregation code
takes its state ordering from this field, and it shows whether EM drifted a
state away from its initialised role.

Without the resolved parameter set, a result is not reproducible. The manifest
is not optional output.

### 4.3 Output tree

```
<output_root>/<recording_id>/
├── manifest.json
├── 00_reorganized/  01_calibrated/  02_segments/  03_smoothed/
├── 04_body_length/  05_rdp_steps/   06_hmm/       07_hmm_filtered/
├── 08_segments/     09_events/      10_step_models/  11_steering/
└── figures/<stage>/
```

Note `02_segments/` (track segmentation) and `08_segments/` (representative HMM
segments) are different things; keep the distinction in naming and in UI labels
— "track segments" vs "HMM segments".

---

## 5. Phases

Each phase ends with its acceptance criteria met and `DECISIONS.md` updated.

### Phase 0 — Scaffold

- Repo layout per `CLAUDE.md` §2; vendor upstream scripts into `legacy/`.
- `requirements.txt`, pytest config, package skeleton.
- `tests/test_layering.py`: asserts `core` and `plots` import with Qt absent
  and that neither module tree imports `matplotlib.pyplot`.

**Acceptance:** `pytest tests/test_layering.py` passes; `legacy/` is
byte-identical to the upstream clone.

---

### Phase 1 — Headless core

The largest phase. No GUI. Natural sub-splits if a session runs long:
**1a** schema/params/validator · **1b** stages 00–08 and 10 plus the RDP rewrite
· **1c** the 09/11 fork · **1d** pipeline/registry/manifest and the baseline test.

1. `schema.py`: canonical names, bidirectional adapter, FIM-Track validation
   with a clear report of missing/unexpected channels (never a `KeyError`).
   Must accept the verified real format: header row `larva(0)…larva(N)`,
   first column `<channel>(<frame>)`, missing values as **empty cells**.
2. `params.py`: one frozen dataclass per stage, defaults from §7.
3. `segments.py` + stage 02 per §2a.
4. `stages.py`: all stages as pure functions. Fork every stage body verbatim;
   import only the pure leaf helpers listed in D-013. Stage 11 reads
   `start_time_s` / `end_time_s` from stage 08's output (D-014). `schema.py`
   validates the columns at every stage boundary.
5. Iterative RDP rewrite (D-006), asserted identical to the recursive version.
6. `pipeline.py`, `registry.py` (incl. filename parsing), `manifest.py`.
7. **Baseline parity test** (D-009): run the *unmodified* upstream scripts once
   on one real recording, freeze their outputs as `tests/data/baseline/`, and
   assert the refactor reproduces them under `assert_frame_equal`. Stage 02 is
   excluded by construction — it has no upstream equivalent — so the baseline
   for stages 03+ is generated with segmentation disabled. Track IDs in the
   baseline are `trajectory_0_N` (D-015). **Stage 11 is a known gap:** the
   unmodified script 11 cannot read script 08's output (D-014), so it has no
   upstream baseline until the author resolves the schema; record this in the
   baseline README.

**Acceptance:**
- Refactored stages reproduce the frozen own-data baseline exactly, with
  segmentation off.
- With segmentation on, the pipeline runs end to end on the validation
  recording without NaN propagation, and no RDP step spans a gap boundary
  (assert this directly — it is the bug stage 02 exists to prevent).
- No `pyplot` import in `core/`.
- Iterative RDP returns identical vertices to recursive RDP.

---

### Phase 2 — Plot library

Every figure in §8 as a factory returning a `Figure`. Shared style module
(fonts, DPI, colour maps, state palette). State and segment colours must be
stable across figures and datasets — colour comes from canonical index, never
from plot order.

**Acceptance:** every §8 figure renders from a console call; figures with
upstream equivalents are visually equivalent; `savefig` appears nowhere in
`plots/`.

---

### Phase 3 — Single-dataset GUI

- **Project tab** — create/open project, set output root, edit the registry,
  run filename parsing, pick the active recording.
- **Ingest tab (00/01)** — file picker (any filename), channel validation
  report, raw table preview, calibration controls.
- **★ Segmentation tab (02)** — see §6a. Build this *second*, right after
  Ingest: nothing downstream is trustworthy until segmentation is right.
- **One tab per stage (03 … 11)** — parameter controls, Run, figure canvas,
  output preview table, stage status.
- **Trajectory Explorer** — animal/segment selector, trajectory canvas,
  independently toggleable layers (raw COM, bridged, smoothed, RDP vertices,
  HMM state colouring, filtered states, event markers, representative HMM
  segments), time scrubber, pan/zoom, export current view.
- Stage completion state persisted; changing a parameter marks downstream
  stages dirty and visibly stale.

**Acceptance:** a full run from a raw FIM-Track CSV to the steering fit by
clicking only; outputs match the headless pipeline exactly; reopening the
project restores stage state.

---

### Phase 4 — Batch

- Queue of recordings from the registry, with a stage-range selector.
- `QThreadPool`, one recording per worker, configurable concurrency
  (default 2 — hmmlearn and RDP are CPU-bound).
- **No cross-recording concatenation at any stage except the pooled HMM fit**
  (D-010), which is explicit, opt-in, and recorded in `hmm_pool`.
- Progress per recording and overall; cancel leaves a consistent tree.
- Figures toggleable off (stage 09 alone emits ~6 PDFs per animal — and after
  stage 02 "per animal" means per segment, so counts rise).
- Skip-if-complete / force-rerun, driven by the manifest.
- Failure isolation: recording 3 failing does not stop 4…N; the failure, its
  traceback and its stage are recorded in the run log and surfaced in the UI.

**Acceptance:** queue N recordings across ≥2 genotypes, run unattended, return
to N complete output trees, a run log, and manifests with full parameters and
metadata. An induced failure in one recording does not affect the others.

---

### Phase 5 — Aggregation and export

- Pooled HMM fit across all recordings in the analysis so states are comparable
  between conditions (D-002); per-condition fitting available as a diagnostic.
- **The pooled fit keys every sequence on `(recording_id, track_id)`, never
  `track_id` alone** (D-015). With segmentation off, every recording contains
  `trajectory_0_N`, and keying on the ID alone would silently merge animals
  from different recordings into one sequence. An explicit test asserts that
  two recordings sharing a track ID yield two sequences.
- Stored state indices are never reordered; φ-ordering for display is read
  from the manifest's `hmm_state_mean_phi` (D-016).
- Group-by any registry metadata key(s).
- Cross-condition figures: state occupancy, dwell times, steering parameters by
  state × condition, head-cast rate, crawl length.
- **Respect the nesting:** segment within larva within recording within
  condition. Stage 02 makes this four levels, not three — several segments from
  one larva are *not* independent observations. Report n at each level on every
  comparison figure, and make the aggregation unit selectable (segment / larva /
  recording) with larva as the default.
- "Export figures" — consistent styling, publication DPI, vector output, one
  directory, with a figure manifest.

**Acceptance:** from a registry of ≥2 genotypes, produce a comparison figure
set in one action, with n reported at segment, larva and recording level, and
an exported directory suitable for a manuscript.

---

## 6. UI map

```
┌─ Project ──────────┬──────────────────────────────────────────┐
│ registry, output   │  [Ingest 00/01] [★ Segmentation 02]      │
│ root, active rec   │  [Smooth 03] [Body 04] [RDP 05]          │
│                    │  [HMM 06] [Filter 07] [HMM-Seg 08]       │
│                    │  [Events 09] [Steps 10] [Steering 11]    │
│                    │  [Trajectory Explorer] [Batch] [Aggregate]│
├────────────────────┼──────────────────────────────────────────┤
│ stage status list  │  params (left)  │  figure canvas (right) │
│ ✓00 ✓01 ✓02 ⚠03 ○05│  Run / Reset    │  output preview table  │
└────────────────────┴──────────────────────────────────────────┘
```

`⚠` = completed but stale because an upstream parameter changed.

### 6a. Segmentation tab (02) — detail

Two linked panels, both live-updating as the parameters change:

**Coverage timeline (top).** One row per larva across the full recording
duration. Each contiguous kept segment drawn in its own colour; gaps and
too-short segments in grey. This is the at-a-glance answer to "how much of this
recording is real".

**Trajectory grid (bottom).** One panel per larva, trajectory coloured by
segment, with the naive gap-bridging chord drawn underneath in pale grey so the
spurious straight line that upstream would analyse is visible. Click a panel to
open that larva in the Trajectory Explorer.

**Controls:** `bridge_max_frames` and `min_segment_seconds`, both live. The
segment counts in the panel titles and the summary table update as you drag, so
the tolerance is chosen by looking rather than guessing.

**Summary table:** per larva — tracked %, gap count, longest gap, segments
found, segments kept, total kept duration. Plus a recording-level line:
larvae in → segments out → segments dropped.

A working prototype of this figure exists (`track_segmentation_preview.png`,
generated from `amiGA-amRNAi_n1_att2.csv`); use it as the layout reference.

---

## 7. Parameter inventory

Upstream values, to be used as defaults. Changing a default silently changes
the science — match these exactly.

| Stage | Parameter | Default | Notes |
|---|---|---|---|
| 00 | `measurements_to_keep` | 20 FIM-Track channels | **Verified present** in a real export; see §7a |
| 00 | `max_frame` | auto from file | Was a call-site constant; derive and validate |
| 00 | `concatenate_recordings` | **removed** | D-010 — never pool recordings here |
| 00 | track relabel / gap interpolation / static removal | disabled upstream | Expose as opt-in, default off; superseded in practice by stage 02 |
| 01 | `frame_rate_fps` | `10.0` | Per recording. **Confirm against your rig** |
| 01 | `millimetres_per_pixel` | `240/2048` | Per recording. Validation file's coordinate range (~1300–1500 px) is consistent with a 2048 px sensor, but confirm arena width |
| 01 | `min_duration_seconds` | `60.0` | **Moves to stage 02**, applied per segment |
| **02** | `bridge_max_frames` | `10` | Gaps ≤ this are interpolated; longer ones split |
| **02** | `min_segment_seconds` | `60.0` | Inherited from upstream 01's intent |
| **02** | `enabled` | `True` | Set `False` to reproduce upstream behaviour for the baseline test |
| 03 | `window_size` | `41` | Forced odd; applied per segment |
| 03 | `polynomial_order` | `3` | |
| 05 | `epsilon_factor` | per body length | ε set from each segment's mean body length |
| 06 | `n_states` | `3` | |
| 06 | `n_iter` | `1000` | |
| 06 | `random_state` | `42` | Keep — baseline test depends on it |
| 06 | `mu_phi` | `0.8` | Manual init, reproduces upstream choice |
| 06 | `sigma_phi_fast` | `0.15` | |
| 06 | `sigma_phi_mixed` | `1.00` | |
| 07 | `MIN_RUN` | `3` | |
| 08 | `MIN_STEPS` | `9` | |
| 09 | `HC_BODYAMP_MIN_DEG` | `30.0` | |
| 11 | `MIN_POINTS_PER_RUN` | `8` | |
| 11 | `CLIP_OUTLIERS` / `CLIP_PCTS` | `True` / `(1, 99)` | |
| 11 | `KAPPA_MIN` | `0.05` | |
| 11 | `MAX_LAG` | `4` | |
| 11 | `MIN_EVENTS_PER_SEG_AUTOCORR` | `4` | Upstream notes ≥ `MAX_LAG+2` recommended |
| 11 | `REQUIRE_STABLE` | `True` | |

### 7a. Verified input format

Confirmed against `amiGA-amRNAi_n1_att2.csv`:

- Header row: `"", larva(0), larva(1), …, larva(N)`
- Row labels: `<channel>(<frame>)`, e.g. `mom_x(0)` … `mom_x(9850)`
- Missing values: **empty string**, not `-1`, not `nan`
- 30 channels exported; all 20 the pipeline needs are present
- Unused channels present and safely ignorable: `acc_dst`, `acceleration`,
  `bending`, `dst_to_origin`, `go_phase`, `left_bended`, `mom_dst`,
  `mov_direction`, `right_bended`, `velocity`

The validator should accept extra channels silently and fail loudly only on
missing required ones.

---

## 8. Figure inventory

★ = new, not present upstream.

**00 Ingest ★** — raw table preview; channel validation report; larva count;
frame count and implied duration at the configured fps.

**01 Calibrate ★** — all-tracks overview in mm; coordinate-range sanity check
against the configured px→mm scale.

**02 Segmentation ★★** — coverage timeline + trajectory grid, per §6a. The
highest-value diagnostic in the application.

**03 Smooth ★** — raw vs Savitzky–Golay overlay on a zoomable COM trace,
per segment. Justifies window 41 / polyorder 3.

**04 Body length ★** — per-segment mean ± SD strip plot; distribution.
Outliers propagate into RDP ε.

**05 RDP ★** — RDP polyline overlaid on the raw trajectory with a live ε
slider; step-length, speed and θ distributions; φ on polar axes. Plus an
assertion-backed check that no step spans a segment boundary.

**06 HMM** — upstream: φ by state (polar), φ overlay (polar).
★ transition-matrix heatmap; per-segment state raster; posterior ribbons;
state-coloured trajectories; log-likelihood convergence trace.

**07 Filter runs ★** — before/after raster; run-length histogram with
`MIN_RUN` marked; count removed.

**08 HMM segments ★** — segment-length distribution against `MIN_STEPS`;
sortable inventory table linking to the explorer.

**09 Events** — upstream: anterior velocity, linear speed, posterior/anterior
probability, head-cast trajectories, circular trajectory (per track).
★ population ethogram: all tracks as stacked crawl/head-cast raster rows.

**10 Step models ★** — event-level and run-anchor step representations on the
same trajectory.

**11 Steering** — upstream: segment trajectories, 4-panel weighted parameter
boxplots, 4-panel weighted KDEs, head-cast rate boxplot, circular
autocorrelation. ★ per-state trajectory gallery; AR(2) diagnostics.

**Trajectory Explorer ★** — layered interactive view; see Phase 3.

**Aggregation ★** — state occupancy, dwell time, steering parameters by
state × condition, head-cast rate — each with n at segment, larva and
recording level.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Tracking gaps silently corrupt every downstream stage** | Stage 02; assertion that no RDP step spans a segment boundary; coverage figure surfaced before anything else runs |
| **Over-splitting**: a too-small `bridge_max_frames` shreds good tracks into sub-60 s fragments and discards them | Live controls with counts; summary table shows total kept duration so the cost of a tolerance is visible |
| Segments from one larva treated as independent n | Aggregation unit selectable, larva by default; n reported at three levels |
| No upstream data to verify against | Own-data frozen baseline (D-009); email the author in parallel |
| matplotlib + threads: silent corruption, hangs | No pyplot; compute in worker, draw on main thread; `FigureCanvasAgg` for batch |
| Recursive RDP blows the stack; batch bottleneck | Iterative rewrite, asserted identical (D-006) |
| Stage 09 figure count rises after segmentation | Figures toggleable; off by default in batch |
| HMM state identity differs between fits | Pooled fit (D-002) + manual initialisation pins state roles + stored indices never reordered + fitted mean φ per state and `hmm_pool` in manifest; φ-ordering at display only (D-016) |
| **Upstream schema drift between stages** (script 11 expects columns script 08 does not write) | `schema.py` validates the columns at every stage boundary; stage 11 constants corrected (D-014); the stage 11 baseline gap is recorded, not hidden |
| Track IDs collide across recordings in the pooled fit | Key on `(recording_id, track_id)` everywhere recordings are combined; explicit test (D-015) |
| Accidental cross-genotype pooling | D-010; concatenation removed from stage 00; pooling only at the explicit HMM fit |
| Column-name drift as code grows | All names in `schema.py`; test forbidding known column-name literals elsewhere |
| Upstream behaviour changed by tidying | `legacy/` read-only; baseline test gates every phase |

---

## 10. Working protocol

One Claude Code session per phase (or sub-phase). Open with:

> Implement Phase N of docs/SPEC.md. Follow CLAUDE.md. Append any choices you
> had to make to docs/DECISIONS.md before finishing.

Return to the planning conversation for scientific or design questions —
figure review in particular, where the question is whether a smoothing window
is eating real head casts rather than whether the code runs.
