# DECISIONS.md

Append-only decision log. Numbered, newest at the bottom.

Append an entry whenever a choice is made that `SPEC.md` did not pre-decide:
a library, a data structure, an algorithm, a trade-off, a scientific
convention. Do not silently reverse an existing decision — append a new entry
proposing the reversal, mark the old one `Superseded by D-NNN`, and say so in
your session summary.

**Format:**

```
## D-NNN — <short title>
**Date:** YYYY-MM-DD · **Status:** Accepted | Superseded by D-NNN · **Phase:** N

**Context.** What forced a choice.
**Decision.** What was chosen.
**Reasoning.** Why, and what was rejected.
**Consequences.** What this commits us to; what would have to change to undo it.
```

---

## D-001 — Fork scripts 09 and 11; import the rest
**Date:** 2026-10-08 · **Status:** Superseded by D-013 · **Phase:** 1

**Context.** The upstream scripts fall into two shapes. Scripts 00, 01, 03, 04,
05, 07, 08 and 10 are function-based with thin `main()` wrappers. Scripts 09
and 11 are module-level: `09` executes `df = pd.read_csv(INPUT_CSV)` and its
per-animal loop at import time; `11` carries ~90 lines of hardcoded module-level
config. Neither can be imported without executing.

**Decision.** Import the well-formed scripts' functions from
`legacy/upstream_scripts/`. Fork 09 and 11 into `core/stages.py`, lifting
top-level code into functions and config blocks into dataclasses. Upstream
copies of all eleven stay in `legacy/`, read-only.

**Reasoning.** Importing preserves provenance where it is cheap. For 09 and 11
there is no import-without-side-effects path, so a fork is unavoidable. Keeping
the originals read-only preserves the manuscript link and gives the regression
test a ground truth.

**Consequences.** Forked code can drift from upstream. The regression test in
`tests/test_regression.py` is the guard and must pass before any phase closes.
If upstream publishes a revision, 09 and 11 need a manual diff.

---

## D-002 — Pooled HMM fit across conditions
**Date:** 2026-10-08 · **Status:** Accepted; state-ordering clause Superseded by D-016 · **Phase:** 5

**Context.** Upstream `06_fit_hmm_on_phi.py` pools all animals into a single
3-state Gaussian HMM fit on φ. With multiple genotypes (and later ages), the
fit could be pooled across conditions or run per condition. States from
separate fits are not automatically comparable.

**Decision.** Fit the HMM pooled across all recordings in an analysis, giving
one shared state definition. Per-condition fitting is available as a diagnostic
to check that three states is appropriate for each group, but is not the basis
of the primary comparison. States are canonically ordered by fitted mean φ so
that state indices and colours are stable across runs.

**Reasoning.** A pooled fit defines states once, so occupancy, dwell time and
steering parameters are directly comparable between conditions. Per-condition
fitting describes each group optimally but requires post-hoc state matching —
an assumption that must be defended and that fails outright if a condition
lacks three separable modes.

The cost is real and should be stated in any write-up: a genotype with
genuinely different geometry is described in states derived partly from the
others. The per-condition diagnostic exists to detect exactly that.

**Consequences.** Adding a recording changes the fit for every recording in the
analysis, so results are analysis-scoped and the manifest must record which
recordings were in the pool. Reviewers will ask about this; the reasoning above
should appear in the methods.

---

## D-003 — Conditions as generic metadata, not named columns
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** The immediate need is genotype comparison; age comparison is
expected later for the ageing work package. A `genotype` column hardcoded
through the pipeline would require a migration when age arrives.

**Decision.** `dataset_registry.csv` maps `recording_id` and `path` to
arbitrary additional metadata columns. Nothing downstream knows the name
`genotype`. Aggregation groups by whichever metadata keys the user selects.

**Reasoning.** Costs nothing now; avoids a refactor through manifest schema,
batch runner and aggregation UI later. Also handles crossed designs
(genotype × age) without further change.

**Consequences.** The aggregation UI must let the user pick grouping keys
rather than assuming one. Metadata values are free text, so typos become
spurious groups — the registry editor should offer existing values for
autocomplete.

---

## D-004 — Canonical column schema with a load/save adapter
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** Upstream CSVs mix English and Portuguese (`animal ID`,
`tempo_inicial (s)`, `estado`, `estado_filtrado`, `tipo`, `dX`, `dY`, `larva`).
These names appear in the manuscript and in distributed data files. Script 10
already contains an independent column sniffer, `detect_columns_traj`.

**Decision.** `core/schema.py` canonicalises to snake_case on load and restores
the original upstream names on save. Internal code uses canonical constants
only; no column-name string literals outside `schema.py`. Script 10's sniffer
logic is folded into `schema.py` rather than kept as a second path.

**Reasoning.** One vocabulary internally, full backward compatibility
externally. A single detection path is testable; two paths will diverge.

**Consequences.** Every new column needs registering in `schema.py`. Output
CSVs keep mixed-language names by design — this is deliberate, not an
oversight.

---

## D-005 — No pyplot in `core/` or `plots/`
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 2

**Context.** All upstream figures are written with `plt.savefig` using pyplot
global state. The application needs the same figures both embedded in Qt
canvases and written to disk by batch workers running off the main thread.

**Decision.** `plots/` factories construct `matplotlib.figure.Figure` directly
and return it; they never save. Callers attach `FigureCanvasQTAgg` for display
or `FigureCanvasAgg` for export. `pyplot` is permitted only in `legacy/`.

**Reasoning.** pyplot's global figure manager is not thread-safe and leaks
figures across repeated runs. Returning `Figure` objects gives one factory
serving both destinations, and makes figures testable headlessly.

**Consequences.** Upstream plotting code cannot be copied verbatim; each figure
is ported to the explicit-`Figure` idiom. Phase 2 must verify ported figures
are visually equivalent to upstream.

---

## D-006 — Iterative RDP rewrite
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** `05_extract_rdp_steps.py` implements Ramer–Douglas–Peucker
recursively in pure Python. Long trajectories risk `RecursionError`, and the
pure-Python inner loop is expected to be the batch bottleneck.

**Decision.** Rewrite RDP iteratively (explicit stack), with a test asserting
identical output vertices to the recursive implementation on every bundled
trajectory. Raising `sys.setrecursionlimit` is not an acceptable alternative.

**Reasoning.** RDP's recursion depth scales with trajectory complexity and is
not bounded by anything the user controls. The iterative form is a contained,
verifiable change with identical semantics. A raised recursion limit converts a
clean exception into a process crash.

**Consequences.** This is the only sanctioned divergence from upstream code in
an imported script. If the identity test ever fails, the rewrite is wrong — the
expected values are never updated to match.

---

## D-007 — Stage results are pure; persistence is the pipeline's job
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** Upstream stages take `(input_csv, output_csv)` and write to disk
themselves, with paths derived from `__file__`. The GUI needs to run a stage,
show its result, let the user change a parameter and re-run — without a disk
round-trip each time, and without a fixed directory layout.

**Decision.** `core/stages.py` functions take dataframes plus a params
dataclass and return a `StageResult` (output dataframes, diagnostics,
warnings). They perform no I/O. `core/pipeline.py` owns reading, writing,
caching and the output tree.

**Reasoning.** Pure stages are testable without fixtures on disk, re-runnable
cheaply for interactive parameter sweeps (the RDP ε slider depends on this),
and free of the `parents[1]` path assumption that makes the upstream repo
unrunnable from a plain clone.

**Consequences.** Memory holds full dataframes per stage. If a recording is
large enough for this to matter, `pipeline.py` gains a spill-to-disk path —
the stage interface does not change.

---

## D-008 — New stage 02: split tracks into contiguous segments
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** A real recording (`amiGA-amRNAi_n1_att2.csv`, 8 larvae, 9851
frames at 10 fps) was validated against the pipeline's assumptions. Seven of
eight larvae have tracking gaps — encoded as empty cells, not `-1` — with
individual gaps up to 4853 frames (≈ 8 min). Three larvae are tracked under 56%
of the window. Upstream's gap-interpolation, relabeling and static-removal
helpers exist in script 00 but are disabled in the published path.

Upstream `01_calibrate_and_filter_trajectories.py` has a docstring/implementation
mismatch: the docstring says it "removes trajectory *segments* shorter than
60 s", but the code computes `groupby("larva")["time"].max() - min()` — span,
not contiguous tracked time. A larva tracked 45% of a 985-second window passes
the filter as a 985-second track.

Downstream, this produces: NaN propagation through Savitzky–Golay smoothing;
RDP chords drawn straight across gaps, entering the HMM as single enormous
steps with nonsense speed and turning angle; and crawl/head-cast events
invented at gap seams.

**Decision.** Add stage 02 (upstream left the number vacant). It bridges gaps
shorter than `bridge_max_frames` by interpolation, splits on anything longer,
assigns each surviving run its own track ID
(`<recording_id>__larva<N>__seg<K>`), and applies the ≥ 60 s minimum per
segment. `min_duration_seconds` moves from stage 01 to stage 02. Every
downstream stage treats a segment as an independent animal track.
`enabled=False` reproduces upstream behaviour for baseline testing.

**Reasoning.** This is what upstream's own docstring describes, so it is better
read as implementing upstream's stated intent than as departing from it. The
alternative — interpolating across multi-minute gaps — fabricates trajectory
data. Dropping affected larvae entirely would discard most of the recording: on
the validation file, segmentation converts 8 nominal tracks into 21 real ones.

**Consequences.** Numerical output will differ from a naive upstream run on
gappy data, by design. This must be stated in any methods section. Sample-size
accounting gains a level — segments nest within larvae — so Phase 5 must not
treat segments as independent replicates; the aggregation unit is selectable
and defaults to larva. A too-aggressive `bridge_max_frames` shreds good tracks
into sub-threshold fragments, so the parameter needs the live visual feedback
specified in SPEC §6a rather than a fixed default chosen blind.

---

## D-009 — Parity baseline from own data, not the upstream repo
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1 ·
**Supersedes the data assumption in** D-001's consequences and CLAUDE.md §6

**Context.** `CLAUDE.md` §6 and SPEC Phase 1 originally assumed the upstream
repo's bundled CSVs could serve as regression fixtures. They cannot: the repo's
README documents a `data/` folder of ten CSVs plus `docs/`, `outputs/` and
`requirements.txt`, but the single published commit contains only eleven `.py`
files, the README, and two CSVs — `mean_body_length_by_trajectory.csv` and
`representative_hmm_segments.csv`. Both are mid-branch intermediates whose
inputs are absent, so **no upstream stage can be run end to end from what is
published**.

**Decision.** Run the unmodified upstream scripts once on one real recording
with segmentation disabled, freeze those outputs in `tests/data/baseline/`, and
assert the refactor reproduces them under `assert_frame_equal`. In parallel,
request the missing `data/` folder from the repository author.

**Reasoning.** The frozen baseline verifies the property that actually protects
the refactor — "my port does not change behaviour" — which is the failure mode
a regression test exists to catch. It does not verify "my port matches the
manuscript"; only the author's data can do that, hence the parallel request.
Proceeding with no parity test at all was rejected: the 09/11 fork is too large
to port unverified.

**Consequences.** The baseline is only as trustworthy as the one recording it
was generated from, and it inherits any upstream bug that is silent on that
data. If the author supplies the real `data/` folder, regenerate the baseline
from it and append a superseding entry. Baseline fixtures must be committed
with the recording they came from identified in a README beside them.

---

## D-010 — One recording per registry row; no cross-recording concatenation
**Date:** 2026-10-08 · **Status:** Accepted; folder-as-one-recording clause Superseded by D-021 · **Phase:** 1

**Context.** Upstream script 00 reads `table_1.csv` … `table_10.csv` and
**concatenates** them into one merged table, adding a recording index to keep
larva IDs unique. That is appropriate when every recording is the same
condition. The user's data is one CSV per recording, spanning multiple
genotypes.

**Decision.** The registry's `path` accepts a single CSV or a folder. Each
registry row is one recording, processed independently end to end. The
concatenation behaviour is removed from stage 00. The only place data from
multiple recordings is combined is the pooled HMM fit (D-002), which is
explicit, opt-in, and recorded in the manifest's `hmm_pool` field.

**Reasoning.** Silent pooling across genotypes would invalidate every
comparison the application exists to make, and would do so without any visible
symptom. Removing the capability is safer than keeping it behind a flag.
Segment IDs are already recording-scoped (D-008), so upstream's recording-index
trick is unnecessary.

**Consequences.** A user who genuinely wants several files treated as one
recording must point a registry row at a folder — which is supported, and is
the explicit way to ask for it. Stage 00's output is per recording, so the
`max_frame` parameter is derived per file rather than shared.

---

## D-011 — Optional filename-pattern parsing to pre-fill registry metadata
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** The user's filenames already encode condition metadata:
`amiGA-amRNAi_n1_att2.csv` carries genotype, replicate and attempt. Typing
these into a registry table by hand across dozens of recordings is slow and
produces typos, and a typo silently creates a phantom experimental group.

**Decision.** A user-editable regex with named groups maps filenames to
metadata columns — e.g. `(?P<genotype>.+?)_n(?P<n>\d+)_att(?P<attempt>\d+)`.
Results are always previewed in the registry table before being applied, always
overridable per row, and never applied automatically on import. The registry
editor also offers existing values as autocomplete.

**Reasoning.** Pre-filling is a convenience, not a source of truth; the preview
and override keep the registry authoritative. Named groups map directly to
metadata column names, so the generic-metadata design (D-003) needs no special
case. A non-matching filename leaves metadata blank rather than guessing.

**Consequences.** The pattern is per project and belongs in project settings,
not in code. Filenames that do not match must be visibly flagged in the
registry table rather than silently skipped — an unlabelled recording that
reaches aggregation is a silent error.

---

## D-012 — Phase 0 scaffold: environment, legacy integrity check, data location
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 0

**Context.** SPEC Phase 0 asks that `legacy/` be byte-identical to the upstream
clone, but no upstream clone is present in the workspace to compare against.
The only Python installs on the development machine were 3.14 conda envs
without hmmlearn, scikit-learn or pytest. The validation recording (15 MB) was
placed in `data/`.

**Decision.**
- A dedicated conda env `larval`: Python 3.11 from conda, every package
  installed with pip from PyPI (`pip install -r requirements.txt`), pandas pinned
  `>=2.2,<3`, PyQt5 5.15, plus the stack in CLAUDE.md §7.
- Legacy integrity is enforced by SHA-256: `tests/data/legacy_sha256.txt`
  records the hashes of the vendored files as received, and
  `tests/test_legacy_unmodified.py` fails if any file changes, is added or is
  removed. `.gitattributes` disables line-ending conversion under `legacy/`.
- `data/` and `outputs/` are git-ignored; raw recordings are not committed.
- `CLAUDE.md` lives at the repository root, as its own §2 specifies.

**Reasoning.** Python 3.11 matches the manifest example in SPEC §4.2 and has
mature wheels for hmmlearn. Packages come from PyPI rather than conda-forge
because Smart App Control on the development machine blocks two conda-forge
DLLs (`graphite2.dll`, `libexpat.dll`), which breaks matplotlib import and
crashes numpy/MKL linear algebra; the PyPI wheels load cleanly. pandas is held below 3 because upstream code relies
on pandas-2 semantics (e.g. `shift(1).fillna(False)` followed by `~` in script
11), and a silent dtype change there would alter results. Checksums give the
"never edited" rule a test without needing the upstream clone.

**Consequences.** The checksums prove the files have not changed since they
were vendored, not that they match upstream; if a clone becomes available,
compare once and note it here. The frozen baseline (D-009) will contain files
derived from a raw recording that is itself not in the repo — the baseline
README must identify it by name and hash.

---

## D-013 — Amend D-001: fork at the function body, not at `main()`
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1 · **Supersedes** D-001

**Context.** D-001 judged scripts 00, 01, 03, 04, 05, 07, 08 and 10 "clean
enough to import" from the shape of their signatures. Reading the bodies shows
otherwise. `calibrate_and_filter(input_csv, output_csv, ...)` in script 01
calls `pd.read_csv` on its first line and `.to_csv` on its last. The same holds
for `raw_to_wide_format` and `process_recordings` (00),
`smooth_trajectory_timeseries` (03), `compute_mean_body_length` (04),
`extract_rdp_steps` (05), `filter_hmm_runs` (07),
`extract_representative_segments` (08) and `fit_hmm_on_phi` (06), which also
writes its figures. Script 10 keeps all of its logic inside an argparse
`main()`. For these the named function *is* the I/O wrapper; there is no pure
stage function underneath to import, and D-007 forbids I/O in a stage.

Two further constraints narrow what can be imported at all. Scripts 06, 09 and
11 import `matplotlib.pyplot` at module level (09 also executes its whole
pipeline, 11 also sets `rcParams`), so importing anything from them breaks the
layering rule in CLAUDE.md §3. And the file names start with a digit, so any
import goes through `importlib`, not an `import` statement.

**Decision.** Fork the body of every stage function into `core/stages.py`.
Import only genuinely pure leaf helpers from `legacy/upstream_scripts/`:

- `euclidean_distance` (04)
- `perpendicular_distance`, `build_steps_and_angles` (05)
- `rdp` (05) — as the recursive reference for the D-006 identity test only;
  the pipeline runs the iterative rewrite
- `run_length_encode`, `filter_state_sequence_one_pass` (07)
- `angle_wrap_pi`, `interp_pos`, `build_step_rows` (10)

Everything in 00, 01, 03, 06, 09 and 11 is forked. In 08,
`extract_segments_for_one_trajectory` is forked too: it performs no I/O, but it
reads the module globals `MIN_STEPS`, `TIME_COL`, `STATE_COL` and `ID_COL`, so
its threshold cannot be supplied from a params dataclass (CLAUDE.md §5) without
patching the legacy module. Script 11's numerical helpers (`fit_ar2`,
`ar2_to_params`, `fit_vonmises`, the weighted statistics, the circular
autocorrelation) are pure in themselves but unreachable without importing
pyplot, so they are forked verbatim. `detect_columns_traj` (10) is folded into
`schema.py` as D-004 already requires.

**Reasoning.** The import-versus-fork distinction in D-001 does not survive
contact with the code. What survives is that `legacy/` stays the read-only
parity reference. Porting bodies verbatim keeps behaviour identical while
removing the path and I/O assumptions.

**Consequences.** More code is forked than D-001 implied, so the baseline
parity test (D-009) carries more weight. Port bodies exactly; do not tidy,
rename or reformat while porting. Most bodies are 20–40 lines, but three are
not: script 09's per-track loop (~190 lines of analysis), script 10's `main()`
(~120) and script 11's `main()` (~260) — these remain the bulk of the porting
effort. The imported helpers `build_steps_and_angles` and `build_step_rows`
emit upstream column names; their output is canonicalised by the `schema.py`
adapter at the stage boundary, which is the one place upstream names cross
into `core/` (D-004).

---

## D-014 — Script 11 cannot read script 08's output (upstream bug)
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** Script 08 writes the columns `animal ID, state, start_index,
end_index, start_time_s, end_time_s, n_steps, duration_s`. The published
`legacy/upstream_data/representative_hmm_segments.csv` has exactly that header.
Script 11 declares `SEG_T0_COL = "t_inicio (s)"` and
`SEG_TF_COL = "t_fim (s)"` and validates them on load, raising `ValueError` if
either is absent. Neither name occurs anywhere else in the upstream repo.

So script 11 raises on script 08's output and on the published CSV, and the
README's quick-start sequence (08 then 11) is broken. These look like stale
constants left behind when the working files were renamed for publication; the
README documents a file-renaming pass, though its table lists file names only,
not column names.

**Decision.** In the forked stage 11, set the two constants to `start_time_s`
and `end_time_s`. Do not rename stage 08's output to the Portuguese names —
those names appear in no upstream artefact. `ID_COL` (`animal ID`) and
`SEG_ESTADO_COL` (`state`) already match and are left alone.
`WEIGHT_COL_SEG = "n_points_total"` is not a break: script 11 computes that
column itself into its own `segment_fits` output. Leave it.

**Reasoning.** The published CSV and script 08's code agree with each other;
only script 11 disagrees with both. Bending 08 to fit 11 would manufacture a
schema nothing upstream produces.

**Consequences.** Stage 11 was evidently never run against these inputs in
published form, so whatever produced the manuscript figures used a different
schema. Flag this to the author alongside the missing `data/` folder (D-009).
The unmodified script 11 cannot run on any available input, so the baseline
test cannot cover stage 11 end to end until that is resolved; record the gap in
the baseline README rather than papering over it. The fix assumes the two
columns carry the same meaning under both names — segment start and end time
in seconds — which is the natural reading but is unconfirmed until the author
replies.

---

## D-015 — Track ID scheme with segmentation off, and the pooled-fit collision
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1

**Context.** `process_recordings` in script 00 relabels `larva(N)` to
`trajectory_{video_index}_{N}`. Both published CSVs use this scheme
(`trajectory_0_101` … `trajectory_9_…`, ten recordings), confirming it is the
manuscript's. `raw_to_wide_format` on its own leaves the raw `larva(N)` label.

**Decision.** With segmentation disabled, track IDs are `trajectory_0_N`,
produced by running `process_recordings` on a single file (a list of one),
consistent with D-010's no-concatenation rule. With segmentation enabled, IDs
remain `<recording_id>__larva<N>__seg<K>` (D-008).

**Reasoning.** It is the scheme the published data uses, so the baseline test
compares like with like. Sort order is the binding constraint, not the HMM:
`trajectory_0_101` and `larva(101)` sort differently, script 06 groups with
`groupby`'s default `sort=True`, and the row order of every downstream CSV
follows — `assert_frame_equal` fails on that alone. (The HMM fit itself is
near order-invariant: the likelihood sums over sequences and `init_params=""`
removes k-means randomness.)

**Consequences.** Under D-010 `video_index` is always 0, so two recordings both
contain `trajectory_0_101`. That is harmless while outputs are per-recording
directories, but the pooled HMM fit (D-002) combines recordings and those IDs
collide **silently** — two animals become one sequence. The pooled fit must key
on `(recording_id, track_id)`, never `track_id` alone, and an explicit test
must assert it. Segment IDs from stage 02 already embed the recording ID, so
the collision is specific to the segmentation-off path; the composite key is
required regardless, so that correctness does not depend on which path ran.

---

## D-016 — HMM state ordering is display-only
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1 ·
**Supersedes the ordering clause of** D-002

**Context.** D-002 says states are "canonically ordered by fitted mean φ".
Applied to the stored `estado` column, that would relabel states relative to
upstream and break parity. Script 06 fits with `init_params=""` and means set
manually to `[+mu_phi, -mu_phi, 0]` in z-space, so state 0 is
persistent-turning, state 1 reversing and state 2 mixed — by construction,
before EM runs.

**Decision.** Stored state indices in all output CSVs are upstream's, never
reordered. Canonical φ-ordering is applied at display and aggregation
presentation only. The manifest records the fitted mean φ per state index
alongside `hmm_pool`. D-002's pooled-fit decision itself stands.

**Reasoning.** The manual initialisation already pins state semantics, so the
state-matching problem D-002 guarded against — which arises from random
initialisation — does not exist here. Reordering would break parity for no
benefit.

**Consequences.** The manifest's mean-φ-per-state record makes an old figure
interpretable without re-running, and lets the user confirm EM did not drift a
state away from its initialised role. Any plotting code that assumes index
order equals semantic order must read the ordering from the manifest, not
assume it.

---

## D-017 — Phase 1a choices: segmentation rules, canonical names, parameter coverage
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1a

**Context.** SPEC §2a fixes what stage 02 does but not its edge rules, and D-004
fixes that canonical names exist but not what they are. Profiling the
validation recording showed that a lost track has **no rows** for the missing
frames (upstream script 00 skips empty cells), and that no frame is partially
tracked: for every larva, either all 20 channels are present or none are.

**Decision.**

*Segmentation (`core/segments.py`).*
- Gaps are detected from jumps in the frame sequence, not from NaN. Frame
  numbers are recovered as `round(time_s × fps)` using stage 01's frame rate.
  If two rows of a track land on the same frame the stage raises rather than
  guessing.
- A gap is the number of missing frames between two tracked frames. It is
  bridged if that number is ≤ `bridge_max_frames`, otherwise it splits.
- A segment's duration is `(last_frame − first_frame) / fps`; it is kept if
  that is ≥ `min_segment_seconds` (inclusive).
- Bridged frames: every numeric channel is linearly interpolated; the two
  tracker flags (`is_coiled`, `is_well_oriented`) carry the last tracked value;
  time is built with stage 01's formula and rounding. Interpolated positions
  are not re-rounded to 3 decimals. Each such row is marked in an
  `interpolated` column.
- Segment numbers are 0-based, in time order, and count every segment found,
  kept or not, so dropping a short segment does not renumber the others. IDs
  are not zero-padded, as written in SPEC §2a.
- A row with any missing position channel counts as untracked.
- The per-larva summary counts gaps of any length against the whole recording,
  so a track that starts late or ends early has a leading or trailing gap. This
  reproduces the table in SPEC §2a exactly.
- The stage 02 output adds three columns with no upstream equivalent:
  `source_track_id`, `segment_index`, `interpolated`.

*Schema (`core/schema.py`).*
- One `TableSchema` per table kind; unknown columns pass through untouched.
- Upstream's `phi` is two quantities: turning persistence in the RDP/HMM step
  tables (canonical `phi`) and absolute heading in the stage 10 tables
  (canonical `heading`). Both are written back as `phi`.
- Schemas for stage 11's six output tables are deferred to Phase 1c, when that
  stage is forked.

*Parameters (`core/params.py`).*
- Every constant upstream wrote inline is a field, not only those in SPEC §7:
  stage 01's rounding decimals, stage 09's filter windows, peak prominences,
  threshold-fit settings, merge gap and minimum crawl duration, and stage 11's
  stability bounds, clipping minimum and KDE bandwidth floors.
- `HmmParams` rejects `n_states != 3`: upstream initialises exactly three
  states and writes three posterior columns, so another value has no defined
  behaviour.
- `ReorganizeParams.max_frame` defaults to `None` (keep every frame) instead of
  upstream's call-site constant 36000.

**Reasoning.** Each rule is the simplest one consistent with SPEC §2a and with
the reference figure; the real-recording test asserts the per-larva segment
counts, gap counts, longest gaps and tracked fractions that the SPEC and the
figure report. Counting all found segments keeps IDs stable when only
`min_segment_seconds` changes. Raising on colliding frames surfaces a real
upstream limitation — time is rounded to 0.1 s, which is only safe at ≤ 10 fps
— instead of silently mis-segmenting.

**Consequences.** Segment IDs do change when `bridge_max_frames` changes, since
that changes which segments exist. Unpadded IDs sort `seg10` before `seg2`;
anything presenting segments must sort by `(larva_index, segment_index)`, not
by ID. Recordings above 10 fps cannot be segmented until the 0.1 s rounding in
stage 01 is made a per-recording choice, which is a departure from upstream and
needs its own decision. The flag rule is untested against real data, because no
script reads the flags downstream.

---

## D-018 — Phase 1b: baseline frozen; parity is exact only through one CSV parse
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1b

**Context.** The baseline of D-009 had to be generated before any stage could
be checked, and porting stages 00–08 and 10 raised choices the SPEC did not
settle.

**Decision.**

*Baseline.* `tools/generate_baseline.py` copies the upstream scripts byte for
byte into a `scripts/ data/ outputs/` tree, runs them unmodified on
`amiGA-amRNAi_n1_att2.csv`, and freezes eleven output CSVs (gzip-compressed,
7 MB) in `tests/data/baseline/` with a README naming the recording, its hash
and the library versions. The harness does only what the upstream README
requires by hand: it calls script 00 through `process_recordings` on a list of
one (D-015), copies outputs that scripts 09 and 10 read from `data/`, and
creates the two output folders those scripts assume exist. Script 11 is not
run (D-014). The script refuses to overwrite a baseline without `--force`.

*Parity test.* `tests/test_regression.py` checks each ported stage in
isolation — fed the frozen upstream output of the stage before it — and
compares with `assert_frame_equal(check_exact=True)` after writing the result
to CSV text and reading it back. Stages 00, 01+02(disabled), 03, 04, 05, 06,
07, 08 and 10 all match bit for bit. Stage 09 follows in Phase 1c.

*One parse per reader.* Upstream re-reads a CSV between every stage, and
pandas' default float parser is not exact: on the smoothed table 11% of
centre-of-mass values come back one unit in the last place away from what was
written. The parse is also not idempotent — writing the parsed value and
parsing again moves a further 1.8%. Exact parity with upstream therefore
requires that each stage receive its input **as parsed once from the CSV the
previous stage wrote**. `pipeline.py` (Phase 1d) must hand tables on that way,
both when running in memory and when resuming from disk, so that a result does
not depend on whether a run was resumed. Chained purely in memory, the stages
still match the baseline to a relative 1e-9, with identical discrete columns;
a test asserts that too.

*Port details.*
- Stage 00 is vectorised rather than looped, with row order equal to
  upstream's first-encounter order. A channel whose values are all whole
  numbers with none missing is returned as integers, which is what upstream's
  text round trip produces.
- Upstream script 01 is split: stage 01 calibrates; stage 02 with
  `enabled=False` applies upstream's span filter and sort. Their composition
  equals upstream's output.
- The iterative RDP (D-006) computes point-to-chord distances with numpy in
  upstream's operation order. It is asserted identical to the recursive
  version on every baseline trajectory at three tolerances, on random walks
  and on degenerate shapes (stationary, collinear, tied, repeated, NaN).
- Stage 03 checks track length against the window before smoothing, so a
  too-short track fails with its name rather than a scipy error.
- Stage 06 takes `sequence_keys`, the columns identifying one animal's
  sequence, defaulting to the track ID. The pooled fit of Phase 5 passes
  `(recording_id, track_id)`; a test shows two recordings sharing a track ID
  collapse to one sequence without it (D-015).
- Stage 06 reports convergence, the log-likelihood history, the transition
  matrix and the fitted mean φ per state (D-016) as diagnostics.

**Reasoning.** Per-stage tests localise a failure to one stage and are immune
to the parser effect; the chained test proves the stages also compose. Making
the pipeline go through the saved representation costs a CSV write and read
per stage, and buys results that are identical whether computed fresh, resumed
from disk, or produced by upstream.

**Consequences.** Do not "optimise away" the write-then-read between stages in
`pipeline.py`, and never pass a table through two parses on the way to one
consumer. Interactive previews in the GUI (the RDP ε slider) may run in memory
for speed, but anything recorded as a result goes through the pipeline. The
baseline inherits upstream's gap behaviour on this recording: with segmentation
off, steps are drawn across lost frames, exactly as D-008 describes. Baseline
files are tied to pandas 2.3 / numpy 2.4 / scipy 1.17 / hmmlearn 0.3.3; a
library upgrade that shifts a float is a parity failure to investigate, not a
reason to regenerate.

---

## D-019 — Phase 1c: stages 09 and 11 forked; stage 11 verified against upstream run live
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1c

**Context.** Scripts 09 and 11 had to be forked whole (D-013). Stage 09 has a
frozen baseline. Stage 11 has none, because the unmodified script cannot read
script 08's output (D-014), and porting 500 lines of fitting code unverified
was not acceptable.

**Decision.**

*Stage 09 (`core/events.py`).* The per-track body of the upstream script is one
function; every inline constant is an `EventParams` field. The two identical
threshold blocks (posterior, anterior) call one helper. Reproduces the frozen
`event_intervals.csv` exactly (4951 events). Besides the event table the stage
returns the per-frame signals and per-track thresholds that upstream's seven
figures per track were drawn from, so `plots/` can redraw them without
re-running detection. Three upstream angles and their velocities that are
computed but never used (`A_cauda`, `A_sp2`, `A_head`) are not ported.

*Stage 11 (`core/steering.py`).* The numerical helpers are copied verbatim;
`main()` becomes `stage_11` without its plotting and file writing; the segment
times are read from `start_time_s` / `end_time_s` (D-014). It returns
upstream's six tables plus three long tables for the figures (head-cast
directions, head-cast θ, crawl lengths, each by state).

*Verification of stage 11.* `tests/test_steering.py` imports the upstream
script and (a) asserts each numerical helper returns identical values to its
upstream original on random and degenerate inputs, and (b) runs upstream
`main()` on the frozen stage 08 and stage 10 outputs — with the two column
constants set as D-014 specifies and its `plot_*` functions disabled — and
asserts all six CSVs equal `stage_11`'s tables exactly. This is done at
upstream defaults (63 run fits over 10 HMM segments) and at a relaxed setting
that admits 120 run fits.

*Departures from upstream, all without effect on valid output.*
- `shift(1).fillna(False)` in `segment_runs_within_segment` is written
  `shift(1, fill_value=False)`: same values, no pandas deprecation.
- Where upstream would raise `KeyError` — a state has a fit but no
  autocorrelation rows at all — `stage_11` reports NaN for the lag-1 head-cast
  correlation.
- `RankWarning` from the threshold fits in stage 09 is silenced; the fits are
  unchanged.
- Tracks shorter than the longest filter window fail with their name.
- The KDE grid constants inside `weighted_kde_peak` (1–99% range, 6% padding,
  500 points) are left inline as upstream wrote them rather than made
  parameters: they are numerical resolution, not analysis choices.

**Reasoning.** Running upstream live gives stage 11 the same protection the
frozen baseline gives the other stages, while changing nothing in upstream
beyond what D-014 already decided. It also cannot go stale: there is no
expected-values file to regenerate.

**Consequences.** The live comparison shows the port matches upstream's code,
not that either matches the manuscript; that still needs the author's data
(D-009, D-014). Upstream calls each track an "animal"; with segmentation on,
`n_animals` in the autocorrelation table counts track segments, which Phase 5
must account for when reporting n. Upstream script 11's own figure code no
longer runs on current matplotlib (`boxplot(labels=...)` was removed), so
Phase 2 ports those figures from reading the code, without a rendered upstream
reference to compare against unless an older matplotlib is installed for that
purpose.

---

## D-020 — Phase 1d: pipeline, manifest and registry
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1d

**Context.** SPEC §3–§4 name `pipeline.py`, `manifest.py` and `registry.py` and
give the output tree and a manifest example, but not how stale state is
detected, how a failed or cancelled stage is left, or what the files are
called.

**Decision.**

*Stage graph.* 00 → 01 → 02 → 03; 04 ← 03; 05 ← 03, 04; 06 ← 05; 07 ← 06;
08 ← 07; 09 ← 03; 10 ← 03, 09; 11 ← 08, 10. Asking for a stage runs its
dependencies first.

*Persistence (D-018).* A stage's tables are written as CSV in upstream column
names; any stage that reads them gets the table parsed back from that file,
once. This holds within a run and across a resume. Run from the raw recording
with segmentation off, the pipeline's files equal all eleven baseline CSVs
exactly.

*File names* are upstream's wherever upstream has the table. New ones:
`02_segments/track_segment_trajectories.csv`, `track_segments.csv`,
`larva_summary.csv`; `09_events/event_signals.csv`, `event_thresholds.csv`;
`11_steering/headcast_directions.csv`, `headcast_thetas.csv`,
`crawl_lengths.csv`. With segmentation off, `track_segment_trajectories.csv`
is what upstream script 01 wrote as `trajectory_timeseries_calibrated.csv`;
`01_calibrated/` holds the calibrated table before any filtering.

*Stale state.* Each stage record stores a fingerprint: a hash of the stage's
resolved parameters, the fingerprints of the stages it reads, and — for stage
00 — the SHA-256 of the raw file. A stage is `ok` if its stored fingerprint
equals the one computed from the current parameters, `stale` if not,
`missing` if it has no record or a file is gone, `failed` if its last run
raised. A stale table can be viewed but is never passed to a stage.

*Failure and cancellation.* Before a stage runs, its old record and folder are
removed, so an interruption leaves `missing`, never a mixture of old and new
files. A stage that raises is recorded as `failed` with its traceback, and the
error is re-raised naming the stage. Cancellation is checked between stages.
The manifest is written through a temporary file and renamed.

*Manifest.* As SPEC §4.2, plus per stage `fingerprint`, `outputs` and
`completed_at`, and `error` on failure. Non-finite numbers are written as
`null`. A single-recording run records `hmm_pool` as just that recording.

*Registry.* Every cell is text, so `n = 01` survives a round trip. Filename
parsing matches the pattern against the **whole** file stem; a name with
anything extra does not match and is flagged rather than half-parsed.
Applying a preview fills blanks only, unless overwrite is asked for. The two
required column names are constants in `registry.py`, not `schema.py`: they
are project bookkeeping, not pipeline data.

*Folders as recordings are refused for now.* D-010 lets a registry row name a
folder of CSVs to be treated as one recording. That needs a rule for track IDs
from several files — upstream's `trajectory_{file}_{N}`, and something for
stage 02's `<recording>__larva<N>__seg<K>`, which has no file slot — and the
SPEC gives none. `recording_files` raises a clear error on a folder.

**Reasoning.** A fingerprint makes "is this result current?" one comparison
and gets every case right without special rules: a changed parameter, a
changed upstream parameter, an edited raw file, and a parameter changed and
then changed back (stale, then current again only once re-run or restored).
Removing before writing costs nothing and makes the tree trustworthy after a
crash.

**Consequences.** Setting a parameter back to its old value makes the old
result current again without re-running, which is correct but may surprise.
A full default run of the validation recording takes about 8 s and writes
about 53 MB, mostly the four per-frame trajectory tables; for large batches a
"keep intermediates" option may be wanted. The folder-as-recording rule is an
open question for the user. Figures are not written yet (Phase 2); the
pipeline only reserves `figures/<stage>/`.

---

## D-021 — Every CSV is one recording; a folder is a way to add many
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 1d ·
**Supersedes the folder clause of** D-010 **and the "refused for now" item of** D-020

**Context.** D-010 said a registry row's `path` "accepts a single CSV or a
folder", the folder being the explicit way to have several files treated as
one recording. Implementing it exposed that no rule existed for naming tracks
from several files (D-020). The user settled it: each CSV is one recording;
several CSVs in a folder may share a condition, but each is still its own
recording.

**Decision.** A registry row always names exactly one CSV. Adding a folder to
the registry adds one row per CSV directly inside it, in name order; nothing
is merged. There is no way to treat several files as one recording. The rest
of D-010 stands: recordings are processed independently, and the only
cross-recording operation is the pooled HMM fit.

**Reasoning.** It matches how the data is produced, removes the unanswered
track-naming question, and removes the last place where files could be
concatenated before stage 02.

**Consequences.** Recording IDs come from file names, so two files with the
same name in different folders collide and the second is rejected; the user
renames one or edits the ID. A shared condition for a folder of recordings is
expressed in the metadata columns like any other, by the filename pattern
(D-011) or by hand; the folder name itself is not used as metadata.

---

## D-022 — Phase 2: plot library, palette, and where figures differ from upstream
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 2

**Context.** SPEC §8 lists the figures; Phase 2 asks that each be a factory
returning a `Figure`, that upstream figures be visually equivalent, and that
state and segment colours be stable across figures and datasets.

**Decision.**

*Structure.* `plots/style.py` holds the palette and helpers.
`segmentation.py`, `trajectory.py`, `hmm.py`, `events.py`, `steering.py` hold
the factories, which take canonical tables and return a `Figure`.
`plots/catalog.py` maps a stage to its figures built from a recording's saved
tables. Writing is `RecordingPipeline.save_figures`, on an Agg canvas, so no
factory touches disk and batch export is thread-safe (D-005).
`tools/render_figures.py` exports a processed recording's figures from a
console. Forty-one figures render for the validation recording at two tracks
per per-track figure.

*seaborn is not used.* It imports pyplot, which the layering rule forbids in
`plots/`. Upstream's two φ density figures used `sns.kdeplot(bw_adjust=0.7)`;
`style.kde_curve` computes the same estimate with scipy (Scott bandwidth ×
0.7, evaluated three bandwidths beyond the data, 200 points), and a test
checks it against scipy directly.

*State colours.* Upstream draws states 0, 1, 2 in matplotlib's blue, orange,
green. Measured for colour-vision deficiency, that orange and green are
indistinguishable to a protanope (ΔE 0.7 where 8 is the target). The same
three hues are kept but stepped to `#2a78d6`, `#eb6834`, `#1baf7a`, which pass
pairwise. A state's colour is looked up from its stored index (D-016), never
from plot order, so a figure showing only state 2 draws it in state 2's
colour. Upstream's stage 11 figures took colours from the plot-order cycle;
they now use the state colours too.

*Other colour roles.* Track segments take a fixed eight-colour order by
segment number (cycling after eight, where position on the timeline also
identifies them). Crawls are drawn dark and head casts red in every figure,
as upstream stage 09 does; upstream stage 11's blue/orange for the same two
things is not kept.

*Deliberate differences from upstream figures.*
- Stage 09's "tail trajectory" figure is not produced: upstream plotted the
  centre of mass in it, so it duplicated the "mom" figure.
- The threshold-distribution figure zooms to five times the threshold and
  states the full range in the axis label. Upstream showed the full range,
  where the fit region is a few pixels wide.
- The four stage 11 summary figures add per-state n to the axis labels, and
  the head-cast rate box plot overlays the individual HMM segments.
- Jitter in the box summaries is seeded, so a figure is reproducible.
- State labels are "State 0/1/2", as upstream; no descriptive names are
  invented.

*New figures (★ in SPEC §8)* are implemented as listed, with two
interpretations: "AR(2) diagnostics" is the fitted (a1, a2) of every run
against the stationarity triangle plus the distribution of points per run;
"per-state trajectory gallery" shows representative HMM segments by state,
longest first.

*Not in this phase.* Stage 00 has no figure — its content is the validation
report and a table, which are Phase 3 widgets. The Trajectory Explorer is
interactive and belongs to Phase 3. Aggregation figures belong to Phase 5.

**Reasoning.** "Visually equivalent" is read as same form, same quantities,
same scales — not same hex values — because the alternative is knowingly
shipping a state palette that a red-green colour-blind reader cannot use.

**Consequences.** Figures in a manuscript will not be pixel-identical to
upstream's in colour. Per SPEC §10, whether a figure is scientifically right —
whether the smoothing window eats head casts, whether the threshold fit is
sound — is for review with the user; these factories only guarantee that the
figure draws what the tables contain. No rendered upstream image exists for
the stage 11 figures (D-019), so their equivalence rests on porting the
drawing code line by line.

---

## D-023 — Phase 3: single-recording GUI
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 3

**Context.** SPEC Phase 3 and §6 describe the tabs and the acceptance test but
not how the window is put together, how threads are used, or how parameters
are edited.

**Decision.**

*Layout.* Left: the active recording, the twelve stages with their state
(✓ current, ⚠ stale, ○ not run, ✗ failed), Run all and Cancel. Right: tabs for
Project, Ingest 00/01, Segmentation 02, one tab per stage 03–11, and the
Trajectory Explorer. Batch and Aggregate tabs arrive with Phases 4 and 5.

*A project* is a folder with `project.json` (output folder, filename pattern)
and `dataset_registry.csv` (`core/project.py`). The recording selected in the
registry table is the active recording. Reopening a project restores each
recording's stage states and the parameters it was last run with.

*Parameters.* Each stage's form is generated from its parameter dataclass.
Whole numbers use spin boxes; **floats are edited as text showing every
digit**. A decimal spin box rounded `240/2048 = 0.1171875` to `0.117188`,
which would have changed the calibration of every recording merely by opening
the form; a test now checks that every form returns exactly the parameters it
was given. Editing a field marks that stage and everything downstream stale
immediately. A value the stage rejects (for example `n_states` other than 3)
is shown as an error and not applied.

*Threads.* Pipeline runs go through `PipelineWorker`, one at a time. While a
run is in progress the pipeline object belongs to the worker and the window
takes stage states from the worker's signals. Reading the raw file for the
validation report, the live segmentation preview and "Export all figures" go
through `FunctionWorker`. Figures shown on screen are built on the GUI thread,
one at a time, only when selected; exported figures are built in the worker
on Agg canvases (CLAUDE.md §9). Results reach the GUI thread through a
`QObject` relay.

*Segmentation tab.* Dragging either tolerance recomputes the segmentation in
memory after a short pause and redraws the coverage timeline, the trajectory
grid and the per-larva summary. Nothing is recorded until "Apply and run stage
02". Clicking a larva panel opens its first kept segment in the explorer.

*Trajectory Explorer.* Eight layers switch independently (raw, bridged
frames, smoothed, RDP steps, HMM states, filtered states, representative HMM
segments, head casts). Toggling a layer keeps the current zoom. The time
slider moves a marker without redrawing. Stale stages are shown as last
computed, with a note saying which.

*Robustness found while building it.* A recording in which stage 09 detects
no events made stage 11 fail on a missing column; stage 11 now returns empty
tables and a warning. Stage 06 with no step that has a φ now fails with an
explanation instead of a scikit-learn message.

**Verification.** `tests/test_gui.py` drives the window without a display:
create a project, add a recording, parse file names, run every stage by
clicking, and compare every output file byte for byte with a headless
pipeline run; reopen and check restored state; edit a parameter and check
stale marking; exercise the segmentation preview, the explorer and a failing
stage. Screenshots of the window on the validation recording were inspected.

**Consequences.** The window has not been operated by a person on a real
display in this phase; behaviour that only appears with a mouse (drag
performance of the sliders on large recordings, dialog flows) is unverified.
Confirmation dialogs (reset, remove recording) are modal and are not covered
by the tests. One recording runs at a time; running several is Phase 4.

---

## D-024 — Phase 4: batch processing
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 4

**Context.** SPEC Phase 4 asks for a queue of recordings with a stage range,
concurrency, cancellation, optional figures, skip-if-complete, failure
isolation and a run log. It does not say which parameters a batch applies, or
what the log contains.

**Decision.**

*One function per recording.* `core/batch.py: run_recording` processes one
recording through the ordinary `RecordingPipeline` and never raises: a failure
comes back as an outcome carrying the stage, the message and the traceback.
The console entry point `run_batch` runs it on a thread pool; the Batch tab
runs the same function on a `QThreadPool` of its own, one recording per
worker. Default two recordings at a time. Nothing is combined across
recordings (D-010): each manifest's `hmm_pool` is that recording alone.

*Stage range* is "run through stage N": the dependencies of N run first, and
anything already current is skipped, driven by the manifest fingerprints
(D-020). "Re-run" forces every stage in the range.

*Which parameters.* A batch either applies one parameter set to every
recording — in the GUI, the set currently shown in the stage tabs, and this is
the default — or lets each recording keep the parameters it was last run with.
The first is the default because recordings analysed with different settings
are not comparable.

*Figures* are off by default and written on request after a recording's
stages finish. Figure drawing is serialised across workers with a lock,
because matplotlib's font and text-layout caches are shared between figures
even without pyplot.

*Cancel* stops each running recording after its current stage and marks
recordings not yet started as cancelled. Finished stages are kept, so a later
batch resumes where this one stopped.

*Run log.* Every batch writes
`<output_root>/run_logs/batch_<timestamp>.json`: start and end time, the
recordings requested, the options and the full parameter set if one was
applied, library versions, a per-result count, and one entry per recording
with its result, stages run and skipped, duration and, on failure, stage,
message and traceback. It is rewritten after each recording, so an
interrupted batch still leaves a log.

*In the window.* While a batch runs, single-recording runs are disabled.
Double-clicking a recording in the batch table makes it the active recording,
with its results loaded, for inspection in the stage tabs.

**Reasoning.** Reusing the single-recording pipeline means a recording
processed in a batch is byte-identical to one processed alone, and inherits
resume, stale detection and failure records without new code. Threads rather
than processes keep cancellation and progress simple; the numerical work
releases the interpreter lock often enough for two workers to help.

**Consequences.** "Same parameters for every recording" includes frame rate
and pixel scale. A project mixing rigs must use "each recording keeps its
own" after setting calibration per recording. Re-running a stage on unchanged
inputs leaves downstream stages current, since their fingerprints do not
change. Throughput on large batches has not been measured; if threads prove
too slow, `run_recording` can be handed to a process pool unchanged.

---

## D-025 — Phase 5: pooled HMM fit, aggregation by condition, figure export
**Date:** 2026-10-08 · **Status:** Accepted · **Phase:** 5

**Context.** SPEC Phase 5 asks for a pooled HMM fit, grouping by any metadata,
five kinds of cross-condition figure with n at three levels and a selectable
unit, and an exported figure set. It does not define the measures, how a
pooled fit relates to each recording's own stage 06, or what is exported.

**Decision.**

*The pooled fit replaces stage 06 of each recording in the pool.* The RDP
steps of the chosen recordings are fitted together, keyed on
(recording, track) (D-015). Each recording's share of the decoded steps is
installed as its stage 06, and its stages 07–11 are re-run on it. Each
manifest records `hmm_pool` (the member recordings) and
`hmm_pool_fingerprint` (a hash of the members, their stage 05 fingerprints and
the HMM parameters). A pooled recording's stage 06 cannot be re-run alone;
resetting stage 06 takes it out of the pool and back to a fit of its own. The
pool is refitted only when its membership, a member's steps or the HMM
parameters change. A pooled analysis applies one parameter set to all members.

*Upstream ordering hazard.* Upstream script 06 stacks sequences in sorted-key
order and writes the decoded states back in row order. The two agree only if
the rows are already grouped in sorted order — true of every table the
pipeline produces, but not guaranteed by the code. Stage 06 now refuses a
table whose rows are not grouped that way, instead of attaching states to the
wrong steps. The pooled table is sorted by (recording, track) before fitting.

*Per-condition fits* are a diagnostic only: the HMM is fitted separately in
each condition and once pooled, and the fitted mean φ per state and the steps
decoded into each are shown side by side. Nothing is written to a recording.

*Measures, each one value per unit.*
- State occupancy: time in each filtered state over total time, where time is
  the summed duration of RDP steps. A state a unit never visits counts as 0.
- Dwell time: mean duration of uninterrupted stays in a filtered state. A stay
  never spans two tracks. Stays cut by the start or end of a track are
  included, which shortens dwell times for short tracks.
- Steering parameters (ρ, κ, μ, σ) by state: mean over the unit's fitted HMM
  segments, weighted by points fitted.
- Head-cast rate: head casts over minutes tracked. A unit that was tracked
  and has no head cast counts as 0.
- Crawl length: mean length of the unit's crawl steps.

Within a unit, raw quantities are pooled before the value is formed; no value
is a mean of means.

*Units and n.* The unit is a track segment, a larva (default) or a recording.
A larva is identified by recording and larva number; larvae are never matched
across recordings. Every comparison figure labels each condition with its
number of recordings, larvae and track segments, and prints above each box how
many units contribute to it.

*Conditions* are the values of the ticked metadata columns, joined. A
recording with a blank in a grouping column is refused rather than grouped on
its own. Comparing states across recordings that are not all in one pooled fit
over exactly that selection is allowed but flagged, on screen and in the
export. A condition with a single recording is flagged.

*Figures.* State occupancy, dwell time, steering parameters, head-cast rate,
crawl length, and optionally the per-condition HMM check. Conditions are
distinguished by position and label; colour is used only for HMM states. Boxes
are medians and quartiles of the units, drawn when at least three units
contribute. **No statistical test is computed.**

*Export* writes one folder: each figure as PDF, SVG and 300 dpi PNG;
`values.csv` and `counts.csv`, the numbers the figures are drawn from; and
`figure_manifest.json` with the unit, the grouping, the recordings, the HMM
pool, n per condition at all three levels, any warnings and library versions.

**Reasoning.** Making the pooled states each recording's real stage 06 means
everything downstream — HMM segments, steering fits, every per-recording
figure — is expressed in the shared states, and the existing manifest, stale
tracking and resume apply unchanged. Time-weighted occupancy and pooled raw
quantities keep a short track segment from counting as much as a long one.

**Consequences.** Adding or removing a recording changes the states of every
recording in the analysis, and re-runs their stages 07–11 (D-002). The measure
definitions above are choices; any of them can reasonably be made differently
(occupancy by step count, dwell times excluding censored stays) and should be
confirmed before use in a manuscript. With one recording per condition, or few
larvae, the boxes describe that recording, not the genotype. Units within a
recording share a plate and a session, so even "larva" overstates
independence when recordings are few; the recording unit is the conservative
choice. Larva identity relies on the tracker's larva numbering within a
recording, which FIM-Track may reassign after a long loss.

