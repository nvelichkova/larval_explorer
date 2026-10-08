# CLAUDE.md — standing rules for this repository

Read this before touching anything. These are constraints, not suggestions.
If a task appears to require breaking one, stop and ask rather than working
around it.

---

## 1. What this project is

A PyQt5 desktop application that wraps the analysis pipeline from
`yuribilk/Hierarchical-dynamics-of-Drosophila-larvae-exploratory-behavior`
(hierarchical dynamics of *Drosophila* larval exploratory behaviour) so that:

- new FIM-Track recordings can be processed without editing Python,
- every stage parameter is exposed and recorded,
- every stage produces inspectable figures — especially trajectories,
- many recordings can be batch-processed and compared across experimental
  conditions (currently genotype; age later).

The upstream scripts are **scientific reference code tied to a manuscript**.
Reproducing their numerical output exactly is a hard requirement, not a
nice-to-have. See §6.

The full plan lives in `docs/SPEC.md`. Architectural choices and their
reasoning live in `docs/DECISIONS.md`. Read both before starting a phase.

---

## 2. Directory layout

```
larval_explorer/
├── CLAUDE.md                  ← this file
├── docs/
│   ├── SPEC.md                ← the phased plan
│   └── DECISIONS.md           ← decision log (append-only)
├── legacy/
│   ├── upstream_scripts/      ← unmodified upstream .py files, never edited
│   └── upstream_data/         ← the only two data files upstream publishes
│                                (mean_body_length_by_trajectory.csv,
│                                representative_hmm_segments.csv); read-only
├── larval_explorer/
│   ├── core/                  ← pure logic. No Qt. No pyplot. No hardcoded paths.
│   │   ├── schema.py
│   │   ├── params.py
│   │   ├── segments.py        ← stage 02: gap splitting
│   │   ├── stages.py
│   │   ├── pipeline.py
│   │   ├── registry.py
│   │   └── manifest.py
│   ├── plots/                 ← figure factories. No Qt. No savefig.
│   ├── workers/               ← QThread/QRunnable wrappers only
│   └── ui/                    ← PyQt5 widgets
├── tests/
│   ├── test_regression.py     ← upstream-parity tests (see §6)
│   └── data/                  ← small fixtures only
└── requirements.txt
```

---

## 3. Layer rules (the important ones)

**`core/` must be importable and runnable with Qt uninstalled.**
If `import larval_explorer.core.stages` fails without PyQt5, the separation
has leaked. There is a test for this; do not skip it.

**Never `import matplotlib.pyplot` in `core/` or `plots/`.**
pyplot carries global state that is unsafe once a worker thread exists, and
it leaks figures. Construct figures explicitly:

```python
from matplotlib.figure import Figure
fig = Figure(figsize=(6, 4))
ax = fig.add_subplot(111)
```

For on-screen use, hand the `Figure` to `FigureCanvasQTAgg`. For batch export,
hand it to `matplotlib.backends.backend_agg.FigureCanvasAgg`. The same factory
function serves both. `pyplot` is permitted only in `legacy/` (which is never
edited) and in throwaway exploration scripts outside the package.

**`plots/` functions never write files.** They return a `Figure`. The caller
decides whether it goes to a canvas or to disk. No `savefig` inside `plots/`.

**`core/` functions never write files either.** They take dataframes and
params, and return a `StageResult`. Persistence is `pipeline.py`'s job.
This is what makes stages testable and re-runnable without side effects.

**`ui/` contains no analysis logic.** If a widget is computing something
scientific, it belongs in `core/`. The test: could the same number be obtained
from a Spyder console without opening the GUI? It must be able to be.

**`workers/` contains no analysis logic either.** It wraps `core/` calls in
threads, emits progress, handles cancellation. Nothing more.

---

## 4. Column names

The upstream CSVs mix English and Portuguese: `animal ID`, `tempo_inicial (s)`,
`estado`, `estado_filtrado`, `tipo`, `dX`, `dY`, `larva`, `phi`, `theta`.
These names are load-bearing — they appear in the manuscript and in the
distributed data files.

**Policy:** `core/schema.py` canonicalises on load and restores on save.

- Inside the package, use canonical snake_case names only.
- CSVs written to the user's output tree keep the **original upstream names**,
  so downstream compatibility and manuscript consistency are preserved.
- Never hardcode a column name anywhere outside `schema.py`. Reference
  canonical constants.
- Upstream `10_build_event_and_run_steps.py` already contains a column sniffer
  (`detect_columns_traj`). Fold its logic into `schema.py` rather than keeping
  a second detection path.

---

## 5. Parameters

No magic numbers in function bodies. Every tunable is a field on a frozen
dataclass in `core/params.py`, with the upstream value as its default.
`docs/SPEC.md` §7 has the complete inventory with current values — match them
exactly; a changed default silently changes the science.

Every run writes its full resolved parameter set into the run manifest.
A result without its parameters is not reproducible and does not count as
a result.

---

## 6. Upstream parity — the regression test

**The upstream repo does not publish the data its README describes.** Only two
mid-branch CSVs are present and neither has its inputs, so nothing upstream
runs end to end. The baseline is therefore generated from the user's own data
(`docs/DECISIONS.md` D-009).

`tests/test_regression.py` asserts the refactored pipeline reproduces
`tests/data/baseline/` — outputs frozen from running the **unmodified** upstream
scripts once on one real recording, with stage 02 segmentation disabled — using
`pandas.testing.assert_frame_equal`.

Rules:

- This test must pass before any phase is considered complete.
- If a refactor changes numerical output, that is a **bug in the refactor**
  until proven otherwise. Do not update the expected values to match new
  output. Stop and report the discrepancy.
- Run the baseline comparison with `SegmentationParams(enabled=False)`.
  Stage 02 has no upstream equivalent and is excluded by construction; with it
  enabled, output is *expected* to differ (D-008) and the baseline does not
  apply.
- The one sanctioned exception within an imported script is the RDP rewrite
  from recursion to iteration (D-006), which must produce *identical*
  vertices — if it does not, the rewrite is wrong.
- HMM fitting uses `random_state=42`. Keep it. Determinism is what makes this
  test meaningful.
- Do not regenerate the baseline to make a failing test pass. The baseline is
  regenerated only when the user says so — e.g. if the upstream author supplies
  the real `data/` folder, in which case append a superseding decision entry.

Separately, assert that **no RDP step spans a track-segment boundary** when
segmentation is enabled. That is the specific corruption stage 02 exists to
prevent, and it is silent if unchecked.

Upstream scripts in `legacy/upstream_scripts/` are **read-only reference**.
Never edit them, never "fix" them, never reformat them. They are the ground
truth the regression test compares against. The two CSVs in
`legacy/upstream_data/` have the same status: they are the only data upstream
publishes, and they are never edited or regenerated.

---

## 7. Running and testing

Target environment: Windows, Anaconda, Python 3, Spyder. Assume the user may
run `core/` code directly in a Spyder console — keep imports clean and avoid
anything that only works under `__main__`.

```bash
pytest tests/ -v                      # all tests
pytest tests/test_regression.py -v    # upstream parity only
python -m larval_explorer             # launch the GUI
```

Do not add a dependency without saying so explicitly and adding it to
`requirements.txt`. Current stack: numpy, pandas, scipy, matplotlib, seaborn,
scikit-learn, hmmlearn, PyQt5, pytest.

---

## 8. Decision log protocol

`docs/DECISIONS.md` is append-only and numbered. When you make a choice the
spec did not pre-decide — a library, a data structure, an algorithm, a
trade-off — append an entry before finishing the session. Format is in the
file header.

Do not silently reverse an existing decision. If one looks wrong, append a new
entry proposing the reversal, mark the old one `Superseded by D-NNN`, and say
so in your summary.

This log is how context survives between sessions. Treat it as part of the
deliverable, not paperwork.

---

## 9. Things that will go wrong — do not do these

- **Threading + matplotlib.** Compute in the worker, draw on the main thread
  for on-screen figures. For batch, use standalone `Figure` + `FigureCanvasAgg`
  in the worker, never pyplot. Mixing these causes silent corruption and hangs
  that are very hard to trace.
- **Qt widget access from a worker thread.** Signals only. No exceptions.
- **Blocking the GUI thread.** Script 09 emits roughly six PDFs per animal;
  script 11 is heavy. Anything touching a full dataset goes through
  `workers/`.
- **`sys.setrecursionlimit`.** The upstream RDP is recursive and will blow up
  on long tracks. The fix is an iterative rewrite (D-006), not a bigger limit.
- **Reformatting or "tidying" upstream code while porting.** Port behaviour
  exactly, then improve in a separate, clearly-labelled commit.
- **Inventing column names, file names, or state labels.** If the data does
  not tell you, ask.
- **Treating `-1` as a missing-value sentinel.** The verified real export
  encodes missing values as **empty cells**. Do not add `-1` handling without
  evidence from an actual file.
- **Concatenating recordings.** Stage 00 upstream merges several tables into
  one. That is removed (D-010). The only cross-recording operation is the
  pooled HMM fit, which is explicit and recorded in the manifest.
- **Keying the pooled HMM fit on track ID alone.** With segmentation disabled
  every recording's tracks are named `trajectory_0_N` (D-015), so two
  recordings both contain `trajectory_0_101` and would merge silently into one
  sequence. Anything that combines recordings keys on
  `(recording_id, track_id)`, never `track_id` alone. There is a test for
  this; do not skip it.
- **Reordering stored HMM state indices.** The state column in every output
  CSV keeps upstream's indices. Ordering by fitted mean φ is for display and
  aggregation presentation only, read from the manifest (D-016).
- **Confusing the two kinds of "segment".** `02_segments/` holds *track*
  segments (contiguous tracked runs); `08_segments/` holds *HMM* segments
  (sustained state runs). Label them distinctly in code and UI.

---

## 10. Scientific caution

This code produces results that go into a manuscript. When something is
ambiguous — which column, which convention, whether a filter should apply
before or after another — **ask rather than guess**. A plausible-looking wrong
answer is far more costly here than a stalled session.

Flag anything that looks scientifically questionable in the upstream code
rather than silently reproducing or silently fixing it. Report it; let the
user decide.
