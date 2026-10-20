# TURN_KINEMATICS.md — stage 12 design proposal

**Status:** ACCEPTED with the answers in §8 (2026-10-20); recorded as D-027
and D-028. Phase T1 (geometry) is built; T2 onwards is not. The stage goes
into `SPEC.md` as revision 4 when it is wired into the pipeline (T3).

> ## ⚠ These data cannot support a genotype comparison
>
> The two recordings available (amiGA-amRNAi, w1118) are **one recording per
> genotype**. Genotype is therefore confounded with plate, session, batch and
> anything else about the day, temperature included. The two also differ in
> crawl speed by 1.4 between-larva SD, at the same body length. amiGA-amRNAi has 8
> larvae; w1118 has 13 larva IDs, several probably the same animal renumbered
> by FIM-Track after a loss. w1118 has no HMM segment that survives stages 08/11
> at all.
>
> **Nothing from these data is to be reported as a genotype difference.** The
> deliverable is the absolute headline number, with its spread, against the
> model's value. The two recordings are two samples for estimating it, not two
> conditions. The condition machinery is built for the ageing work package,
> and the summary **refuses to emit a condition contrast when any condition has
> only one recording** (T3). That makes this caveat impossible to forget later.
>
> The synthetic control (`tests/data/synthetic/SYNTHETIC.md`, D-034) shows
> that between-condition differences in the *magnitude* of this metric are
> biased even with clean data and 93 turns per condition. The bias enters at
> the reduction to five spine points. The *sign* (share of counter-rotating
> turns) survives.

**Question.** For each turn, how much of the net heading change has happened
by the end of the head sweep, before the body straightens? The model says
about 20%.

---

## 0. What the data can and cannot support — read first

These were measured on `amiGA-amRNAi_n1_att2.csv`, larva 1 (9851 frames, fully
tracked). They limit what the analysis can answer.

| Fact | Value | Consequence |
|---|---|---|
| Spine points per frame | **5**: head, spinepoint 1–3, tail | **3 interior boundaries**, not 10 |
| Arc position of the boundaries (median) | sp1 ≈ 0.29, sp2 ≈ 0.54, sp3 ≈ 0.79 of head→tail length | Boundaries are not equally spaced; the tail link is the shortest |
| Coordinate precision | **Whole pixels** (0 of 9851×10 values are fractional) | Each link is 6–8 px long, so 1 px of noise moves a link's direction by 7–10° |
| Spine length | ~28 px ≈ 3.3 mm at 0.117 mm/px | |
| Bend angle at sp1/sp2/sp3, all frames | SD 13.6° / 11.6° / 14.9°; median abs 7.1° / 5.0° / 8.1° | On a straight larva, noise is about the size of a small bend |
| `is_well_oriented == 0` | 3.1% of frames | No upstream stage reads this flag (D-017) |
| Frame rate | 10 fps | A ~1 s head sweep is about 10 frames |
| Recordings available | **1** (one genotype) | The check that conditions really differ cannot be tried on real data yet |

### 0.1 Ten boundaries cannot be made from five points

You asked me to "resample by arc length to a fixed number of points". Doing
that from 5 points to 11 would create 7 boundaries whose bend angles are
interpolated from their neighbours. They would carry no new information, and
a bend peak at "4.1 of 10" would come from the interpolation method, not from
the larva. **I propose not to resample.** All bend positions are reported two
ways:

- `*_boundary` — fractional index on the native 1–3 scale (1 = sp1, the most
  anterior);
- `*_arc_fraction` — position along the spine, 0 = head, 1 = tail, using that
  frame's own link lengths.

**Answered (Q1).** The model body is 11 segments, T1 … A8, as 12 equally
spaced midline points over 4.0 mm (SEGLEN = 0.3636 mm). Model boundary i
(1 … 10) lies between segments i and i+1, at arc fraction i/11 from the head:
boundary 1 = T1/T2, boundary 10 = A7/A8. Every bend position is therefore also
reported as `*_model_boundary` = 11 × arc fraction. **Arc fraction is the
primary measure; model-boundary numbers are indicative.** The model has no
head or tail beyond its segments, but FIM-Track's head and tail points are
contour extremes that include the cephalic region. So the conversion is
approximate, and least reliable at the anterior end.

The resolution is tight. The model's peak (4.1) sits at arc 0.37 and its
trailing edge (5.8) at 0.53. Those fall between sp1 and sp2, and on sp2. A
peak at sp1 and a peak at sp2 can be told apart; 4.1 and 5.0 probably cannot.

### 0.2 The comparison value for bend travel is in different units

The request defines `bend_travel_per_seglen` as boundaries per
segment-length crawled. The model table gives **0.69 boundaries per completed
wave**. These differ by a factor of stride length in segment lengths, roughly
3–5 if a stride is about one-third to one-half of a body length of ~11
segments. **Answered (Q2): report both; per segment-length crawled is
primary.** The per-wave slope uses each larva's own stride length from stage 09
crawls (spine-length minimum to minimum, i.e. peristaltic cycles).
`n_body_segments = 11`. The model measures distance as raw centroid path
length. It is noiseless, so that equals the forward-projected distance used
here (§2.7).

### 0.3 Which heading the model's 20% uses

The headline number depends on which body axis defines heading. That is why
it comes with a sensitivity analysis. **Answered (Q3).** The model's heading
is `atan2(world[3] − world[11])`: tail to the midline point at arc 3/11 ≈ 0.27,
spanning the posterior 73%. sp1, at arc ≈ 0.29, is practically the same
point, so **`tail_sp1` is the model-matched axis**, the only like-for-like
comparison with 0.20. `tail_sp2` stays the primary biological measure.
`turns.csv` carries both (§5.1).

---

## 1. Where it sits

New **stage 12 — Turn kinematics**, folder `12_turn_kinematics/`, module
`core/turns.py`, params `TurnKinematicsParams`, plots `plots/turns.py`, one GUI
tab. It reads:

- `03/smoothed`: per-frame spine points in mm, plus the `interpolated` column
  from stage 02. Only the centre of mass is smoothed there; spine points are
  raw.
- `09/events`: head casts, used as the turn detector (§3).
- `09/signals`: crawl timing, used for stride length (§0.2).
- `04/body_length`: mean spine length per track.

It does not touch stages 00–11, so the parity baseline is unaffected. Stage 12
has no upstream equivalent and is excluded from the baseline, as stage 02 is.
Its own correctness tests are in §7.

Cross-recording outputs (per-condition summary, pooled turn table) are built by
the Phase 5 aggregation machinery, keyed on `(recording_id, track_id)` and with
larva as the animal (D-015, D-025). The stage itself is per recording.

---

## 2. Per-frame geometry

Calculated for every frame of every track. Saved only for frames inside a turn,
plus `frame_padding_s` on each side, as `turn_frames.csv`, so that any turn can
be checked by hand.

1. **Spine smoothing.** Integer-pixel spine points are smoothed over time with
   Savitzky–Golay, per coordinate, per track: window `spine_sg_window` and order
   `spine_sg_order`, both parameters. Stage 09 uses 21 frames (2.1 s), which is
   longer than a head sweep and would smear the sweep peak in time. I propose 7
   frames, order 2, **and** I propose to rerun the headline number at 5, 7 and 11
   so its dependence on smoothing is visible (§6). Bridged frames are smoothed
   through, but turns that contain any are excluded (§4).
2. **Points** p0 = head, p1 = sp1, p2 = sp2, p3 = sp3, p4 = tail.
3. **Bend at boundary i** (i = 1, 2, 3, at p_i): the signed angle from vector
   p[i−1]→p[i] to vector p[i]→p[i+1], computed as
   `atan2(cross, dot)` and wrapped to (−180°, 180°]. **Sign:** image y points
   down, so positive is clockwise **as seen in the image**. The output metadata
   states this, and plots label left and right in image coordinates.
4. **Total bend** = |bend₁| + |bend₂| + |bend₃|.
5. **Head angle** = angle from the posterior axis (tail→sp2) to the anterior axis
   (sp2→head). This is the `body_angle` that stage 09 already uses for its 30°
   head-cast amplitude test, so "head angle" means the same thing in both
   stages.
6. **Heading ψ(axis)**: direction of the chosen axis vector (from rear point to
   front point), **unwrapped over time per track** so that a 200° turn is not
   folded to −160°. Axis options:

   | name | vector | spans (arc) | role |
   |---|---|---|---|
   | `tail_sp3` | tail → sp3 | posterior ~21% | most posterior; one link, so the noisiest |
   | `tail_sp2` | tail → sp2 | posterior ~46% | **primary biological measure.** Equals upstream's `A_posterior` |
   | `tail_sp1` | tail → sp1 | posterior ~71% | **model-matched** (Q3) |
   | `tail_mom` | tail → centre of mass | ~ posterior half, from the contour | an independent estimate from the same region |
   | `tail_head` | tail → head | whole body | **control, not a candidate.** It should give a fraction near 1. If it does not, the pipeline is wrong |

7. **Posterior distance crawled.** The path length of the smoothed tail point
   would build up jitter even in a larva that is standing still. With 1 px ≈
   0.12 mm and a segment length of about 0.3 mm, this bias is large. I propose
   `distance = ∫ max(0, v_tail · ê_axis) dt`: forward tail speed along the
   posterior axis, with backward motion not counted. The stationary jitter floor
   (the same integral over frames that stage 09 classes as neither crawl nor
   head cast) is reported in the diagnostics. Raw path length is kept as a
   sensitivity column (agreed, Q4). The axis is always `tail_sp2`, whichever
   heading axis is being analysed, so distance crawled does not change with
   the heading choice.
8. **Segment length** = body length ÷ `n_body_segments` (Q1). Body length is the
   track's **mean** spine length from stage 04, not a per-frame value. The
   per-frame length changes 10–20% through each peristaltic cycle, which would
   make "one segment length" change in the middle of a turn. Per-animal mean,
   agreed (Q5).
9. **Head/tail flip detection.** A frame is a flip if both
   |head_t − tail_{t−1}| and |tail_t − head_{t−1}| are under `flip_match_mm`,
   **or** the `tail_head` axis turns more than `flip_axis_jump_deg` in one frame.
   Flips are counted and reported, never corrected.

---

## 3. Turn segmentation

The repository has a head-cast detector (stage 09) but no turn detector. A
stage 09 head cast covers the head sweep only: it ends partway down the next
opposite-sign anterior angular-velocity peak, before the body straightens. So
I propose:

- **Turn candidates = stage 09 head casts.** "A turn" then means the same event
  the rest of the pipeline (and stage 11's head-cast rate) counts. Stage 09
  already merges casts closer than 1 s, so a multi-sweep turn is usually one
  candidate.
- **`t_onset`** = the last frame at or before the head-cast start where
  |head angle| < `onset_head_angle_deg`, i.e. where the body was last straight.
  Heading at onset is therefore taken from a straight body. If |head angle|
  never drops below it within `max_onset_lookback_s`, the turn is excluded
  (`onset_not_found`).
- **Sweeps.** Within the turn, a sweep is a run of frames where
  |head angle| ≥ `sweep_head_angle_deg`, all of one sign. A sign change starts
  a new sweep. `n_sweeps` = the number of such runs.
- **`t_sweep_end`** = the frame of maximum |head angle| within the **first**
  sweep.
- **`t_end`** = the first frame after `t_sweep_end` that starts a run of
  `end_hold_frames` frames, all with total bend < `end_total_bend_deg` **and**
  forward tail speed > `end_forward_speed_bl_s` (body lengths per second).
  Search stops at the next turn's onset, the track end, or
  `max_turn_duration_s`; the turn is then censored (`end_not_found`) and
  excluded.
- **`t_max_bend`** = frame of maximum total bend in [t_onset, t_end].

The alternative, your threshold detector without stage 09, is available as
`turn_source="threshold"`. In that case onset is when |head angle| first
crosses `sweep_head_angle_deg`, and every other rule stays the same.
**Answered (Q6): stage 09.** The censoring report starts with the candidate
count before any filter, per track, and that count must equal stage 09's
`n_head_casts` for the track. A test asserts it, so stage 11's head-cast rate
and stage 12's turn set can always be reconciled.

**Thresholds tied to the noise floor.** Fixed degree thresholds mean different
things for different smoothing. The stage measures the noise floor: the SD of
each bend angle and of the head angle during stage 09 crawls. It reports the
floor and warns if `end_total_bend_deg` is below about 2× that floor (since
then the body never counts as straight). Proposed defaults, **to be revised
once the floor has been measured in phase T1**: onset 15°, sweep 20°, end total
bend 30° (summed over 3 boundaries), forward speed 0.1 BL/s, hold 3 frames,
lookback 2 s, maximum turn length 15 s.

---

## 4. Exclusion filters (applied in this order, each counted)

| reason | rule |
|---|---|
| `near_track_edge` | turn window plus `frame_padding_s` runs past the track start or end |
| `interpolated_frames` | any stage 02 bridged frame inside [t_onset, t_end] |
| `spine_flip` | any flip inside [t_onset − pad, t_end + pad] |
| `not_well_oriented` | any frame with `is_well_oriented == 0` inside the turn (switchable; on by default, Q7). Stage 09 counts these casts, so stage 12's turn set is a strict subset of the casts stage 11 counts, for a reason upstream never applies (D-028) |
| `coiled` | any `is_coiled == 1` |
| `onset_not_found` / `end_not_found` | §3 |
| `small_turn` | \|dpsi_total\| < `min_abs_dpsi_total_deg` (15°). The turn stays in the table; only `frac_heading_in_sweep` is set to blank |

**Bias check.** For each filter, compare the dropped turns with the kept ones
on `peak_head_angle_deg`, which can be computed for every candidate, and on
\|dpsi_total\| where it can be computed. Report medians and a Mann–Whitney p
for each. The filter of most concern is `end_not_found`: if large turns are
more often still unresolved when the next turn starts, the sample is biased
toward small, sweep-dominated turns. A filter whose dropped turns have a
**higher** median peak head angle is flagged in the report and in the
summary.

---

## 5. Output tables

All column names are new (no upstream name exists), registered in `schema.py`
and written as written here.

### 5.1 `turns.csv` — one row per candidate turn, primary heading axis

| column | definition |
|---|---|
| `recording_id`, `track_id`, `larva_index` | identifiers. The pooled table adds `larva_id` and every registry metadata column (condition) |
| `turn_index` | 0-based within the track, in time order, counting every candidate |
| `t_onset_s`, `t_sweep_end_s`, `t_max_bend_s`, `t_end_s` | §3 |
| `head_cast_start_s`, `head_cast_end_s` | the stage 09 event it came from (blank for `threshold`) |
| `heading_axis` | name of the primary axis, `tail_sp2` by default |
| `dpsi_sweep_model_axis_deg`, `dpsi_total_model_axis_deg`, `frac_heading_in_sweep_model_axis`, `frac_heading_in_sweep_signed_model_axis` | the same four quantities on the model-matched axis `tail_sp1` (Q3) |
| `dpsi_sweep_deg` | ψ(t_sweep_end) − ψ(t_onset), unwrapped, signed |
| `dpsi_total_deg` | ψ(t_end) − ψ(t_onset), unwrapped, signed |
| `frac_heading_in_sweep` | \|dpsi_sweep\| / \|dpsi_total\|; blank if \|dpsi_total\| < 15° or the turn is excluded |
| `frac_heading_in_sweep_signed` | dpsi_sweep / dpsi_total. **Proposed addition:** the absolute ratio cannot distinguish a sweep that turned the tail *away* from the final heading from one that turned it partway toward it. Negative values are real behaviour, not noise |
| `peak_head_angle_deg` | signed head angle at t_sweep_end |
| `n_sweeps`, `multi_sweep` | §3; `multi_sweep = n_sweeps > 1` |
| `turn_duration_s`, `sweep_duration_s` | t_end − t_onset; t_sweep_end − t_onset |
| `max_total_bend_deg` | total bend at t_max_bend |
| `bend_peak_boundary_at_max` | at t_max_bend: boundary with the largest \|bend\|, refined by fitting a parabola through it and its two neighbours, on the 1–3 scale. At an end boundary there is no neighbour on one side, so the integer is reported |
| `bend_peak_at_edge` | true if the largest bend is at boundary 1 or 3 (true peak may lie beyond it) |
| `bend_peak_arc_fraction_at_max` | the same position as a fraction of head→tail arc length. **Primary** |
| `bend_peak_model_boundary_at_max`, `bend_trail_model_boundary_at_max` | 11 × arc fraction; indicative (§0.1) |
| `bend_trail_boundary_at_max` | posterior-most position where \|bend\| ≥ 0.2 × peak: linear interpolation of the 20% crossing between the last boundary at or above it and the next one below |
| `bend_trail_censored` | true if boundary 3 is still ≥ 20%, so the trailing edge lies behind sp3 and cannot be seen |
| `bend_trail_arc_fraction_at_max` | arc-fraction form |
| `bend_travel_per_seglen` | **Primary.** OLS slope of the bend peak's model-boundary position (11 × arc fraction) against posterior distance crawled in segment lengths (track mean body length ÷ 11), over frames in [t_max_bend, t_end] with total bend ≥ `travel_min_total_bend_deg` |
| `bend_travel_per_wave` | `bend_travel_per_seglen` × the larva's median stride in segment lengths (§0.2) |
| `bend_travel_r2`, `bend_travel_n_frames`, `bend_travel_distance_seglen` | quality of that fit; slope is blank if fewer than `travel_min_frames` frames or less than `travel_min_distance_seglen` crawled |
| `segment_length_mm`, `body_length_mm` | §2.8 |
| `n_interpolated_frames`, `n_not_well_oriented_frames`, `flip_in_window` | quality counts, kept even for included turns |
| `included`, `exclusion_reason` | §4; first failing filter |

On `bend_travel_distance_seglen`: if real larvae pass the bend backward
during a near-pause, with little forward crawling, the slope becomes very large
or undefined. **That would be a result, not a failure.** It would mean the
bend's passage is not tied to forward crawling in the way the model assumes.
The table keeps the distance column so this case is visible rather than
filtered out without notice.

### 5.2 `turn_axis_sensitivity.csv` — one row per (turn, axis, smoothing window)

`recording_id, track_id, turn_index, heading_axis, spine_sg_window,
dpsi_sweep_deg, dpsi_total_deg, frac_heading_in_sweep,
frac_heading_in_sweep_signed, included, exclusion_reason`. Turn timing is
re-detected for each smoothing window, so the inclusion columns can differ
from row to row.

### 5.3 `turn_censoring.csv` — one row per filter

`filter, n_in, n_dropped, n_out, dropped_median_peak_head_angle_deg,
kept_median_peak_head_angle_deg, mannwhitney_p, dropped_median_abs_dpsi_total_deg,
biased_toward_large_turns`.

### 5.4 `turn_quality.csv` — one row per track

Frames, flips, not-well-oriented frames, interpolated frames, candidates,
included turns, and the noise floor per boundary and for the head angle.

### 5.5 Per-condition summary (aggregation, long format)

`condition, <each grouping column>, heading_axis, subset (all | single_sweep |
multi_sweep), measure, n_turns, n_larvae, n_recordings, turn_mean, turn_sd,
turn_se, turn_median, larva_mean, larva_sd, larva_se`. Larva-level values are
the mean of each larva's per-turn values, then mean, SD and SE across larvae.
**Only larva-level SE is valid for comparing conditions** (turns are nested in
larvae). Both are reported, with that label. No statistical test is computed,
as elsewhere (D-025).

### 5.6 Sanity checks that fail loudly

- For each grouping column: distinct values and number of recordings per
  value, printed in the summary and the manifest.
- Two conditions with identical summary rows for any measure → **error**, not
  a warning.
- A difference between conditions smaller than the between-larva SD is marked
  `within_larva_spread` in the summary.
- The `tail_head` control axis with a median fraction below 0.8 → warning
  that heading or sweep timing is probably wrong.

The manifest records every parameter, the axis, the sign convention, the
noise floor and the filter counts.

---

## 6. Proposed parameters (`TurnKinematicsParams`)

None of these has an upstream value; every default is a new choice that needs
your approval.

| field | proposed | note |
|---|---|---|
| `turn_source` | `"stage09_head_casts"` | Q6 |
| `heading_axis` | `"tail_sp2"` | primary biological |
| `model_heading_axis` | `"tail_sp1"` | model-matched (Q3) |
| `sensitivity_axes` | `("tail_sp3","tail_sp2","tail_sp1","tail_mom","tail_head")` | |
| `spine_sg_window`, `spine_sg_order` | 7, 2 | |
| `sensitivity_sg_windows` | (5, 7, 11) | |
| `onset_head_angle_deg` | 15 | revise after T1 noise floor |
| `sweep_head_angle_deg` | 20 | |
| `end_total_bend_deg` | 30 | |
| `end_forward_speed_bl_s` | 0.1 | |
| `end_hold_frames` | 3 | |
| `max_onset_lookback_s` | 2.0 | |
| `max_turn_duration_s` | 15.0 | |
| `min_abs_dpsi_total_deg` | 15.0 | from your brief |
| `trail_fraction_of_peak` | 0.2 | from your brief |
| `n_body_segments` | 11 | Q1; model boundary i at arc i/11 |
| `travel_min_total_bend_deg`, `travel_min_frames`, `travel_min_distance_seglen` | 30, 4, 0.5 | |
| `exclude_not_well_oriented` | True | |
| `flip_match_mm`, `flip_axis_jump_deg` | 0.35 (≈3 px), 120 | |
| `frame_padding_s` | 1.0 | |

---

## 7. Build phases

Each phase ends with tests passing (including `test_regression.py`, unchanged)
and a DECISIONS entry.

**T1 — Geometry only.** `core/turns.py` per-frame quantities (§2), no turns.
Tests on synthetic spines with known answers: a rigid rotation gives zero bend
and exactly the imposed Δψ on every axis; a single known bend placed at
sp1/sp2/sp3 gives the right signed angle and the right peak/trail positions;
a bend moving backward at a set rate per unit tail distance gives that slope; a
synthetic head/tail swap is detected; heading unwrap across ±180°. On real
data: the noise floor table for the validation recording. **Stop here and
review the thresholds against the measured floor.**

**T2 — Turns on one track, checked by hand.** Segmentation, per-turn table,
exclusions and censoring for **one track you choose**. I produce
`turn_frames.csv` for 3–5 of its turns and a text dump: onset, sweep-end,
max-bend and end frames, the raw and smoothed spine coordinates at each, and the
heading at each, so every number in the row can be recomputed by hand from the
CSV. **No plots; stop for your check.**

**T3 — Whole recording and aggregation.** All tracks, axis and smoothing
sensitivity, per-condition summary, the loud sanity checks. Tests: two
synthetic "conditions" built to be identical trigger the error; a synthetic
filter that removes large turns triggers the bias flag; pooling keys on
`(recording_id, track_id)`.

**T4 — Figures, GUI, batch.** Four figures: distribution of
`frac_heading_in_sweep` by condition (larva means overlaid, model 0.20
marked); bend-peak arc position against normalised turn time (onset = 0,
end = 1), median and IQR across turns, with sweep end marked; bend position
against distance crawled for one selected turn with the fitted slope, plus the
distribution of slopes; and the headline number against heading axis and
smoothing window. Also a stage 12 tab, inclusion in batch and in the Aggregate
tab and Excel export.

---

## 8. Questions and answers (2026-10-20)

- **Q1.** 11 segments T1 … A8; boundary i at arc i/11; `n_body_segments = 11`.
  Arc fraction primary, model-boundary numbers indicative (§0.1).
- **Q2.** Both; per segment-length crawled primary.
- **Q3.** Model heading is tail → arc 3/11; matched axis `tail_sp1`, reported
  alongside the primary `tail_sp2`.
- **Q4.** Forward-projected tail distance.
- **Q5.** Per-animal mean body length.
- **Q6.** Stage 09 head casts, with the pre-filter candidate count reported
  and reconciled with stage 09.
- **Q7.** Exclude by default; separate decision entry (D-028).
- **Q8.** Second recording added: `amiRNAi_w1118_n1_att2.csv`.

---

## 9. T1 results — noise floor (2026-10-20)

Both recordings, 21 tracks each, clean frames (no flip, well oriented, not
bridged) inside stage 09 crawls. Values are the median over tracks, in
degrees. `tools/turn_noise_floor.py` reproduces them.

| quantity | window | amiGA-amRNAi SD | median abs | p95 abs | w1118 SD | median abs | p95 abs |
|---|---|---|---|---|---|---|---|
| bend 1 (sp1) | 7 | 12.1 | 7.7 | 25.3 | 12.5 | 8.8 | 25.7 |
| bend 2 (sp2) | 7 | 10.0 | 6.9 | 21.6 | 9.3 | 5.2 | 17.6 |
| bend 3 (sp3) | 7 | 12.9 | 7.4 | 27.3 | 10.9 | 9.2 | 23.5 |
| head angle | 7 | 16.1 | 11.1 | 33.5 | 14.7 | 9.9 | 29.0 |
| total bend | 7 | 15.1 | 24.4 | 55.8 | 13.6 | 25.7 | 51.2 |

Windows 5 and 11 differ from 7 by under 10%: the spread is not pixel jitter
and smoothing does not remove it (D-027).

Tracking: 128 and 115 frames flagged as flips; 4766 (8.9%) and 2396 (6.2%)
frames not well oriented; every flip but one sits next to a not-well-oriented
frame (D-028). Tail path length runs 20–100% above forward distance.

**Proposed revised thresholds, for approval before T2.** These are fixed
values, the same for every track and condition. Thresholds scaled to each
track's own floor would let the definition of a turn differ between genotypes
that crawl with different wiggle.

| parameter | design value | proposed | basis |
|---|---|---|---|
| `onset_head_angle_deg` | 15 | 15 | crawl median abs head angle 10–11°; onset means "back to crawling straightness" |
| `sweep_head_angle_deg` | 20 | **30** | crawl p95 of abs head angle 29–34°; equals stage 09's 30° cast amplitude |
| `end_total_bend_deg` | 30 | **35** | above crawl median total bend (23–27°), with the 3-frame hold carrying the rest |
| `travel_min_total_bend_deg` | 30 | **55** | crawl p95 total bend 51–59°; below it the peak position is mostly noise |

User review (2026-10-20): sweep 30° and travel 55° approved. Sweep 30° is
only tolerable because stage 09 also requires amplitude and duration; do not
lower it. End total bend is to come from the crawl distribution (25th
percentile), not a fixed 35°.

## 10. Pre-T2 diagnostics (2026-10-20)

`tools/diagnose_turn_confounds.py`; findings and the proposal are in D-029.
In short:

- **Angle floor by genotype:** similar. Recording medians differ by under
  1 between-larva SD everywhere except bend SD at sp2 (1.15 SD). Total-bend 25th
  percentile is 17.0° vs 18.3°. **Crawl speed differs by 1.8 SD**
  (0.64 vs 0.89 mm/s). Only one recording per genotype, so this is also
  plate and session.
- **`is_well_oriented`:** 52–55% of head casts contain a flagged frame. The
  flag rises steeply with bend, and also marks real head/tail swaps (tail-first
  motion in 56–58% of flagged moving frames). Excluding on it would remove most
  large turns. Proposed: exclude on geometric swap evidence; flag exclusion
  as a sensitivity variant.

---

## 11. State after T2 revisions (2026-10-20)

Decisions D-032 to D-034. In short:

- **Primary output: heading curves.** For each turn, the cumulative signed
  heading change as a fraction of its own total, on normalised progress,
  distance crawled and time. The signed fraction, i.e. the curve read at the
  sweep end, is the primary scalar; the absolute fraction sits beside it.
  Both are reported as distributions.
- **Head/tail swaps are corrected, not excluded.** Bracketing decides only
  brief flicker; tail-first stretches without a correct neighbour stay
  excluded, so a real reversal is never relabelled.
- **Sweeps use hysteresis** (release at 60% of the running maximum).
- **Bends are signed tail→head**, the same sign as the head angle.
- **The model is compared only through `tools/synthetic_control.py`.**
  Through this code path the default model (slow_bend) gives a signed fraction
  of +0.276 on `tail_sp1` (median; truth +0.239), with 2% of turns negative.
  fast_bend gives −0.115, with 92% negative.

| `tail_sp1`, signed | n turns (larvae) | q25 | median | q75 | share negative |
|---|---|---|---|---|---|
| real amiGA-amRNAi | 70 (8) | −0.00 | +0.096 | +0.19 | 0.26 |
| real w1118 | 83 (10) | −0.10 | +0.127 | +0.26 | 0.34 |
| model slow_bend, through the pipeline | 56 | +0.25 | +0.276 | +0.37 | 0.02 |
| model fast_bend, through the pipeline | 71 | −0.20 | −0.115 | −0.05 | 0.92 |

Caveats: the real turns come through stage 09. The model's are seeded from the
truth file, because stage 09 does not detect the model's large turns: their
anterior angular velocity stays at or below its threshold (D-035). The model's
cast is not slow (45–55°/s rise, as fast in large turns as in small ones).
What takes 2.4–4.2 s is |head angle| reaching its maximum as the bend migrates
to mid-body. So under the current definition the model's "sweep end" is the
bend reaching sp2, not the end of the cast. Which instant the sweep end should
be is open (D-035).
