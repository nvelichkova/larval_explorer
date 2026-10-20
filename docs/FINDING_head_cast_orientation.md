# Finding: head casts and FIM-Track's `is_well_oriented` flag

**Date:** 2026-10-20 · **Data:** `amiGA-amRNAi_n1_att2.csv`, `amiRNAi_w1118_n1_att2.csv`
(one recording each, default parameters, stage 02 segmentation on) ·
**Decisions:** D-028, D-029, D-030 · **Status:** stages 00–11 unchanged.

## Summary

1. **Head casts are where the tracker's orientation flag fires.** Only 6–9%
   of frames carry `is_well_oriented == 0`, but **52% and 55% of stage 09 head
   casts contain at least one such frame.** No upstream stage reads the flag.
2. **The flag mostly marks posture, not errors.** The chance a frame is flagged
   rises from 1–3% when the larva is straight (|head angle| < 15°) to 72–83%
   when the head angle exceeds 90°.
3. **Excluding flagged casts would wrongly halve the head-cast rate.** Doing
   so cuts the median per-segment rate in stage 11 from 0.053 to 0.019 Hz
   (amiGA-amRNAi), and per-larva casts per minute by 12–100% (median about
   half). Most of the removed casts are real.
4. **The cast *count* is robust to the orientation problem.** With head/tail
   labels un-swapped where they are demonstrably wrong, stage 09's unmodified
   detector finds 205 casts instead of 203 and 210 instead of 211. Of the casts
   that contain a label jump, 59 of 60 and 44 of 50 are still detected.
5. **What is exposed is the kinematics measured inside casts.** **30% and 24%
   of casts contain a head/tail label jump partway through.** Within such a
   cast, stage 09's anterior angle and stage 10/11's head-cast direction and θ
   are computed across labels that changed mid-event. Casts lying entirely on
   swapped labels are rare: 0 of 203 and 1 of 211.

So the stage 11 head-cast **rate** stands. Head-cast **direction and θ**
statistics include a quarter to a third of casts with a mid-cast label jump.
Their effect on those statistics has not been measured.

## Evidence

**Flag versus posture** (fraction of frames flagged):

| \|head angle\| | amiGA-amRNAi | w1118 |
|---|---|---|
| 0–15° | 0.029 | 0.013 |
| 30–45° | 0.177 | 0.125 |
| 60–90° | 0.379 | 0.461 |
| > 90° | 0.719 | 0.826 |

**Real head/tail swaps are found by label continuity, not by the flag.**
Between consecutive frames, the labels have swapped if matching each end to
the *other* end's previous position costs less distance than matching it to
its own. No threshold is involved; fewer than 0.1% of frames are close to a
tie. Each track is then split at these jumps. Each label-constant piece is
oriented by its own net motion: a piece that crawls tail-first has swapped
labels.

| piece duration | swapped (tail-first) | correct (head-first) |
|---|---|---|
| < 5 s | present, 90–92% of frames flagged | present |
| ≥ 5 s | **none** | almost all (4% flagged) |

Swapped spells are short. They total 374 and 319 frames, under 1% of
tracking, against 4766 and 2396 flagged frames.

**Casts by label state** (stage 09 casts):

| | amiGA-amRNAi | w1118 |
|---|---|---|
| on correct labels, not flagged | 90 | 94 |
| on correct labels, flagged (posture) | 36 | 63 |
| contains a label jump | 60 (59 flagged) | 50 (all flagged) |
| entirely on swapped labels | 0 | 1 |
| orientation unresolved | 17 | 3 |
| **total** | 203 | 211 |

## Jumps between two correctly oriented pieces

A third of label jumps (45 of 131 and 35 of 115) first appeared to sit
between two pieces that both move head-first. Most are brief tracker flicker
that reverts. 11 and 12 pieces of 1–3 frames sit between two correct pieces,
and their orientation had been read from motion over too few frames to mean
anything. Stage 12 now decides such pieces by their bracketing jumps and
corrects them (D-033). After correction, 42 jumps per recording remain, all
around stretches whose orientation cannot be resolved; turns there are excluded.

## What was not done

Stages 00–11 are unchanged, and the parity baseline is intact. Correcting
labels in stage 09 would depart from upstream. The evidence here does not
call for it for cast counts. It might for head-cast direction and θ; that
needs its own measurement and decision.

## Reproducing

```
python tools/diagnose_turn_confounds.py <output_root> data/amiGA-amRNAi_n1_att2.csv data/amiRNAi_w1118_n1_att2.csv
python tools/turns_t2.py <output_root> data/amiGA-amRNAi_n1_att2.csv data/amiRNAi_w1118_n1_att2.csv
```

The label-correction rerun of stage 09 is a one-off console script, not in
the repository. It swaps head↔tail and sp1↔sp3 in frames whose label piece is
`swapped` (`turns.prepare_recording`), then calls
`events.detect_events_for_track` on the original and on the corrected table.
