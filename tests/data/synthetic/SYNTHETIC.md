# Synthetic recordings with a known answer

A positive control for stage 12. The model is run, its output degraded exactly the way the
camera and tracker degrade a real larva — 5 spine points, 10 fps, whole pixels — and the
turn analysis run on the result is compared with what actually happened.

This answers "can this measurement survive this data?" before any recordings are spent on it.

## Files

```
slow_bend.csv          the degraded recording: this is what the pipeline reads
slow_bend_truth.csv    one row per turn, the answer the pipeline should recover
slow_bend_meta.json    scale, frame rate, arc positions, model parameters
fast_bend.*            the same, with the bend migrating 2.3x faster
synthtrack.js          generator (needs larval_locomotion_sandbox.html beside it)
synthcheck.py          minimal reader: recovers the headline number from the CSV
bendtrack.js           turn tracking in the model, used by the generator
headingaxes.js         the model's headline number on every heading axis
```

Two conditions, 8 larvae each, 900 s. `slow_bend` is `migK` 0.6 (the model default),
`fast_bend` is 1.4. They differ in exactly the mechanism under test, which is the hardest
case for a shape-based metric and therefore the right one.

```bash
node synthtrack.js outdir 900 8 10 5      # outdir, seconds, larvae, fps, spine points
python3 synthcheck.py outdir 7            # dir, Savitzky-Golay window
```

`synthtrack.js` takes a scale as a sixth argument; passing something tiny (`0.001`) removes
pixel quantisation, which is how the losses below were separated.

## Three stages, so the losses can be told apart

| stage | what it is |
|---|---|
| `ideal` | the model's own 12-point geometry in mm, its own heading axis |
| `clean5` | 5 points in mm, un-quantised, tracking-side definitions |
| `pixels` | the same, recovered from the integer-pixel CSV |

`ideal → clean5` is the cost of 5 points. `clean5 → pixels` is the cost of whole pixels.

## What it showed

175 turns from 590 candidates (only 30% clear a 15° minimum; median turn is 4.4°).

**The population mean survives.**

| stage | mean | median | sd |
|---|---|---|---|
| ideal | 0.231 | 0.239 | 0.075 |
| clean5 | 0.191 | 0.203 | 0.087 |
| pixels | 0.206 | 0.231 | 0.123 |

So comparing a measured mean against a model prediction is sound. Net turn magnitude
recovers almost perfectly (r = 0.99).

**The difference between conditions does not.**

| stage | fast_bend | slow_bend | difference |
|---|---|---|---|
| ideal (truth) | 0.242 ± 0.079 | 0.219 ± 0.068 | **+0.023** |
| clean5 | 0.140 ± 0.058 | 0.249 ± 0.077 | −0.109 |
| pixels | 0.131 ± 0.086 | 0.291 ± 0.102 | **−0.160** |

The truth is that these two conditions barely differ on this metric. The measurement
reports a difference seven times larger, **with the opposite sign**.

**It is bias, not noise.** At 14–15 turns per condition the same pattern appears; at 93 and
82 it is unchanged. More recordings make the wrong answer more confident, not less.

**It enters at the 12 → 5 point reduction, not at pixel quantisation.** Re-running with the
scale set so quantisation is negligible reproduces the same split (0.139 / 0.273). Raising
the spine to 7 or 11 points shrinks it but does not remove it (differences of −0.064 and
+0.087 against a truth of ~0.01).

## What to do with that

1. **Use `frac_heading_in_sweep` as an absolute number** against the model's prediction.
   That is what it is for, and it works.
2. **Do not compare conditions on it** without running them through this control first. If
   a condition difference appears in real data, generate a synthetic pair matched to those
   conditions and check whether the measurement manufactures one.
3. Net turn magnitude, turn rate and residence-type measures are not affected — the problem
   is specific to partitioning a turn into sweep and post-sweep, which depends on locating
   the sweep end precisely in a shape that a 5-point spine renders coarsely.

## Caveats

- **The model's turns are stereotyped** (sd 0.075 against a mean of 0.23, and far less
  within a condition). This set therefore tests *bias* well and *variance* poorly — a
  per-turn correlation computed on it is close to meaningless, and the low values it gives
  should not be read as information loss. Real larvae vary much more.
- The synthetic larva is 4.0 mm, spanning ~34 px at 0.117 mm/px. A 3.3 mm larva spans ~28,
  so the real geometry is slightly harsher than this; regenerate with a larger scale to
  bracket it.
- The turn boundaries in the truth file are the model's own state transitions, not a
  detector's output. Running a detector against them measures the detector; the numbers
  above deliberately use the true boundaries so that geometry loss is isolated from
  detection error.
- `is_coiled` is always 0 and `is_well_oriented` always 1: the model has no tracking
  failures, so the filters that handle them cannot be exercised on this data.
