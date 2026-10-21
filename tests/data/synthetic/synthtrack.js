// synthtrack.js — a synthetic recording with a known answer.
//
//   node synthtrack.js [outdir] [duration_s] [larvae_per_condition]
//
// WHY
//
// The turn analysis has to run on 5-point spines at 10 fps quantised to whole
// pixels. Whether the quantity of interest survives that is a question about
// the measurement chain, not about larvae, and it can be answered before any
// real recording is spent on it: run the model, degrade its output exactly the
// way the camera and tracker do, and compare what comes back with what went in.
//
// This writes three things per condition:
//
//   <name>.csv            the degraded "recording" - 5 spine points, 10 fps,
//                         integer pixels. This is what the pipeline reads.
//   <name>_truth.csv      one row per turn, with the answer computed two ways
//                         (below). This is what the pipeline should recover.
//   <name>_meta.json      scale, frame rate, arc positions, model parameters,
//                         model fingerprint.
//
// TWO TRUTHS, because the losses are worth separating:
//
//   ideal_*   from the full 12-point spine in mm at 20 Hz, using the model's
//             own definitions. What the model actually did.
//   clean5_*  from the 5-point spine in mm at 10 fps, un-quantised, using the
//             TRACKING analysis's definitions (head angle from tail->sp2 and
//             sp2->head). What a perfect tracker with 5 points would report.
//
// So: ideal vs clean5 is the cost of 5 points and 10 fps. clean5 vs whatever
// the pipeline reports from the CSV is the cost of pixel quantisation and
// whatever smoothing it applies. If the second gap is large, the recording
// cannot answer the question and no amount of analysis will fix it.
//
// Two conditions are emitted with genuinely different turn mechanics, so the
// "do these conditions actually differ" checks can be exercised on data where
// the difference is known to be real - which a single real recording cannot do.

const fs = require('fs');
const path = require('path');
const { trackTurns, analyse } = require('./bendtrack');

let MM_PER_PX = 0.117;            // from the validation recording, overridable
let FPS = 10;                     // ditto, overridable
let ARC = [0, 0.29, 0.54, 0.79, 1.0];   // measured median spinepoint positions
/** n equally spaced points, for asking what a better tracker would buy */
function arcFor(n) {
  if (n === 5) return [0, 0.29, 0.54, 0.79, 1.0];
  return Array.from({ length: n }, (_, i) => i / (n - 1));
}
const SEG = 11;                   // model segments, T1..A8
const SEGLEN = 4.0 / SEG;

const wrap = (a) => { while (a > Math.PI) a -= 2*Math.PI; while (a < -Math.PI) a += 2*Math.PI; return a; };
const D = 180 / Math.PI;
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);

/* ---- geometry ----------------------------------------------------------- */

/** resample a polyline onto given arc fractions */
function resample(pts, fracs) {
  const seg = [], cum = [0];
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
    seg.push(d); cum.push(cum[i-1] + d);
  }
  const L = cum[cum.length-1];
  return fracs.map((f) => {
    const target = f * L;
    if (target <= 0) return pts[0].slice();
    if (target >= L) return pts[pts.length-1].slice();
    let i = 1; while (i < cum.length && cum[i] < target) i++;
    const t = (target - cum[i-1]) / (cum[i] - cum[i-1] || 1);
    return [pts[i-1][0] + t*(pts[i][0]-pts[i-1][0]),
            pts[i-1][1] + t*(pts[i][1]-pts[i-1][1])];
  });
}

const ang = (a, b, c) => {          // signed turn from a->b to b->c
  const ux=b[0]-a[0], uy=b[1]-a[1], vx=c[0]-b[0], vy=c[1]-b[1];
  return Math.atan2(ux*vy - uy*vx, ux*vx + uy*vy);
};
const dirOf = (from, to) => Math.atan2(to[1]-from[1], to[0]-from[0]);

/* ---- truth per turn ------------------------------------------------------ */

/** the tracking analysis's head angle on a 5-point spine */
function headAngle5(s) {            // s = [head, ...interior..., tail]
  const mid = Math.round((s.length - 1) * 0.54);   // the ~54% arc point
  return wrap(dirOf(s[mid], s[0]) - dirOf(s[s.length-1], s[mid]));
}

/* Forward-projected tail distance, the tracking side's definition: the tail's
   motion along the posterior axis, with backward motion not counted, so that
   jitter on a stationary animal does not accumulate. */
function tailForward(frames, i0, i1) {
  let d = 0;
  for (let i = i0 + 1; i <= i1; i++) {
    const a = frames[i-1].spine5, b = frames[i].spine5;
    const T = a.length - 1;
    const ax = a[2][0]-a[T][0], ay = a[2][1]-a[T][1];      /* tail -> sp2 */
    const L = Math.hypot(ax, ay) || 1;
    const vx = b[T][0]-a[T][0], vy = b[T][1]-a[T][1];
    const fwd = (vx*ax + vy*ay) / L;
    if (fwd > 0) d += fwd;
  }
  return d;
}

function turnTruth(frames, fps, segRef) {
  // frames: [{t, spine12, spine5, cx, cy}] through one turn, in order
  if (frames.length < 5) return null;

  // --- clean5: the tracking analysis's own definitions on perfect 5 points
  let iSweep = 0, best = -1;
  for (let i = 0; i < frames.length; i++) {
    const h = Math.abs(headAngle5(frames[i].spine5));
    if (h > best) { best = h; iSweep = i; }
    else if (best > 20/D && h < best * 0.6) break;     // first sweep only
  }
  const T = frames[0].spine5.length - 1;
  const at = (f) => Math.max(0, Math.min(T, Math.round(T * f)));
  const axes5 = { tail_sp1: at(0.29), tail_sp2: at(0.54), tail_sp3: at(0.79), tail_head: 0 };
  const out = { n_frames: frames.length,
                t_onset_s: +frames[0].t.toFixed(3),
                t_sweep_end_s: +frames[iSweep].t.toFixed(3),
                t_end_s: +frames[frames.length-1].t.toFixed(3),
                peak_head_angle_deg: +(best*D).toFixed(2) };
  for (const [name, k] of Object.entries(axes5)) {
    const p0 = dirOf(frames[0].spine5[T], frames[0].spine5[k]);
    const pS = dirOf(frames[iSweep].spine5[T], frames[iSweep].spine5[k]);
    const pE = dirOf(frames[frames.length-1].spine5[T], frames[frames.length-1].spine5[k]);
    const dS = wrap(pS-p0), dT = wrap(pE-p0);
    out['clean5_dpsi_sweep_deg_'+name] = +(dS*D).toFixed(2);
    out['clean5_dpsi_total_deg_'+name] = +(dT*D).toFixed(2);
    out['clean5_frac_'+name] = Math.abs(dT*D) >= 15
      ? +(Math.abs(dS)/Math.abs(dT)).toFixed(4) : '';
  }

  /* --- the cast end: peak of the most anterior bend (sp1).
     The head-angle maximum tracks the bend arriving at the sp2 pivot, not the
     end of the cast, so it merges the sweep with the migration whenever
     migration is slow. The anterior bend is local to one articulation and
     cannot inherit that, and it sits at arc 0.29, which maps to T3/A1 - where
     the asymmetric output is initiated. */
  let iCast = 0, bestC = -1;
  for (let i = 0; i < frames.length; i++) {
    const sp = frames[i].spine5;
    const a1 = Math.abs(ang(sp[0], sp[1], sp[2]));
    if (a1 > bestC) { bestC = a1; iCast = i; }
  }
  out.t_cast_end_s = +frames[iCast].t.toFixed(3);
  out.cast_end_frame_offset = iCast;
  out.peak_sp1_bend_deg = +(bestC * D).toFixed(2);
  for (const [name, k] of Object.entries(axes5)) {
    const p0 = dirOf(frames[0].spine5[T], frames[0].spine5[k]);
    const pC = dirOf(frames[iCast].spine5[T], frames[iCast].spine5[k]);
    const pE = dirOf(frames[frames.length-1].spine5[T], frames[frames.length-1].spine5[k]);
    const dC = wrap(pC-p0), dT = wrap(pE-p0);
    out['clean5_cast_dpsi_deg_'+name] = +(dC*D).toFixed(2);
    out['clean5_cast_frac_'+name] = Math.abs(dT*D) >= 15
      ? +(dC/dT).toFixed(4) : '';                /* SIGNED: the primary form */
    out['clean5_cast_absfrac_'+name] = Math.abs(dT*D) >= 15
      ? +(Math.abs(dC)/Math.abs(dT)).toFixed(4) : '';
  }
  /* sp1 -> sp2 lag, the migration observable, per turn */
  let iMid = 0, bestM = -1;
  for (let i = 0; i < frames.length; i++) {
    const sp = frames[i].spine5;
    const a2 = Math.abs(ang(sp[1], sp[2], sp[3]));
    if (a2 > bestM) { bestM = a2; iMid = i; }
  }
  out.t_sp2_peak_s = +frames[iMid].t.toFixed(3);
  out.lag_sp1_sp2_s = +(frames[iMid].t - frames[iCast].t).toFixed(3);
  let dMig = 0;
  for (let i = Math.min(iCast, iMid) + 1; i <= Math.max(iCast, iMid); i++)
    dMig += Math.hypot(frames[i].cx-frames[i-1].cx, frames[i].cy-frames[i-1].cy);
  out.crawled_during_lag_seglen = +(dMig/SEGLEN).toFixed(4);
  /* the matched definition: forward-projected tail distance over a segment
     length taken from the 5-point polyline measured while crawling straight.
     A 5-chord polyline is a constant ~84% of the true arc, so dividing by the
     model's true 4.0/11 instead inflates nothing here but would make the two
     sides disagree by ~19%. */
  const dFwd = tailForward(frames, Math.min(iCast, iMid), Math.max(iCast, iMid));
  out.crawled_during_lag_mm_tailfwd = +dFwd.toFixed(4);
  out.segment_length_ref_mm = +segRef.toFixed(5);
  out.crawled_during_lag_seglen_matched = +(dFwd/segRef).toFixed(4);

  // --- ideal: the model's own 12-point geometry and axis
  const psi12 = (s) => dirOf(s[11], s[3]);
  const p0 = psi12(frames[0].spine12);
  const pS = psi12(frames[iSweep].spine12);
  const pE = psi12(frames[frames.length-1].spine12);
  const dS = wrap(pS-p0), dT = wrap(pE-p0);
  out.ideal_dpsi_sweep_deg = +(dS*D).toFixed(2);
  out.ideal_dpsi_total_deg = +(dT*D).toFixed(2);
  out.ideal_frac = Math.abs(dT*D) >= 15 ? +(Math.abs(dS)/Math.abs(dT)).toFixed(4) : '';
  const pC12 = psi12(frames[iCast].spine12);
  const dC12 = wrap(pC12-p0);
  out.ideal_cast_dpsi_deg = +(dC12*D).toFixed(2);
  out.ideal_cast_frac = Math.abs(dT*D) >= 15 ? +(dC12/dT).toFixed(4) : '';
  out.ideal_cast_absfrac = Math.abs(dT*D) >= 15 ? +(Math.abs(dC12)/Math.abs(dT)).toFixed(4) : '';

  // bend profile on the full spine, at maximum total bend
  let iMax = 0, bestB = -1;
  const prof = frames.map((f) => {
    const b = [];
    for (let k = 1; k <= 10; k++) b.push(ang(f.spine12[k-1], f.spine12[k], f.spine12[k+1]));
    return b;
  });
  prof.forEach((b, i) => { const m = b.reduce((x,y)=>x+Math.abs(y),0); if (m > bestB) { bestB = m; iMax = i; } });
  const bm = prof[iMax].map(Math.abs);
  let pk = 0; for (let i = 1; i < bm.length; i++) if (bm[i] > bm[pk]) pk = i;
  let trail = pk; for (let i = 0; i < bm.length; i++) if (bm[i] >= 0.2*bm[pk]) trail = i;
  out.ideal_max_total_bend_deg = +(bestB*D).toFixed(2);
  out.ideal_bend_peak_boundary = pk + 1;
  out.ideal_bend_peak_arc = +((pk+1)/SEG).toFixed(4);
  out.ideal_bend_trail_boundary = trail + 1;
  out.ideal_bend_trail_arc = +((trail+1)/SEG).toFixed(4);

  // travel: boundary peak against centroid distance, over [t_max_bend, t_end]
  const xs = [], ys = [];
  let d = 0;
  for (let i = iMax + 1; i < frames.length; i++) {
    d += Math.hypot(frames[i].cx-frames[i-1].cx, frames[i].cy-frames[i-1].cy);
    const b = prof[i].map(Math.abs);
    let p = 0; for (let j = 1; j < b.length; j++) if (b[j] > b[p]) p = j;
    xs.push(d/SEGLEN); ys.push(p+1);
  }
  if (xs.length >= 4 && xs[xs.length-1] >= 0.5) {
    const mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0;
    for (let i = 0; i < xs.length; i++) { sxy += (xs[i]-mx)*(ys[i]-my); sxx += (xs[i]-mx)**2; }
    out.ideal_travel_per_seglen = sxx > 1e-9 ? +(sxy/sxx).toFixed(4) : '';
    out.ideal_travel_distance_seglen = +xs[xs.length-1].toFixed(3);
  } else { out.ideal_travel_per_seglen = ''; out.ideal_travel_distance_seglen = ''; }
  return out;
}

/* ---- CSV ---------------------------------------------------------------- */
function toCsv(rows) {
  if (!rows.length) return '';
  const cols = [];
  rows.forEach((r) => Object.keys(r).forEach((k) => { if (!cols.includes(k)) cols.push(k); }));
  return cols.join(',') + '\n' + rows.map((r) =>
    cols.map((c) => (r[c] === undefined || r[c] === null) ? '' : r[c]).join(',')).join('\n') + '\n';
}

/* ---- one condition ------------------------------------------------------- */
function emit(name, params, larvae, duration, outDir) {
  const frameRows = [], truthRows = [];
  let fingerprint = '';
  for (let L = 1; L <= larvae; L++) {
    const r = trackTurns({ duration, seed: L, params });
    if (!r.ok) { console.error('  ' + name + ' larva ' + L + ' failed: ' + r.error); continue; }

    // 20 Hz samples -> 10 fps
    const stride = Math.max(1, Math.round(20 / FPS));
    const keep = r.samples.filter((s, i) => s.spine && i % stride === 0);
    const built = keep.map((s) => ({
      t: s.t, spine12: s.spine, spine5: resample(s.spine, ARC), cx: s.cx, cy: s.cy,
      state: s.state }));

    built.forEach((f, i) => {
      const px = f.spine5.map((p) => [Math.round(p[0]/MM_PER_PX), Math.round(p[1]/MM_PER_PX)]);
      const len = f.spine5.reduce((a, _, k) => k ? a + Math.hypot(
        f.spine5[k][0]-f.spine5[k-1][0], f.spine5[k][1]-f.spine5[k-1][1]) : 0, 0);
      const row = { larva: name + '_L' + L, frame: i, time_s: +(i/FPS).toFixed(3),
        head_x: px[0][0], head_y: px[0][1] };
      for (let q = 1; q < px.length - 1; q++) {
        row['spinepoint_' + q + '_x'] = px[q][0];
        row['spinepoint_' + q + '_y'] = px[q][1];
      }
      row.tail_x = px[px.length-1][0]; row.tail_y = px[px.length-1][1];
      row.mom_x = Math.round(f.cx/MM_PER_PX); row.mom_y = Math.round(f.cy/MM_PER_PX);
      row.spine_length = +(len/MM_PER_PX).toFixed(2);
      row.is_coiled = 0; row.is_well_oriented = 1; row.go_phase = f.state === 'run' ? 1 : 0;
      frameRows.push(row);
    });

    /* segment-length reference: the 5-point polyline length while NOT turning,
       which is what a tracker measures on a straight animal */
    const turnSpans = analyse(r.samples).map((t) => [t.frames[0].t, t.frames[t.frames.length-1].t]);
    const inTurn = (tt) => turnSpans.some((sp) => tt >= sp[0] - 1 && tt <= sp[1] + 1);
    const plen = [];
    built.forEach((f) => {
      if (inTurn(f.t)) return;
      let L = 0;
      for (let k = 1; k < f.spine5.length; k++)
        L += Math.hypot(f.spine5[k][0]-f.spine5[k-1][0], f.spine5[k][1]-f.spine5[k-1][1]);
      plen.push(L);
    });
    plen.sort((x, y) => x - y);
    const segRef = (plen.length ? plen[Math.floor(plen.length/2)] : 4.0) / SEG;

    // ground truth from the model's own turn boundaries, not from a detector
    const byT = new Map(built.map((f) => [f.t.toFixed(2), f]));
    analyse(r.samples).forEach((turn, ti) => {
      const fr = turn.frames.map((x) => byT.get(x.t.toFixed(2))).filter(Boolean);
      const tr = turnTruth(fr, FPS, segRef);
      if (!tr) return;
      truthRows.push(Object.assign(
        { condition: name, larva: name + '_L' + L, turn_index: ti,
          true_onset_frame: Math.round(fr[0].t*FPS),
          true_sweep_end_frame: Math.round((+tr.t_sweep_end_s)*FPS),
          true_end_frame: Math.round(fr[fr.length-1].t*FPS) }, tr));
    });
  }
  fs.writeFileSync(path.join(outDir, name + '.csv'), toCsv(frameRows));
  fs.writeFileSync(path.join(outDir, name + '_truth.csv'), toCsv(truthRows));
  fs.writeFileSync(path.join(outDir, name + '_meta.json'), JSON.stringify({
    source: 'larval locomotion sandbox, synthetic', condition: name,
    model_params: params, larvae, duration_s: duration,
    mm_per_px: MM_PER_PX, fps: FPS, spinepoint_arc_fractions: ARC,
    model_segments: SEG, model_segment_length_mm: +SEGLEN.toFixed(4),
    model_body_mm: 4.0, model_heading_axis: 'tail -> midline point 3 (arc 0.27)',
    note: 'coordinates are whole pixels; truth file carries the un-degraded answer'
  }, null, 2));
  return { frames: frameRows.length, turns: truthRows.length };
}

function main() {
  const outDir = path.resolve(process.argv[2] || 'synthetic');
  const duration = +(process.argv[3] || 600);
  const larvae = +(process.argv[4] || 4);
  /* The model's replay history is stored at 20 Hz, so that is the ceiling.
     Asking for more silently mislabels frame times and breaks alignment with
     the truth file, which is a bug I shipped once already. */
  const want = +(process.argv[5] || 10);
  const stride = Math.max(1, Math.round(20 / want));
  FPS = 20 / stride;
  if (Math.abs(FPS - want) > 1e-9)
    console.log(`  requested ${want} fps; history is 20 Hz, using ${FPS} fps`);
  ARC = arcFor(+(process.argv[6] || 5));
  if (process.argv[7]) MM_PER_PX = +process.argv[7];
  fs.mkdirSync(outDir, { recursive: true });
  console.log(`  ${ARC.length} spine points, ${FPS} fps\n`);

  // two conditions whose turn mechanics genuinely differ: slow bend migration
  // with large turns, against fast migration with small ones
  const CONDS = {
    slow_bend: { migK: 0.6 },
    fast_bend: { migK: 1.4 }
  };
  console.log('synthetic recordings -> ' + outDir + '\n');
  for (const [name, params] of Object.entries(CONDS)) {
    const r = emit(name, params, larvae, duration, outDir);
    console.log(`  ${name.padEnd(10)} ${r.frames} frames, ${r.turns} turns ` +
      `(migK ${params.migK})`);
  }
  console.log('\n  each condition: <name>.csv, <name>_truth.csv, <name>_meta.json');
  console.log('  the pipeline reads the first; the second is what it should recover.');
}

if (require.main === module) main();
module.exports = { emit, resample, turnTruth };
