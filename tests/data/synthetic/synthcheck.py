#!/usr/bin/env python3
"""synthcheck.py - how much of the answer survives the measurement chain?

    python3 synthcheck.py <synthetic_dir> [sg_window]

Reads the degraded synthetic recording exactly as the analysis pipeline will -
5 spine points, 10 fps, whole pixels, nothing else - recovers the headline
number, and compares it with the truth file.

Three values per turn:

    ideal    the model's own 12-point geometry in mm at 20 Hz
    clean5   5 points in mm at 10 fps, un-quantised, tracking-side definitions
    pixels   the same, recovered from the integer-pixel CSV with smoothing

ideal -> clean5 is the cost of 5 points and 10 fps.
clean5 -> pixels is the cost of pixel quantisation and smoothing.

This is deliberately a minimal implementation, not the pipeline: it uses the
TRUE turn boundaries from the truth file, so it isolates the geometry loss from
any error in turn detection. The pipeline's own detector will add to it.
"""
import sys, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def heading(sp, k):
    """direction from tail to spine point k"""
    t = sp.shape[1] - 1
    return np.arctan2(sp[:, k, 1] - sp[:, t, 1], sp[:, k, 0] - sp[:, t, 0])


def head_angle(sp):
    """angle from the posterior axis (tail->~54% arc) to the anterior axis"""
    t = sp.shape[1] - 1
    mid = int(round(t * 0.54))
    post = np.arctan2(sp[:, mid, 1] - sp[:, t, 1], sp[:, mid, 0] - sp[:, t, 0])
    ante = np.arctan2(sp[:, 0, 1] - sp[:, mid, 1], sp[:, 0, 0] - sp[:, mid, 0])
    return np.array([wrap(a - p) for a, p in zip(ante, post)])


def load_spines(csv, mm_per_px, sg_window, sg_order=2):
    df = pd.read_csv(csv)
    out = {}
    n_inner = sum(1 for c in df.columns if c.startswith('spinepoint_') and c.endswith('_x'))
    cols = ([('head_x', 'head_y')]
            + [(f'spinepoint_{i}_x', f'spinepoint_{i}_y') for i in range(1, n_inner + 1)]
            + [('tail_x', 'tail_y')])
    for larva, g in df.groupby('larva'):
        g = g.sort_values('frame')
        sp = np.stack([np.stack([g[cx].to_numpy(float), g[cy].to_numpy(float)], axis=1)
                       for cx, cy in cols], axis=1) * mm_per_px
        if sg_window and sg_window > sg_order and sg_window <= len(g):
            w = sg_window if sg_window % 2 else sg_window + 1
            sp = savgol_filter(sp, w, sg_order, axis=0)
        out[larva] = (g['frame'].to_numpy(), sp)
    return out


def frac_for_turn(sp, i0, i1, axis_k):
    """fraction of heading change accumulated by maximum |head angle|"""
    seg = sp[i0:i1 + 1]
    if len(seg) < 3:
        return None, None, None
    ha = np.abs(head_angle(seg))
    isw = 0
    best = -1.0
    for i, h in enumerate(ha):
        if h > best:
            best, isw = h, i
        elif best > math.radians(20) and h < best * 0.6:
            break
    psi = heading(seg, axis_k)
    d_sweep = wrap(psi[isw] - psi[0])
    d_total = wrap(psi[-1] - psi[0])
    if abs(math.degrees(d_total)) < 15:
        return None, math.degrees(d_total), isw
    return abs(d_sweep) / abs(d_total), math.degrees(d_total), isw


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic')
    sg = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    print(f'reading {d}, Savitzky-Golay window {sg}\n')
    rows = []
    for meta_path in sorted(d.glob('*_meta.json')):
        name = meta_path.name.replace('_meta.json', '')
        meta = json.loads(meta_path.read_text())
        spines = load_spines(d / f'{name}.csv', meta['mm_per_px'], sg)
        truth = pd.read_csv(d / f'{name}_truth.csv')

        for _, t in truth.iterrows():
            key = t['larva']
            if key not in spines:
                continue
            frames, sp = spines[key]
            i0 = int(np.searchsorted(frames, t['true_onset_frame']))
            i1 = int(np.searchsorted(frames, t['true_end_frame']))
            if i1 <= i0 or i1 >= len(frames):
                continue
            k_sp1 = max(1, int(round((sp.shape[1] - 1) * 0.29)))
            f_px, dtot_px, isw = frac_for_turn(sp, i0, i1, k_sp1)
            rows.append({
                'condition': t['condition'], 'larva': key,
                'ideal': t.get('ideal_frac', np.nan),
                'clean5': t.get('clean5_frac_tail_sp1', np.nan),
                'pixels': f_px if f_px is not None else np.nan,
                'dtot_ideal': t.get('ideal_dpsi_total_deg', np.nan),
                'dtot_px': dtot_px,
                'sweep_frame_err': (isw + i0 - int(np.searchsorted(frames, t['true_sweep_end_frame'])))
                                   if isw is not None else np.nan,
            })

    r = pd.DataFrame(rows)
    for c in ['ideal', 'clean5', 'pixels', 'dtot_ideal', 'dtot_px', 'sweep_frame_err']:
        r[c] = pd.to_numeric(r[c], errors='coerce')
    print(f'{len(r)} turns\n')

    print('  headline number (fraction of heading change during the sweep), axis tail->sp1')
    print('  stage      n    mean    median     sd')
    for col in ['ideal', 'clean5', 'pixels']:
        v = pd.to_numeric(r[col], errors='coerce').dropna()
        if len(v):
            print(f'  {col:<9} {len(v):>3}   {v.mean():.3f}   {v.median():.3f}   {v.std():.3f}')

    both = r.dropna(subset=['clean5', 'pixels'])
    both = both[pd.to_numeric(both['clean5'], errors='coerce').notna()]
    if len(both):
        c = pd.to_numeric(both['clean5'], errors='coerce')
        p = pd.to_numeric(both['pixels'], errors='coerce')
        keep = c.notna() & p.notna()
        c, p = c[keep], p[keep]
        err = (p - c).abs()
        print(f'\n  per-turn |pixels - clean5|: median {err.median():.3f}, '
              f'90th pct {err.quantile(0.9):.3f}, n {len(c)}')
        if c.std() > 0:
            print(f'  correlation clean5 vs pixels: r = {np.corrcoef(c, p)[0,1]:.3f}')

    print('\n  turn magnitude recovery')
    v = r.dropna(subset=['dtot_ideal', 'dtot_px'])
    if len(v):
        a = pd.to_numeric(v['dtot_ideal'], errors='coerce').abs()
        b = pd.to_numeric(v['dtot_px'], errors='coerce').abs()
        k = a.notna() & b.notna()
        print(f'    |dpsi_total| ideal {a[k].mean():.1f} deg, from pixels {b[k].mean():.1f} deg, '
              f'r = {np.corrcoef(a[k], b[k])[0,1]:.3f}')

    print('\n  sweep-end timing error (frames, pixels vs truth)')
    e = pd.to_numeric(r['sweep_frame_err'], errors='coerce').dropna()
    if len(e):
        print(f'    median {e.median():.1f}, mean abs {e.abs().mean():.2f}, '
              f'within 1 frame: {100*(e.abs() <= 1).mean():.0f}%')

    # The 15 deg filter admits a DIFFERENT set of turns at each stage, so the
    # three rows above are not the same turns. Comparing them as if they were
    # is how a measurement artefact gets read as a condition difference.
    common = r.dropna(subset=['ideal', 'clean5', 'pixels'])
    print(f'\n  common subset defined at all three stages: {len(common)} of {len(r)} turns')
    print('  stage      mean    median     sd')
    for col in ['ideal', 'clean5', 'pixels']:
        v = common[col]
        print(f'  {col:<9} {v.mean():.3f}   {v.median():.3f}   {v.std():.3f}')
    if len(common) > 2:
        # where the information is lost: 5 points and 10 fps, or pixel noise?
        print('  per-turn agreement on the common set (Pearson r):')
        print(f'    ideal  vs clean5  {np.corrcoef(common["ideal"], common["clean5"])[0,1]:+.3f}'
              '   cost of 5 points and 10 fps')
        print(f'    clean5 vs pixels  {np.corrcoef(common["clean5"], common["pixels"])[0,1]:+.3f}'
              '   cost of whole-pixel coordinates')
        print(f'    ideal  vs pixels  {np.corrcoef(common["ideal"], common["pixels"])[0,1]:+.3f}'
              '   end to end')

    print('\n  by condition, COMMON SUBSET ONLY (same turns at every stage)')
    for col in ['ideal', 'clean5', 'pixels']:
        s2 = common.groupby('condition')[col].agg(['count', 'mean', 'std'])
        vals = ', '.join(f'{i} {row["mean"]:.3f}+-{row["std"]:.3f} (n {int(row["count"])})'
                         for i, row in s2.iterrows())
        print(f'    {col:<8} {vals}')

    print('\n  how many turns clear the 15 deg floor at all?')
    for col in ['ideal', 'clean5', 'pixels']:
        print(f'    {col:<8} {r[col].notna().sum():>3} of {len(r)} '
              f'({100*r[col].notna().mean():.0f}%)')
    print(f'    mean |dpsi_total| over ALL turns: {r["dtot_ideal"].abs().mean():.1f} deg '
          f'(median {r["dtot_ideal"].abs().median():.1f})')

    print('\n  by condition (all turns, differing subsets - shown for contrast)')
    for col in ['ideal', 'clean5', 'pixels']:
        g = r.copy()
        g[col] = pd.to_numeric(g[col], errors='coerce')
        s = g.groupby('condition')[col].agg(['count', 'mean', 'std'])
        vals = ', '.join(f'{i} {row["mean"]:.3f}+-{row["std"]:.3f} (n {int(row["count"])})'
                         for i, row in s.iterrows())
        print(f'    {col:<8} {vals}')


if __name__ == '__main__':
    main()
