# Synthetic positive control for stage 12

Supplied by the user on 2026-10-20 (D-034). See `SYNTHETIC.md` for what it is and how
it was made. The generator (`synthtrack.js`) needs `larval_locomotion_sandbox.html`
beside it, which is not included, so the recordings cannot be regenerated from this
folder; they are fixtures.

- `slow_bend.csv.gz`, `fast_bend.csv.gz`: the degraded recordings, gzip-compressed;
  `SHA256SUMS.txt` holds the hashes of the uncompressed files as received.
- `*_truth.csv`, `*_meta.json`, `SYNTHETIC.md`, `synthcheck.py`, `synthtrack.js`: as
  received.

Never edit or regenerate these files. `tests/test_synthetic_control.py` runs stage 12 on
them; `tools/synthetic_control.py tests/data/synthetic <out>` writes the full report.
