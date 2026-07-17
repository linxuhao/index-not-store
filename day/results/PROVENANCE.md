# Result-file provenance

All files whose names contain a `_peN_` tag (e.g. `accum_naked_n48_pe2_s1234.json`) are the
**canonical runs used in the paper**. `analyze.py` reads only these.

Files **without** a `_peN_` tag (20 files: `accum_{naked,bf,ewc,local,replay}_n{12,48}_s*.json`)
predate the recognition probe and the probe-cadence flag: they carry `recog: null` and an implicit
probe cadence, and were superseded on 2026-07-04 when the recognition meter landed. They are kept
for provenance (per our protocol, superseded runs are archived, not deleted) and are not cited by
the paper or the analysis script. The `n12` files are early pilots at a 12-fact stream length.

`pre_recog_2026-07-04/` holds a second, later provenance layer: the original `_pe2_` baseline runs
backed up on 2026-07-04 immediately before the recognition-instrumented re-runs replaced them at
the same filenames (also uncited; the shipped top-level `_pe2_` files are the canonical ones).
