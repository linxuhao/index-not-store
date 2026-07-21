# J-space extension — the readout gap has a mechanism

Added 2026-07-21. Backs the paper's §"A mechanism for the readout gap: present in the workspace,
lost in emission". Adapts the Jacobian-lens (J-lens) of Anthropic's *Verbalizable Representations
Form a Global Workspace in Language Models* (transformer-circuits.pub/2026/workspace) to the
answer slot, on the frozen base, and reads each fact's answer-direction projection along the
forgetting timeline.

## Scripts
- `jlens_dirs.py` — compute & cache the 48 answer-token J-lens directions on the frozen base,
  plus the two null controls (matched-norm random, frequency-matched mismatch-token). Emits
  `dirs_s<seed>.pt`.
- `probe_jspace.py` — standalone smoke: written-vs-unwritten J-projection AUC by layer, adapter
  on vs off (instrument validation).
- `probe_totrescue.py` — tip-of-the-tongue rescue attempt: gold-token rank, top-k self-candidates,
  Δlp vs full-sequence-lp selectors (shows familiarity authenticates, does not select).
- `plot_timeline.py` — the two-panel figure (cross-mechanism state hierarchy + event-aligned dip).
- Timeline integration lives in the main `phase5/phase5_accum.py` via `--jdirs dirs_s<seed>.pt`
  (run with `--mech ewcreplay` and `--mech naked` to reproduce the shallow/deep rows).

## Results (this dir)
- `accum_ewcreplay_*` / `accum_naked_*` — per-fact J-projection at probe points, two mechanisms,
  3 seeds. State hierarchy @L23: recalled 0.222/0.218, recognized-not-recalled 0.175/0.110 (above
  null), gone 0.137(n=11)/0.090≈null(n=678); nulls ~0.08 / ~0.
- `jspace_smoke_*` — instrument AUC (written vs unwritten): 0.984/1.000/0.993 @L23.
- `totrescue_*` — rank-2 confirmed; Δlp selector 42→31 (harmful), full-seq lp +1 (safe).

Honest scope: answer-slot J-lens is a deliberate narrowing of the original all-context Jacobian;
2B substrate; access-consciousness analogy only, no phenomenal claim.
