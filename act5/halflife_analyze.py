#!/usr/bin/env python3
"""Readout half-life ladder, FREE tiers (CLAIMS queue 8; no GPU).

From existing queue-5 pe2 timelines: event-aligned survival per tier —
  T1 recall (hits), T3 2-AFC (margins>0), T4 J-alive (own_L23 > per-probe rand null).
t = probe_step - write_step (units: fact-writes; 1 unit = ws grad steps [+replay]).
HL = first t with pooled P(alive) < 0.5, right-censored at stream end.
naked = primary (no maintenance); ewcreplay = descriptive (replay-maintained).
"""
import json, sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).parent / "results"
SETS = {
    "naked": [R / "jspace_timeline_naked" / f"accum_naked_n48_pe2_s{s}.json" for s in (1234, 2025, 777)],
    "ewcreplay": [R / "jspace_timeline" / f"accum_ewcreplay_l300_Pmiss_n48_pe2_s{s}.json" for s in (1234, 2025, 777)],
    # --rank-probe re-runs (queue 8 T2 key window; also T1/T3 consistency check vs above)
    "naked-rank": [R / "halflife" / f"accum_naked_n48_pe2_s{s}.json" for s in (1234, 2025, 777)],
    "ewcreplay-rank": [R / "halflife" / f"accum_ewcreplay_l300_Pmiss_n48_pe2_s{s}.json" for s in (1234, 2025, 777)],
}

L23 = 3  # index of layer 23 in jproj layers [12,16,20,23]

for mech, files in SETS.items():
    alive = {t: defaultdict(list) for t in ("recall", "key10", "afc", "j")}
    for fp in files:
        if not fp.exists():
            print(f"[skip] {fp}")
            continue
        d = json.loads(fp.read_text())
        for pt in d["curve"]:
            k = pt["k"]
            jrows = {r["fid"]: r for r in pt.get("jproj", {}).get("rows", [])}
            rnull = [r.get("rand", [None]*4)[L23] for r in jrows.values() if r.get("rand")]
            rmean = sum(rnull) / len(rnull) if rnull else None
            for fid in range(k):
                t = k - (fid + 1)
                alive["recall"][t].append(pt["hits"][fid])
                alive["afc"][t].append(int(pt["margins"][fid] > 0))
                if "granks" in pt:
                    alive["key10"][t].append(int(pt["granks"][fid] <= 10))
                if rmean is not None and fid in jrows:
                    alive["j"][t].append(int(jrows[fid]["own"][L23] > rmean))
    print(f"\n=== {mech} ===")
    for tier in ("recall", "key10", "afc", "j"):
        pts = sorted(alive[tier])
        if not pts:
            print(f"  {tier}: no data")
            continue
        hl = None
        curve = []
        for t in pts:
            v = alive[tier][t]
            p = sum(v) / len(v)
            curve.append((t, round(p, 3), len(v)))
            if hl is None and p < 0.5 and len(v) >= 30:
                hl = t
        show = [c for c in curve if c[0] in (0, 1, 3, 7, 15, 23, 31, 39, 47)]
        print(f"  {tier:6s} HL(t50)={hl}  curve(t,P,n): {show}")
