#!/usr/bin/env python3
"""fig_ladder.png: (A) per-tier event-aligned survival on naked ws=8 (queue 8);
(B) write-dose grid — acquisition / end-state recognition / end-state key10 (queue 9)."""
import json, glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"

def survival(pattern, tier):
    alive = defaultdict(list)
    for fp in sorted(glob.glob(str(pattern))):
        d = json.load(open(fp))
        for pt in d["curve"]:
            k = pt["k"]
            for fid in range(k):
                t = k - (fid + 1)
                if tier == "recall":
                    alive[t].append(pt["hits"][fid])
                elif tier == "key10" and "granks" in pt:
                    alive[t].append(int(pt["granks"][fid] <= 10))
                elif tier == "afc":
                    alive[t].append(int(pt["margins"][fid] > 0))
    ts = sorted(t for t in alive if len(alive[t]) >= 15)
    return ts, [sum(alive[t]) / len(alive[t]) for t in ts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
pat = R / "halflife" / "accum_naked_n48_pe2_s*.json"
for tier, lab, c in (("recall", "free recall (HL=2)", "#c0392b"),
                     ("key10", "gold in top-10 — key tier (HL=7)", "#e67e22"),
                     ("afc", "2-AFC — auth tier (HL=16)", "#27ae60")):
    ts, ps = survival(pat, tier)
    ax1.plot(ts, ps, marker="o", ms=3, lw=1.8, color=c, label=lab)
ax1.axhline(0.5, ls=":", c="gray", lw=1)
ax1.set_xlabel("fact-writes since this fact's write  (naked, ws=8, 3 seeds pooled)")
ax1.set_ylabel("P(tier alive)")
ax1.set_title("A. Forgetting splits by readout tier")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_ylim(-0.03, 1.03)

WS = [1, 2, 4, 8]
acq, fin, key = [], [], []
for ws, pat2 in ((1, "halflife_ws1/accum_*.json"), (2, "halflife_ws2/accum_*.json"),
                 (4, "halflife_ws4/accum_*.json"), (8, "halflife/accum_naked_*.json")):
    A, F, K = [], [], []
    for fp in sorted(glob.glob(str(R / pat2))):
        d = json.load(open(fp))
        for pt in d["curve"]:
            k = pt["k"]
            for fid in range(k):
                if k - (fid + 1) <= 1:
                    A.append(int(pt["margins"][fid] > 0))
        f0 = d["curve"][-1]
        F += [int(m > 0) for m in f0["margins"]]
        K += [int(g <= 10) for g in f0.get("granks", [])]
    acq.append(sum(A) / len(A)); fin.append(sum(F) / len(F)); key.append(sum(K) / len(K))
x = range(len(WS)); w = 0.27
ax2.bar([i - w for i in x], acq, w, label="acquisition (2-AFC @ t≤1)", color="#5dade2")
ax2.bar(list(x), fin, w, label="end-state recognition", color="#27ae60")
ax2.bar([i + w for i in x], key, w, label="end-state key tier (top-10)", color="#e67e22")
ax2.axhline(fin[-1], ls=":", c="gray", lw=1)
ax2.set_xticks(list(x)); ax2.set_xticklabels([f"ws={w_}" for w_ in WS])
ax2.set_xlabel("write dose (SGD steps per fact), 3 seeds each")
ax2.set_title("B. Dose buys acquisition, not end-state capacity")
ax2.legend(fontsize=8); ax2.set_ylim(0, 1.08)
fig.tight_layout()
out = R / "figs"; out.mkdir(exist_ok=True)
fig.savefig(out / "fig_ladder.png", dpi=150)
print("saved", out / "fig_ladder.png")
