#!/usr/bin/env python3
"""Cross-mechanism readout figure (CLAIMS queue 5, revised 2026-07-21 after naked-SGD run).
Panel A: J-projection by memory state, ewcreplay (shallow) vs naked (deep), 3 seeds each, with
the two math-life null controls. Panel B: ewcreplay event-aligned first-miss (dip and recover)."""
import json, glob, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LAY = 23

def load(pat):
    byst = defaultdict(lambda: {"own": [], "mism": [], "rand": []})
    ev = defaultdict(dict)
    for f in sorted(glob.glob(str(HERE / pat))):
        d = json.load(open(f))
        for pt in d["curve"]:
            if "jproj" not in pt:
                continue
            hits, marg = pt["hits"], pt["margins"]; li = pt["jproj"]["layers"].index(LAY)
            for r in pt["jproj"]["rows"]:
                fid = r["fid"]
                s = "recalled" if hits[fid] else ("recog_only" if marg[fid] > 0 else "gone")
                for k in ("own", "mism", "rand"):
                    byst[s][k].append(r[k][li])
                ev[(f, fid)][pt["k"]] = (s, r["own"][li])
    return byst, ev

ewc, ewc_ev = load("results/jspace_timeline/accum_ewcreplay*.json")
naked, _ = load("results/jspace_timeline_naked/accum_naked*.json")

def m(b, s, k="own"):
    return st.mean(b[s][k]) if b[s][k] else float("nan")
def nulls(b):
    return (st.mean(sum([b[s]["mism"] for s in b], [])), st.mean(sum([b[s]["rand"] for s in b], [])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
states = ["recalled", "recog_only", "gone"]
labels = ["recalled", "recognized\nnot recalled", "gone\n(not recognized)"]
x = range(len(states)); w = 0.36
ewc_mism, ewc_rand = nulls(ewc); nk_mism, nk_rand = nulls(naked)
ax1.bar([i - w/2 for i in x], [m(ewc, s) for s in states], w, color="#2a7",
        label="ewcreplay (shallow forget)", zorder=3)
ax1.bar([i + w/2 for i in x], [m(naked, s) for s in states], w, color="#39c",
        label="naked SGD (deep forget)", zorder=3)
for i, s in enumerate(states):
    ax1.annotate(f"n={len(ewc[s]['own'])}", (i - w/2, m(ewc, s)), fontsize=6, ha="center", xytext=(0,2), textcoords="offset points")
    ax1.annotate(f"n={len(naked[s]['own'])}", (i + w/2, m(naked, s)), fontsize=6, ha="center", xytext=(0,2), textcoords="offset points")
ax1.axhline((ewc_mism + nk_mism) / 2, ls="--", c="gray", label=f"mismatch null (~{(ewc_mism+nk_mism)/2:.3f})")
ax1.axhline((ewc_rand + nk_rand) / 2, ls=":", c="k", label="random null (~0)")
ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel(f"J-projection @ layer {LAY}")
ax1.set_title("A. J-projection co-tracks the readout hierarchy\n(recalled > recognized > gone→null), both mechanisms")
ax1.legend(fontsize=7, loc="upper right")

# panel B: ewcreplay event-aligned
curve = {"pre": [], "miss": [], "rec": []}
for tl in ewc_ev.values():
    seen = False
    for k in sorted(tl):
        s, o = tl[k]
        if s == "recalled" and not seen: curve["pre"].append(o)
        elif s != "recalled" and not seen: seen = True; curve["miss"].append(o)
        elif seen and s == "recalled": curve["rec"].append(o); seen = False
vals = [st.mean(curve[c]) for c in ("pre", "miss", "rec")]
ax2.plot([0,1,2], vals, "o-", ms=10, lw=2, color="#e94", zorder=3)
for xx, v in zip([0,1,2], vals):
    ax2.annotate(f"{v:.3f}", (xx, v), textcoords="offset points", xytext=(0,10), ha="center")
ax2.axhline(ewc_mism, ls="--", c="gray", label=f"mismatch null ({ewc_mism:.3f})")
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["last recall\nbefore miss","at first\nmiss","after\nrecovery"])
ax2.set_ylabel(f"J-projection @ layer {LAY}")
ax2.set_title("B. Shallow regime: dip at miss, recover\n(rank displacement, not amplitude loss)")
ax2.legend(fontsize=8); ax2.set_ylim(ewc_mism - 0.02, max(vals) + 0.03)

fig.suptitle("Readout-gap mechanism: workspace amplitude tracks recognition; forgetting is rank displacement",
             fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "results" / "jspace" / "timeline_3curve.png", dpi=150)
print("saved. ewcreplay:", [round(m(ewc,s),3) for s in states],
      "| naked:", [round(m(naked,s),3) for s in states],
      "| nulls ewc", round(ewc_mism,3), "naked", round(nk_mism,3),
      "| event", [round(v,3) for v in vals], "ns", {k:len(v) for k,v in curve.items()})
