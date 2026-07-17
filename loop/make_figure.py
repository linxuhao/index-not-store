"""Regenerate the paper's horizon figure (paper/fig_horizon.pdf) from results/loop_*.json."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "fig_horizon.pdf")
STYLES = {"full": ("tab:green", "full (ceiling)"), "misslog": ("tab:blue", "error-gated (misslog)"),
          "markov": ("tab:orange", "recency-only (markov)")}

def final_entry(d):
    return [e for e in d["trajectory"] if "post_hits" in e][-1]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
for night, (c, lbl) in STYLES.items():
    curves = [final_entry(json.load(open(f)))["recall_by_day"]
              for f in sorted(glob.glob(f"{R}/loop_{night}_D30_f24_s*.json"))]
    if not curves:
        continue
    n = len(curves[0])
    for c2 in curves:
        ax.plot(range(n), c2, color=c, alpha=0.25, lw=1)
    ax.plot(range(n), [sum(c2[i] for c2 in curves) / len(curves) for i in range(n)], color=c, lw=2, label=lbl)
ax.set_xlabel("day the facts were written"); ax.set_ylabel("recalled /24 (after night 30)")
ax.set_title("(a) Retention by age, day 30 (720 facts)"); ax.set_ylim(-0.5, 25)
ax.legend(fontsize=8, loc="center left")

ax = axes[1]
for night, (c, lbl) in STYLES.items():
    xs, ys = [], []
    for D, N in [(6, 144), (12, 288), (30, 720)]:
        vals = [100 * sum(final_entry(json.load(open(f)))["post_hits"]) / N
                for f in sorted(glob.glob(f"{R}/loop_{night}_D{D}_f24_s*.json")) if "oncap" not in f]
        if vals:
            xs.append(N); ys.append(sum(vals) / len(vals))
            ax.scatter([N] * len(vals), vals, color=c, s=12, alpha=0.5)
    ax.plot(xs, ys, "-o", color=c, lw=2, label=lbl)
ax.set_xlabel("total facts written (horizon)"); ax.set_ylabel("% of history recalled")
ax.set_title("(b) Fixed nightly budget vs. growing history"); ax.set_ylim(0, 102)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT)
print("wrote", OUT)
