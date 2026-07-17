"""Re-derive every table and figure of the paper from results/loop_*.json.

Definitions: post-night recall = core-only greedy cloze over the full history after each night;
recognition = 2-AFC logprob margin > 0 vs a same-stream distractor (chance = half);
R/L/G survival = next-probe recall of facts that were recalled / latent (recognized-only) / gone
at the previous full probe (consecutive full-probe days only)."""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def runs(night, D, oncap=False):
    pat = f"loop_{night}_oncap_D{D}_f24_s*.json" if oncap else f"loop_{night}_D{D}_f24_s*.json"
    fs = sorted(glob.glob(os.path.join(R, pat)))
    fs = [f for f in fs if ("oncap" in f) == oncap]
    return [json.load(open(f)) for f in fs]

if __name__ == "__main__":
    for D, N in [(6, 144), (12, 288), (30, 720)]:
        print(f"=== D{D} ({N} facts) ===")
        for night in ["full", "misslog", "markov", "hybrid", "recite"]:
            rs = runs(night, D)
            if not rs: continue
            fins, recogs, fws = [], [], set()
            surv = {"recalled": [0, 0], "latent": [0, 0], "gone": [0, 0]}
            for d in rs:
                t = [e for e in d["trajectory"] if "post_hits" in e]
                fins.append(sum(t[-1]["post_hits"])); recogs.append(t[-1].get("n_recognized"))
                fw = d.get("firewall", {})
                fws.add((fw.get("gsm8k_base"), fw.get("gsm8k_off")))
                for t0, t1 in zip(t, t[1:]):
                    if t1["day"] - t0["day"] != 1: continue
                    for i in range(len(t0["post_hits"])):
                        cat = ("recalled" if t0["post_hits"][i] else
                               "latent" if t0["margins"][i] > 0 else "gone")
                        surv[cat][0] += t1["post_hits"][i]; surv[cat][1] += 1
                if "oncap" in t[-1]:
                    o = t[-1]["oncap"]
                    print(f"    [oncap final] core+day GSM8K {o.get('gsm8k_core_day')} NLL {o.get('ppl_core_day')}"
                          f" | core GSM8K {o.get('gsm8k_core')} NLL {o.get('ppl_core')}")
            sv = {k: (f"{100*a/b:.0f}% (n={b})" if b else "-") for k, (a, b) in surv.items()}
            print(f"  {night:8s} final={fins}/{N} mean={sum(fins)/len(fins):6.1f} ({100*sum(fins)/len(fins)/N:.0f}%)"
                  f" recog={recogs} survivalRLG={sv} fw={sorted(fws)}")
        for night in ["misslog"]:
            rs = runs(night, D, oncap=True)
            if not rs: continue
            print(f"  [dual-state capability re-runs, {night}, n={len(rs)}] (identical training; --oncap probes; separate files)")
            for d in rs:
                t = [e for e in d["trajectory"] if "post_hits" in e]
                o = t[-1].get("oncap", {})
                print(f"    s{d['seed']}: final={sum(t[-1]['post_hits'])} | core+day GSM8K {o.get('gsm8k_core_day')} "
                      f"NLL {o.get('ppl_core_day')} | core GSM8K {o.get('gsm8k_core')} NLL {o.get('ppl_core')} "
                      f"| base NLL {d['firewall'].get('ppl_base')}")

    print("=== substrate transfer: SmolLM2-1.7B-Instruct, D6 (144 facts) ===")
    for night in ["full", "misslog", "markov", "hybrid", "recite"]:
        fs = sorted(glob.glob(os.path.join(R, f"loop_{night}_MSmolLM217BInst_D6_f24_s*.json")))
        if not fs:
            continue
        fins, recogs, fws = [], [], set()
        for f in fs:
            d = json.load(open(f)); t = [e for e in d["trajectory"] if "post_hits" in e]
            fins.append(sum(t[-1]["post_hits"])); recogs.append(t[-1].get("n_recognized"))
            fw = d.get("firewall", {}); fws.add((fw.get("gsm8k_base"), fw.get("gsm8k_off")))
        print(f"  {night:8s} final={fins}/144 mean={sum(fins)/len(fins):6.1f} recog={recogs} fw={sorted(fws)}")
