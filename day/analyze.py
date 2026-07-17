"""accum result analysis: finals + re-instatement (recovery) stats + first-miss + recognition, per arm.

PINNED definitions (the ones behind every number in the paper tables):
- recovery: a fact is a recovery CANDIDATE if it was recalled at some probe and later missed one; it
  RECOVERED if it is recalled again at any probe after its first miss. recovery-% =
  recovered/candidates, pooled over seeds.
- first-miss: for each recovery candidate, the number of subsequent writes between the fact's own
  write and its first post-acquisition miss; the table reports the median pooled over seeds.
- alive: facts recalled at the final probe, summed over seeds (/48 per seed).
- recognition: `n_recognized` at the final probe (2-AFC logprob margin > 0; chance = n_stream/2).
All computed from the per-probe `hits` vectors in the result JSONs."""
import json, glob, os, statistics
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def stats(path):
    d = json.load(open(path))
    curve = d["curve"]
    n = d["n_stream"]
    # build per-fact timeline across probes
    probes = [(c["k"], c["hits"]) for c in curve]
    alive_end = sum(probes[-1][1])
    missed, recovered, first_miss = 0, 0, []
    for fid in range(n):
        seq = [(k, h[fid]) for k, h in probes if len(h) > fid]
        hits = [v for _, v in seq]
        if 1 not in hits:
            continue  # never stored -> not a recovery candidate
        first = hits.index(1)
        after = hits[first:]
        if 0 in after:
            missed += 1
            i0 = first + after.index(0)
            first_miss.append(seq[i0][0] - (fid + 1))  # writes since this fact's own write
            if 1 in hits[i0:]:
                recovered += 1
    return d["final_n_recalled"], alive_end, missed, recovered, d.get("firewall") or {}, first_miss

if __name__ == "__main__":
    for mech in ["naked", "bf_N32_dt0.5", "ewc_l300", "replay", "ewcreplay_l300"]:
        for arm in ["", "_Learly", "_Lmid", "_Llate", "_Pmiss", "_Llate_Pmiss", "_Fcounterfact", "_Pmiss_Fcounterfact",
                    "_MSmolLM217BInst", "_Pmiss_MSmolLM217BInst"]:
            files = sorted(glob.glob(os.path.join(R, f"accum_{mech}{arm}_n48_pe2_s*.json")))
            if not files:
                continue
            rows = [stats(f) for f in files]
            fin = [r[0] for r in rows]
            tot_r = sum(r[3] for r in rows); tot_m = sum(r[2] for r in rows)
            pct = 100 * tot_r / tot_m if tot_m else float("nan")
            alive = sum(r[1] for r in rows)
            fm = [x for r in rows for x in r[5]]
            med_fm = statistics.median(fm) if fm else float("nan")
            fw = [f"{r[4].get('gsm8k_base')}->{r[4].get('gsm8k_off')}" for r in rows if r[4]]
            finals = [json.load(open(f))["curve"][-1] for f in files]
            nrec = [c.get("n_recognized") for c in finals]
            para = ["%d/%d" % (sum(h for h in c["para_hits"] if h), sum(1 for h in c["para_hits"] if h is not None))
                    for c in finals if c.get("para_hits")]
            print(f"{mech:16s} {arm or '_all':12s} n={len(files)}  final={fin} mean={sum(fin)/len(fin):5.1f}  "
                  f"first-miss(med)={med_fm:g}  alive={alive}/{48*len(files)}  "
                  f"recover={tot_r}/{tot_m} ({pct:.0f}%)  recog={nrec}"
                  + (f"  para={para}" if para else "") + (f"  firewall={fw}" if fw else ""))

    print("--- self-test cadence sweep (miss policy, ewcreplay) ---")
    for pe in [6, 12]:
        files = sorted(glob.glob(os.path.join(R, f"accum_ewcreplay_l300_Pmiss_n48_pe{pe}_s*.json")))
        if not files:
            continue
        rows = [stats(f) for f in files]
        fin = [r[0] for r in rows]
        tot_r = sum(r[3] for r in rows); tot_m = sum(r[2] for r in rows)
        print(f"  pe{pe}: final={fin} mean={sum(fin)/len(fin):5.1f}  recover={100*tot_r/max(tot_m,1):3.0f}%")
