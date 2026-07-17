#!/usr/bin/env python3
"""Minimal familiarity-gate demo (pre-registered CLAIMS 2026-07-15).

96 question-form queries (48 written + 48 never-written); log contains only the written facts.
- Ungated: adapter-mounted model answers everything; any value-like answer on a never-written
  attribute is a fabrication.
- Gated: generate a candidate answer, score its familiarity dlp = lp_adapter - lp_base of the
  CANDIDATE tokens; below threshold -> "don't know / consult log" (log hit iff written);
  above -> trust candidate.
Threshold: calibrated on --calibrate-seed's adapter (max Youden J on candidate-dlp), then FROZEN
and applied to the other seeds (pass --threshold to reuse).
Outputs per adapter: ungated fabrication rate on novel, gated fabrication rate, written-side
correct-handling rate (familiar -> log lookup, which is correct by construction; we report the
familiar-recall = fraction of written queries the gate sends to the log OR trusts a correct answer).
"""
import argparse, json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import lib as L

_argv = sys.argv
sys.argv = ["accum.py"]
from accum import make_facts, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--novel-seed-offset", type=int, default=50000)
ap.add_argument("--threshold", type=float, default=None)  # frozen threshold; None = calibrate here
ap.add_argument("--contrast", action="store_true")  # contrastive candidate scoring (subtract reference-pseudoword dlp)
ap.add_argument("--probe-form", choices=["question", "cloze"], default="question")  # gate probes in its own write form (form-alignment design rule)
ap.add_argument("--signal", choices=["cand_dlp", "slot_kl"], default="cand_dlp")  # slot_kl: answer-independent KL(adapter||base) at the first answer-slot position
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "familiarity"))
args = ap.parse_args()
DEV = args.dev

L.check_env(); torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
sd = torch.load(args.adapter, map_location=DEV)
res = model.load_state_dict(sd, strict=False)
assert not res.unexpected_keys, "adapter mount mismatch"
print(f"[gate] mounted {os.path.basename(args.adapter)}")


def attr_of(f):
    return f["statement"][len("The user's "): f["statement"].find(" is ")]


written = make_facts(args.n, args.seed)
novel_pool = make_facts(args.n * 2, args.seed + args.novel_seed_offset)
wvals = {f["answer"] for f in written}
wattrs = {attr_of(f) for f in written}
novel = [f for f in novel_pool if f["answer"] not in wvals and attr_of(f) not in wattrs][: args.n]
log_store = {attr_of(f): f["answer"] for f in written}          # the ground-truth log


def question(f):
    if args.probe_form == "cloze":
        return f"The user's {attr_of(f)} is"
    return f"Q: What is the user's {attr_of(f)}?\nA: It is"


@torch.no_grad()
def generate_candidate(f):
    model.eval()
    ids = tok(question(f), return_tensors="pt").to(DEV)
    g = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
    comp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
    return comp


@torch.no_grad()
def cand_dlp(f, cand):
    """Per-token adapter-vs-base dlp of the candidate as continuation of the question."""
    if not cand:
        return -99.0
    pre = question(f)
    full = tok(pre + " " + cand, return_tensors="pt").to(DEV)
    npre = tok(pre, return_tensors="pt")["input_ids"].shape[1]
    ids = full["input_ids"][0]
    n = len(ids) - npre
    if n < 1:
        return -99.0
    la = model(**full).logits[0]
    with model.disable_adapter():
        lb = model(**full).logits[0]
    lpa = torch.log_softmax(la[:-1], -1)
    lpb = torch.log_softmax(lb[:-1], -1)
    idx = torch.arange(npre - 1, len(ids) - 1)
    return float((lpa[idx, ids[npre:]] - lpb[idx, ids[npre:]]).sum()) / n


@torch.no_grad()
def slot_kl(f):
    """Answer-independent familiarity: KL(adapter || base) of the next-token distribution
    at the cloze slot (memory = a specific spike in the answer-slot distribution)."""
    ids = tok(question(f), return_tensors="pt").to(DEV)
    la = model(**ids).logits[0, -1]
    with model.disable_adapter():
        lb = model(**ids).logits[0, -1]
    pa = torch.log_softmax(la, -1)
    pb = torch.log_softmax(lb, -1)
    return float((pa.exp() * (pa - pb)).sum())


def is_valuelike(ans):
    """A fabrication on novel = any confident value-ish answer (not an abstention)."""
    low = ans.lower()
    return not any(k in low for k in ("don't know", "do not know", "unknown", "not sure",
                                      "no information", "cannot", "can't", "unsure"))


REF_ANSWERS = [f["answer"] for f in make_facts(24, args.seed + 90001)]
import random as _rnd
_rng = _rnd.Random(args.seed)

rows = []
for grp, facts in (("written", written), ("novel", novel)):
    for f in facts:
        cand = generate_candidate(f)
        if args.signal == "slot_kl":
            d = slot_kl(f)
        else:
            d = cand_dlp(f, cand)
            if args.contrast:
                refs = _rng.sample(REF_ANSWERS, 4)
                d = d - sum(cand_dlp(f, ra) for ra in refs) / len(refs)
        rows.append({"grp": grp, "attr": attr_of(f), "gold": f["answer"] if grp == "written" else None,
                     "cand": cand, "dlp": round(d, 3), "valuelike": is_valuelike(cand),
                     "cand_correct": (grp == "written" and L.contains_match_ci(f["answer"], cand))})

# threshold: calibrate (max Youden J on dlp, written=pos novel=neg) or use frozen
if args.threshold is None:
    cands = sorted({r["dlp"] for r in rows})
    best_t, best_j = None, -1
    P = [r["dlp"] for r in rows if r["grp"] == "written"]
    N = [r["dlp"] for r in rows if r["grp"] == "novel"]
    for t in cands:
        tpr = sum(1 for x in P if x > t) / len(P)
        fpr = sum(1 for x in N if x > t) / len(N)
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
    thr = best_t
    print(f"[gate] calibrated threshold={thr:.3f} (Youden J={best_j:.3f}) — FREEZE for other seeds")
else:
    thr = args.threshold
    print(f"[gate] using frozen threshold={thr:.3f}")

nov = [r for r in rows if r["grp"] == "novel"]
wri = [r for r in rows if r["grp"] == "written"]
ungated_fab = sum(1 for r in nov if r["valuelike"]) / len(nov)
gated_fab = sum(1 for r in nov if r["dlp"] > thr and r["valuelike"]) / len(nov)
# written handling: familiar -> log lookup (correct by construction); we also count trusted-correct
wri_familiar = sum(1 for r in wri if r["dlp"] > thr) / len(wri)
wri_handled = sum(1 for r in wri if r["dlp"] > thr) / len(wri)  # log hit = correct
wri_trust_correct = sum(1 for r in wri if r["dlp"] > thr and r["cand_correct"]) / len(wri)

summary = {"adapter": os.path.basename(args.adapter), "seed": args.seed, "threshold": round(thr, 3),
           "calibrated_here": args.threshold is None,
           "ungated_fabrication_novel": round(ungated_fab, 3),
           "gated_fabrication_novel": round(gated_fab, 3),
           "written_familiar_rate": round(wri_familiar, 3),
           "written_handled_via_log": round(wri_handled, 3),
           "written_trusted_and_correct": round(wri_trust_correct, 3)}
print(f"[gate] ungated fabrication on novel: {ungated_fab:.1%} -> gated: {gated_fab:.1%}")
print(f"[gate] written: familiar {wri_familiar:.1%} (log-handled {wri_handled:.1%}, "
      f"trusted+correct {wri_trust_correct:.1%})")

os.makedirs(args.out, exist_ok=True)
tagc = "_contrast" if args.contrast else ""
fn = f"gate_{args.probe_form}_{args.signal}{tagc}_{os.path.basename(args.adapter).replace('.pt','')}.json"
json.dump({"summary": summary, "rows": rows}, open(os.path.join(args.out, fn), "w"), indent=2)
print(f"[gate] saved {fn}")
