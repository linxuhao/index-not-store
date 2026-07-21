#!/usr/bin/env python3
"""TOT-rescue END-STATE smoke (CLAIMS queue 6; pre-registered + SOTA'd 2026-07-20).

Per fact on a saved end-of-stream adapter:
  rank   : gold first-token rank at the answer slot (tests "high-rank-not-1" prediction)
  greedy : standard recall (rank-1 path)
  rescue : top-k first tokens -> each greedily completed (adapter ON, NO closed-world pool)
           -> selector = Δlp(candidate) = lp_adapter - lp_base (teacher-forced, presented regime)
Metrics by state (recalled / recog_only via 2-AFC margin): P(gold∈cands), selector accuracy,
net recall lift. Self-argmax caution: rank-1 candidate is the greedy incumbent by construction.
"""
import argparse, os, random, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L
_argv = sys.argv; sys.argv = ["phase5_accum.py"]
from phase5_accum import make_facts, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--dev", default="cuda:1")
args = ap.parse_args()
DEV = args.dev
L.check_env()

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
sd = torch.load(args.adapter, map_location=DEV)
res = model.load_state_dict(sd, strict=False)
model.eval()
print(f"[tot] adapter loaded unexpected={len(res.unexpected_keys)}", flush=True)

facts = make_facts(args.n, args.seed)

@torch.no_grad()
def slot_logits(prompt):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    return model(**ids).logits[0, -1], ids

@torch.no_grad()
def greedy_from(prompt, max_new=10, force_first=None):
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].to(DEV)
    outp = []
    if force_first is not None:
        ids = torch.cat([ids, torch.tensor([force_first], device=DEV)])
        outp.append(force_first)
    for _ in range(max_new):
        nxt = int(model(input_ids=ids[None]).logits[0, -1].argmax())
        if nxt == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([nxt], device=DEV)])
        outp.append(nxt)
        if "." in tok.decode([nxt]) or "\n" in tok.decode([nxt]):
            break
    return tok.decode(outp, skip_special_tokens=True).strip().rstrip(".")

@torch.no_grad()
def answer_lp(prompt, ans, use_adapter):
    full = tok(prompt + " " + ans + ".", return_tensors="pt").to(DEV)
    npre = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    ctx = model if use_adapter else model.disable_adapter()
    if use_adapter:
        logits = model(**full).logits[0]
    else:
        with model.disable_adapter():
            logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())

rows = []
for f in facts:
    p = cloze(f)
    gold = f["answer"]
    gold_first = tok(" " + gold, add_special_tokens=False)["input_ids"][0]
    logits, _ = slot_logits(p)
    order = torch.argsort(logits, descending=True)
    grank = int((order == gold_first).nonzero()[0]) + 1
    # greedy recall
    g = greedy_from(p)
    hit = int(L.contains_match_ci(gold, g))
    # 2-AFC margin (state)
    others = [x["answer"] for x in facts if x["fid"] != f["fid"]]
    dis = random.Random(31 * f["fid"] + args.seed).choice(others)
    marg = answer_lp(p, gold, True) - answer_lp(p, dis, True)
    # candidates from top-k first tokens
    cands = []
    for t in order[: args.topk].tolist():
        c = greedy_from(p, force_first=t)
        if c and c not in cands:
            cands.append(c)
    gold_in = any(L.contains_match_ci(gold, c) for c in cands)
    # two selectors logged: full-seq adapter lp (beam-style rescoring) vs Δlp (familiarity —
    # s1234 v1 showed Δlp picks written-syllable frankensteins: authentication ≠ answer selection)
    scored = []
    for c in cands:
        la = answer_lp(p, c, True)
        lb = answer_lp(p, c, False)
        scored.append({"c": c, "lpA": la, "dlp": la - lb})
    sel_lpa = max(scored, key=lambda x: x["lpA"])["c"] if scored else ""
    sel_dlp = max(scored, key=lambda x: x["dlp"])["c"] if scored else ""
    rows.append({"fid": f["fid"], "state": "recalled" if hit else ("recog_only" if marg > 0 else "gone"),
                 "grank": grank, "greedy_hit": hit, "gold_in_cands": int(gold_in),
                 "n_cands": len(cands),
                 "rescued_lpa": int(L.contains_match_ci(gold, sel_lpa)),
                 "rescued_dlp": int(L.contains_match_ci(gold, sel_dlp)),
                 "sel_lpa": sel_lpa[:40], "sel_dlp": sel_dlp[:40],
                 "cands": [{"c": s["c"][:30], "lpA": round(s["lpA"], 2), "dlp": round(s["dlp"], 2)} for s in scored]})
    print(f"[tot] f{f['fid']:2d} {rows[-1]['state']:10s} rank={grank:4d} hit={hit} in={int(gold_in)} "
          f"lpa={rows[-1]['rescued_lpa']} dlp={rows[-1]['rescued_dlp']}", flush=True)

import json
byst = {}
for s in ("recalled", "recog_only", "gone"):
    v = [r for r in rows if r["state"] == s]
    if v:
        byst[s] = {"n": len(v), "gold_in_cands": sum(r["gold_in_cands"] for r in v),
                   "rescued_lpa": sum(r["rescued_lpa"] for r in v),
                   "rescued_dlp": sum(r["rescued_dlp"] for r in v),
                   "median_rank": sorted(r["grank"] for r in v)[len(v) // 2]}
summary = {"adapter": os.path.basename(args.adapter), "topk": args.topk,
           "greedy_recall": sum(r["greedy_hit"] for r in rows),
           "rescued_lpa": sum(r["rescued_lpa"] for r in rows),
           "rescued_dlp": sum(r["rescued_dlp"] for r in rows), "n": len(rows), "by_state": byst}
out = os.path.join(os.path.dirname(__file__), "results", "jspace",
                   f"totrescue_{os.path.basename(args.adapter).replace('.pt','')}.json")
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
