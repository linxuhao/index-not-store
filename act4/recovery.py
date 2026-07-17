#!/usr/bin/env python3
"""Suppression recovery probe (#9, review-prep 2026-07-12) — Ebbinghaus savings paradigm.

Question: is EWC/SmolLM2's 0/48 recall with above-chance recognition a latent (suppressed) trace,
or a mis-scaled regularizer that simply froze the adapter (hostile reading of paper-1 Table 4)?

Design: per-fact isolated relearning at matched tiny budget.
  For each fact: reset adapter to the saved end-of-stream state, take k value-token grad steps on
  that fact alone (same loss/optimizer/lr as the stream writes, no EWC penalty), probe that fact.
  Arms: OLD = the 48 facts written during the stream (recall 0/48 now).
        NOVEL = 48 same-distribution facts never written (make_facts with a disjoint seed).
Savings = OLD re-acquired faster than NOVEL at matched k  ->  latent storage (suppression).
No savings (OLD ~ NOVEL)  ->  the trace is gone or unusable; downgrade the suppression claim.

Also logs gold-answer logprob delta per fact (continuous savings measure) and a zero-step
sanity probe (expect ~0/48 recall on OLD).
"""
import argparse, copy, json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import lib as L

_argv = sys.argv
sys.argv = ["accum.py"]          # shim: accum parses args at import
from accum import make_facts, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)          # saved .pt from recovery_rerun
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--steps", type=int, nargs="*", default=[1, 2])
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--seed", type=int, required=True)   # stream seed (fact regeneration)
ap.add_argument("--novel-seed-offset", type=int, default=50000)
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "recovery_rerun"))
args = ap.parse_args()
DEV = args.dev

L.check_env()
torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")   # accum's adapter name

sd0 = torch.load(args.adapter, map_location=DEV)
res = model.load_state_dict(sd0, strict=False)
n_loaded = len(sd0) - len(res.unexpected_keys)
assert n_loaded > 0 and not res.unexpected_keys, \
    f"adapter mismatch: {len(res.unexpected_keys)} unexpected of {len(sd0)}"
print(f"[recovery] mounted {n_loaded} adapter tensors from {os.path.basename(args.adapter)}")

old = make_facts(args.n, args.seed)
novel = make_facts(args.n, args.seed + args.novel_seed_offset)
old_vals = {f["answer"] for f in old}
novel = [f for f in novel if f["answer"] not in old_vals]
print(f"[recovery] {len(old)} old facts, {len(novel)} novel facts (collisions dropped)")


@torch.no_grad()
def gold_lp(f):
    """Mirror of accum.answer_lp(model, tok, f, f['answer'])."""
    model.eval()
    full = tok(cloze(f) + " " + f["answer"] + ".", return_tensors="pt").to(DEV)
    npre = tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]
    logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())


@torch.no_grad()
def recalled(f):
    """Mirror of accum.recall(): greedy cloze, 12 new tokens, first line, ci-match."""
    model.eval()
    ids = tok(cloze(f), return_tensors="pt").to(DEV)
    g = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
    comp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
    return bool(L.contains_match_ci(f["answer"], comp))


def relearn(f, k):
    """Reset adapter -> k grad steps on f (same value-token loss as stream writes, no EWC)."""
    model.load_state_dict(sd0, strict=False)
    lp_params = [p for n_, p in model.named_parameters() if "lora" in n_.lower()]
    opt = torch.optim.AdamW(lp_params, lr=args.lr)
    model.train()
    for _ in range(k):
        b = tok(f["statement"], return_tensors="pt").to(DEV)
        labels = b["input_ids"].clone()
        labels[:, : tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]] = -100
        out = model(**b, labels=labels)
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(lp_params, 1.0); opt.step()


# zero-step sanity + baseline logprobs at the saved state
model.load_state_dict(sd0, strict=False)
sanity = sum(recalled(f) for f in old)
print(f"[recovery] sanity zero-step recall on OLD: {sanity}/{len(old)} (expect ~0)")
lp0 = {("old", i): gold_lp(f) for i, f in enumerate(old)}
lp0.update({("novel", i): gold_lp(f) for i, f in enumerate(novel)})

out = {"adapter": os.path.basename(args.adapter), "model": args.model, "seed": args.seed,
       "sanity_zero_step_recall": sanity, "arms": {}}
for k in args.steps:
    for arm, facts in (("old", old), ("novel", novel)):
        rec, dlp = [], []
        for i, f in enumerate(facts):
            relearn(f, k)
            rec.append(int(recalled(f)))
            dlp.append(gold_lp(f) - lp0[(arm, i)])
        out["arms"][f"{arm}_k{k}"] = {"recall": sum(rec), "n": len(facts),
                                      "rec_vec": rec, "dlp_mean": sum(dlp) / len(dlp),
                                      "dlp_vec": [round(x, 3) for x in dlp]}
        print(f"[recovery] {arm} k={k}: recall {sum(rec)}/{len(facts)} "
              f"dlp_mean {sum(dlp)/len(dlp):+.2f}")

os.makedirs(args.out, exist_ok=True)
fn = f"recovery_ewc_l300_MSmolLM217BInst_k{'-'.join(map(str,args.steps))}_s{args.seed}.json"
json.dump(out, open(os.path.join(args.out, fn), "w"), indent=2)
print(f"[recovery] saved {fn}")

# reset once more so the mounted state isn't left mid-relearn if reused interactively
model.load_state_dict(sd0, strict=False)
