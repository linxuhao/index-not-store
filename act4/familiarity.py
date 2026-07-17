#!/usr/bin/env python3
"""Familiarity-gated retrieval probe (CLAIMS OPEN/NEXT #0, 2026-07-15).

Question: can the memory adapter's CANDIDATE-FREE familiarity signal
    dlp(f, form) = lp_adapter(answer | prompt(form)) - lp_base(answer | prompt(form))
discriminate written vs never-written facts — and does it survive a FORM CHANGE
(interrogative) that kills recall (1-6/48)?

Per adapter: 48 written facts (stream seed) vs 48 never-written (disjoint seed, collision-
filtered), dlp under (a) write-form cloze, (b) question form. Report AUC per form.
AUC(question-form) high => familiarity generalizes across forms => gate is viable.
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
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--seed", type=int, required=True)      # stream seed of the adapter
ap.add_argument("--novel-seed-offset", type=int, default=50000)
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
assert not res.unexpected_keys and len(sd) > 0, "adapter mount mismatch"
print(f"[fam] mounted {len(sd)} tensors from {os.path.basename(args.adapter)}")

def attr_of(f):
    return f["statement"][len("The user's "): f["statement"].find(" is ")]

written = make_facts(args.n, args.seed)
novel = make_facts(args.n * 2, args.seed + args.novel_seed_offset)
wvals = {f["answer"] for f in written}
wattrs = {attr_of(f) for f in written}
novel = [f for f in novel if f["answer"] not in wvals and attr_of(f) not in wattrs][: args.n]
print(f"[fam] {len(written)} written vs {len(novel)} novel (answer+attr collision-filtered)")


def prompt_of(f, form):
    if form == "cloze":
        return cloze(f)
    return f"Q: What is the user's {attr_of(f)}?\nA: It is"


@torch.no_grad()
def ans_lp(f, form, use_adapter):
    model.eval()
    pre = prompt_of(f, form)
    full = tok(pre + " " + f["answer"], return_tensors="pt").to(DEV)
    npre = tok(pre, return_tensors="pt")["input_ids"].shape[1]
    if use_adapter:
        logits = model(**full).logits[0]
    else:
        with model.disable_adapter():
            logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    n_ans = len(ids) - npre
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(n_ans, 1)


def auc(pos, neg):
    """AUC via rank statistic."""
    pairs = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return pairs / (len(pos) * len(neg))


import random as _rnd


@torch.no_grad()
def span_dlp(pre_text, span_text):
    """Per-token adapter-vs-base dlp of span_text continuing pre_text (prompt-side familiarity)."""
    full = tok(pre_text + span_text, return_tensors="pt").to(DEV)
    npre = tok(pre_text, return_tensors="pt")["input_ids"].shape[1]
    ids = full["input_ids"][0]
    n_span = len(ids) - npre
    if n_span < 1:
        return 0.0
    logits_a = model(**full).logits[0]
    with model.disable_adapter():
        logits_b = model(**full).logits[0]
    lpa = torch.log_softmax(logits_a[:-1], -1)
    lpb = torch.log_softmax(logits_b[:-1], -1)
    idx = torch.arange(npre - 1, len(ids) - 1)
    return float((lpa[idx, ids[npre:]] - lpb[idx, ids[npre:]]).sum()) / n_span


REF_ANSWERS = [f["answer"] for f in make_facts(24, args.seed + 90001)]  # reference pseudowords

out = {"adapter": os.path.basename(args.adapter), "model": args.model, "seed": args.seed,
       "n_written": len(written), "n_novel": len(novel), "signals": {}}
rnd = _rnd.Random(args.seed)
for form in ("cloze", "question"):
    # S0 base-only lp (the cheap-uncertainty baseline a skeptic would propose): base is identical
    # across arms, so this SHOULD sit near 0.5 while dlp moves — the 2501.12835 defense row.
    b_w = [ans_lp(f, form, False) for f in written]
    b_n = [ans_lp(f, form, False) for f in novel]
    a_w = [ans_lp(f, form, True) for f in written]
    a_n = [ans_lp(f, form, True) for f in novel]
    dl_w = [a - b for a, b in zip(a_w, b_w)]
    dl_n = [a - b for a, b in zip(a_n, b_n)]
    # S2 contrastive: gold dlp minus mean dlp of K reference pseudowords in same prompt
    def contrast(f, gold_dlp):
        refs = rnd.sample(REF_ANSWERS, 4)
        ref_dlp = []
        for ra in refs:
            g = dict(f); g = {**f}
            pre = prompt_of(f, form)
            full_gold_ans = ra
            # reuse ans_lp machinery via temporary answer swap
            fa = {**f, "answer": ra}
            ref_dlp.append(ans_lp(fa, form, True) - ans_lp(fa, form, False))
        return gold_dlp - sum(ref_dlp) / len(ref_dlp)
    c_w = [contrast(f, d) for f, d in zip(written, dl_w)]
    c_n = [contrast(f, d) for f, d in zip(novel, dl_n)]
    # S3 prompt-side attr familiarity (deployable: no answer needed)
    if form == "cloze":
        p_w = [span_dlp("The user's ", attr_of(f)) for f in written]
        p_n = [span_dlp("The user's ", attr_of(f)) for f in novel]
    else:
        p_w = [span_dlp("Q: What is the user's ", attr_of(f)) for f in written]
        p_n = [span_dlp("Q: What is the user's ", attr_of(f)) for f in novel]
    for name, (w, n) in {"S0_base_lp": (b_w, b_n), "S1_raw_answer": (dl_w, dl_n),
                         "S2_contrastive": (c_w, c_n), "S3_prompt_attr": (p_w, p_n)}.items():
        a = auc(w, n)
        out["signals"][f"{form}:{name}"] = {"auc": round(a, 3),
                                            "written_mean": round(sum(w)/len(w), 3),
                                            "novel_mean": round(sum(n)/len(n), 3)}
        print(f"[fam] {form}:{name}: AUC={a:.3f}  written {sum(w)/len(w):+.3f} vs novel {sum(n)/len(n):+.3f}")

os.makedirs(args.out, exist_ok=True)
fn = f"fam_{os.path.basename(args.adapter).replace('.pt','')}.json"
json.dump(out, open(os.path.join(args.out, fn), "w"), indent=2)
print(f"[fam] saved {fn}")
