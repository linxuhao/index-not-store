#!/usr/bin/env python3
"""Recognition-meter zero-point control (2026-07-17, user audit question).

Runs the EXACT 2-AFC recognition comparison from accum.recall() — same facts, same
deterministic distractor draw (random.Random(31*fid+seed)), same sum-logprob answer_lp — on the
UNTRAINED BASE model (no adapter). Expected if the meter is unbiased: ~24/48 margins > 0.
Also reports (a) per-token-normalized margins (length-bias check) and (b) gold/distractor token
count distributions.
"""
import argparse, random, sys, os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lib as L

_argv = sys.argv
sys.argv = ["accum.py"]
from accum import make_facts, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--seeds", type=int, nargs="*", default=[1234, 2025, 777])
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--dev", default="cuda:0")
args = ap.parse_args()
DEV = args.dev

L.check_env()
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
model.eval()


@torch.no_grad()
def answer_lp(f, ans):
    full = tok(cloze(f) + " " + ans + ".", return_tensors="pt").to(DEV)
    npre = tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]
    logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    tot = float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())
    return tot, len(ids) - npre


for seed in args.seeds:
    facts = make_facts(args.n, seed)
    pos_sum = pos_norm = 0
    dtok = []
    for f in facts:
        others = [x["answer"] for x in facts if x["fid"] != f["fid"]]
        dis = random.Random(31 * f["fid"] + seed).choice(others)   # exact accum draw
        lg, ng = answer_lp(f, f["answer"])
        ld, nd = answer_lp(f, dis)
        pos_sum += int(lg - ld > 0)
        pos_norm += int(lg / ng - ld / nd > 0)
        dtok.append((ng, nd))
    gt = [a for a, _ in dtok]; dt = [b for _, b in dtok]
    print(f"[zero] seed={seed}: BASE 2-AFC sum-lp {pos_sum}/{args.n} | per-token-norm {pos_norm}/{args.n} "
          f"| gold tokens mean {sum(gt)/len(gt):.2f} vs distractor {sum(dt)/len(dt):.2f}")
