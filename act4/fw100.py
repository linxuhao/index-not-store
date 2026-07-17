#!/usr/bin/env python3
"""Expanded capability probe (#8, review-prep 2026-07-12): GSM8K-100 instead of the 10-item pilot.

Answers the audit weakness "10-item GSM8K (+-0.1 resolution) cannot resolve modest degradation":
  - base GSM8K-100 for both substrates (the firewall reference at 10x resolution);
  - ON-state GSM8K-100 with each saved misslog D6 core mounted (paper-2 "not aphasic" claim).

Frozen 100-item subset: seeded sample of openai/gsm8k test, excluding the 10 pilot indices,
saved to gsm8k_fw100_ids.json at the repo root on first run.
"""
import argparse, json, os, random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--cores", nargs="*", default=[])   # core .pt files to mount (rank 32, adapter 'core')
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
args = ap.parse_args()
DEV = args.dev

HERE = os.path.dirname(__file__)
IDS100 = os.path.join(HERE, "..", "gsm8k_fw100_ids.json")   # frozen 100-item subset (repo root)
PILOT = os.path.join(HERE, "gsm8k_pilot_ids.json")

if not os.path.exists(IDS100):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    pilot = set(json.load(open(PILOT))["indices"])
    pool = [i for i in range(len(ds)) if i not in pilot]
    idx = sorted(random.Random(args.seed).sample(pool, args.n))
    json.dump({"indices": idx, "seed": args.seed, "excludes": sorted(pilot)}, open(IDS100, "w"))
    print(f"[fw100] froze {args.n} ids -> {IDS100}")
items = L.load_gsm8k_subset(IDS100)[: args.n]

L.check_env()
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
base.eval()

out = {"model": args.model, "n": len(items), "seed": args.seed}
acc = L.eval_gsm8k(base, tok, items, device=DEV)
out["base"] = acc
print(f"[fw100] {args.model} BASE gsm8k-{len(items)} = {acc:.2f}")

if args.cores:
    cfg = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear",
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg, adapter_name="core")
    for cp in args.cores:
        sd = torch.load(cp, map_location=DEV)
        res = model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(res.unexpected_keys)
        assert n_loaded > 0 and not res.unexpected_keys, \
            f"core mount mismatch: {len(res.unexpected_keys)} unexpected of {len(sd)}"
        model.set_adapter("core"); model.eval()
        acc = L.eval_gsm8k(model, tok, items, device=DEV)
        key = os.path.basename(cp).replace(".pt", "")
        out[key] = acc
        print(f"[fw100] core={key} ({n_loaded} tensors) gsm8k-{len(items)} = {acc:.2f}")

tag = "" if args.model == "Qwen/Qwen3.5-2B" else \
    "_M" + args.model.split("/")[-1].replace("-", "").replace(".", "")[:14]
fn = f"fw100{tag}_n{len(items)}_s{args.seed}.json"
json.dump(out, open(os.path.join(args.out, fn), "w"), indent=2)
print(f"[fw100] saved {fn}")
