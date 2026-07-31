#!/usr/bin/env python3
"""GSM8K-100 firewall for raw .pt LoRA adapters (rank-64 'mem'), 14b capability check."""
import argparse, json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--dev", default="cuda:1")
ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
args = ap.parse_args()
L.check_env()
HERE = os.path.dirname(__file__)
items = L.load_gsm8k_subset(os.path.join(HERE, "gsm8k_fw100_ids.json"))[:100]

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32).to(args.dev)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(args.adapter, map_location=args.dev), strict=False)
model.eval()

with model.disable_adapter():
    b = L.eval_gsm8k(model, tok, items, device=args.dev)
print(f"[fw_pt] BASE = {b:.2f}", flush=True)
a = L.eval_gsm8k(model, tok, items, device=args.dev)
print(f"[fw_pt] +ADAPTER = {a:.2f}", flush=True)
NEUTRAL = ("The committee reviewed the proposal in detail and concluded that the schedule "
           "was feasible, provided that the funding arrived before the end of the quarter. "
           "Several members raised questions about maintenance costs, which the chair "
           "promised to address in a follow-up meeting next month. The final vote was "
           "postponed until all departments had submitted their annual reports.")
def nll(use):
    ids = tok(NEUTRAL, return_tensors="pt").to(args.dev)
    with torch.no_grad():
        if use:
            lg = model(**ids, labels=ids["input_ids"])
        else:
            with model.disable_adapter():
                lg = model(**ids, labels=ids["input_ids"])
    return float(lg.loss)
nb, na = nll(False), nll(True)
print(f"[fw_pt] accent NLL/token base={nb:.3f} adapter={na:.3f} (+{na-nb:.3f})", flush=True)
json.dump({"base": b, "adapter_acc": a, "nll_base": nb, "nll_adapter": na,
           "adapter": os.path.basename(args.adapter)},
          open(os.path.join(HERE, "results", f"fwpt_{os.path.basename(args.adapter).replace('.pt','')}.json"), "w"))
