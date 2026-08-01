#!/usr/bin/env python3
"""1-task smoke (pre-reg 2026-08-01): does UN-differenced scan (base+lora likelihood)
carry relevance? Rank all 125 log lines under the macaron-task draft context by
(a) base-only lp, (b) base+lora lp, (c) dlp difference; plus next-token top-100 at the
elicitation position, base vs base+lora. Out results/inmind/rawscan_smoke.json"""
import json, os, re, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
HERE = os.path.dirname(__file__)

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
LOG = [f"[session {i:03d}] {t['user_message']}" for i, t in enumerate(tasks)]
full = json.load(open(os.path.join(HERE, "results", "inmind", "answers_full.json")))
frow = {r["task_id"]: r for r in full["rows"]}

# the macaron / tree-nut task
sel_i = next(i for i, t in enumerate(tasks) if "tree nut" in t["user_message"].lower()
             or "nut allergy" in t["user_message"].lower())
t = tasks[sel_i]
print(f"[rawscan] task_id={t['task_id']} fact={t['user_message'][:60]} q={t['query'][:60]}",
      flush=True)
draft = frow[t["task_id"]]["draft"]

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(HERE, "results", "inmind",
                                 "adapter_full.pt"), map_location=DEV), strict=False)
model.eval()

PRE = f"Context: {draft[:600]}\nRelevant user fact:"

@torch.no_grad()
def lp(text, use_adapter):
    full_txt = PRE + " " + text
    inp = tok(full_txt, return_tensors="pt").to(DEV)
    npre = tok(PRE, return_tensors="pt")["input_ids"].shape[1]
    if use_adapter:
        lg = model(**inp).logits[0]
    else:
        with model.disable_adapter():
            lg = model(**inp).logits[0]
    sm = torch.log_softmax(lg[:-1].float(), -1)
    ids = inp["input_ids"][0]
    return float(sm[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(len(ids) - npre, 1)

rows = []
for i, ln in enumerate(LOG):
    on = lp(tasks[i]["user_message"], True)
    off = lp(tasks[i]["user_message"], False)
    rows.append({"i": i, "on": on, "off": off, "dlp": on - off})
    if i % 25 == 0:
        print(f"[rawscan] {i}/125", flush=True)

def rank_of_gold(key, sign=1):
    order = sorted(rows, key=lambda r: -sign * r[key])
    return next(j for j, r in enumerate(order) if r["i"] == sel_i) + 1, \
           [(r["i"], LOG[r["i"]][:60]) for r in order[:5]]

rk_off, top_off = rank_of_gold("off")
rk_on, top_on = rank_of_gold("on")
rk_dlp, top_dlp = rank_of_gold("dlp")

STOP = set("about would could there their which where".split())
words = {w for w in re.findall(r"[a-zA-Z]{4,}", (t.get("entity_1") or "") + " " + t["user_message"])
         if w.lower() not in STOP}
ids = set()
for w in words:
    for v in (w, " " + w, w.lower(), " " + w.lower(), w.capitalize(), " " + w.capitalize()):
        enc = tok(v, add_special_tokens=False)["input_ids"]
        if enc:
            ids.add(enc[0])

@torch.no_grad()
def top100(use_adapter):
    inp = tok(PRE, return_tensors="pt").to(DEV)
    if use_adapter:
        lg = model(**inp).logits[0, -1].float()
    else:
        with model.disable_adapter():
            lg = model(**inp).logits[0, -1].float()
    topi = torch.topk(torch.log_softmax(lg, -1), 100).indices.tolist()
    hits = [tok.decode([i]) for i in topi if i in ids]
    return hits, [tok.decode([i]) for i in topi[:20]]

hits_on, t20_on = top100(True)
hits_off, t20_off = top100(False)

out = {"task_id": t["task_id"], "gold_rank": {"base_only": rk_off, "base_plus_lora": rk_on,
                                              "dlp": rk_dlp},
       "top5_base_only": top_off, "top5_base_plus_lora": top_on, "top5_dlp": top_dlp,
       "top100_bridge_hits": {"base_plus_lora": hits_on, "base_only": hits_off},
       "top20_tokens": {"base_plus_lora": t20_on, "base_only": t20_off}}
json.dump(out, open(os.path.join(HERE, "results", "inmind", "rawscan_smoke.json"), "w"),
          indent=1)
print(json.dumps(out, indent=1))
