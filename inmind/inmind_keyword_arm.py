#!/usr/bin/env python3
"""chain31 (pre-reg 2026-08-01): confirmatory arm for the category-tier
elicit-keyword signal. All 125 tasks: generate the elicitation keyword adapter-ON
and adapter-OFF; retrieve with each via the A3c machinery (grep + embed top-4, <=6
lines); primary readout = Delta gold_in(ON-OFF), stratified by domain.
Out results/inmind/keyword_arm.json"""
import json, os, re, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
HERE = os.path.dirname(__file__)

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
N = len(tasks)
LOG = [f"[session {i:03d}] {t['user_message']}" for i, t in enumerate(tasks)]
full = json.load(open(os.path.join(HERE, "results", "inmind", "answers_full.json")))
frow = {r["task_id"]: r for r in full["rows"]}

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(HERE, "results", "inmind",
                                 "adapter_full.pt"), map_location=DEV), strict=False)
model.eval()
etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()

@torch.no_grad()
def embed(texts):
    b = etok(texts, padding=True, truncation=True, return_tensors="pt").to(DEV)
    return torch.nn.functional.normalize(emb(**b).last_hidden_state[:, 0], dim=-1)

LOGV = embed(LOG)

def etop(text, k):
    qv = embed([text])
    return [LOG[i] for i in torch.argsort((LOGV @ qv.T).squeeze(1), descending=True)[:k].tolist()]

ELICIT = ("Question: {q}\n\nBefore answering, what single keyword about the user's "
          "personal situation should be checked in the memory log? Keyword:")

@torch.no_grad()
def gen_keyword(t, use_adapter):
    p = tok.apply_chat_template([{"role": "user", "content": ELICIT.format(q=t["query"])}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    if use_adapter:
        g = model.generate(**inp, max_new_tokens=10, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    else:
        with model.disable_adapter():
            g = model.generate(**inp, max_new_tokens=10, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def kw_retrieve(kw, draft):
    terms = [w.strip("-• ").lower() for w in re.split(r"[\n,;/]", kw) if w.strip()][:5]
    hits = [ln for ln in LOG if any(len(w) > 3 and w in ln.lower() for w in terms)][:4]
    return list(dict.fromkeys(hits + etop(kw or draft, 4)))[:6]

rows = []
for i, t in enumerate(tasks):
    draft = frow[t["task_id"]]["draft"]
    kw_on = gen_keyword(t, True)
    kw_off = gen_keyword(t, False)
    r_on = kw_retrieve(kw_on, draft)
    r_off = kw_retrieve(kw_off, draft)
    gold_ln = LOG[i]
    row = {"task_id": t["task_id"], "domain": t["domain"],
           "kw_on": kw_on, "kw_off": kw_off,
           "gold_in": {"ON": int(gold_ln in r_on), "OFF": int(gold_ln in r_off)},
           "ret_on": r_on, "ret_off": r_off}
    rows.append(row)
    print(f"[kwarm] {i:3d} ON={row['gold_in']['ON']} OFF={row['gold_in']['OFF']} "
          f"kw_on={kw_on[:20]!r} kw_off={kw_off[:20]!r}", flush=True)

g_on = sum(r["gold_in"]["ON"] for r in rows)
g_off = sum(r["gold_in"]["OFF"] for r in rows)
e16 = {r["task_id"]: r["gold_in"]["A3a16"] for r in full["rows"]}
new_over_e16 = sum(1 for r in rows if r["gold_in"]["ON"] == 1 and e16[r["task_id"]] == 0)
by_dom = {}
for r in rows:
    d = by_dom.setdefault(r["domain"], [0, 0, 0])
    d[0] += r["gold_in"]["ON"]; d[1] += r["gold_in"]["OFF"]; d[2] += 1
summary = {"n": N, "gold_in_ON": g_on / N, "gold_in_OFF": g_off / N,
           "delta": (g_on - g_off) / N, "net_tasks": g_on - g_off,
           "ON_new_over_embed16": new_over_e16,
           "by_domain": {k: {"ON": v[0], "OFF": v[1], "n": v[2]} for k, v in by_dom.items()}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", "keyword_arm.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
