#!/usr/bin/env python3
"""#23 Amendment 6: token-tier probe (chain28). Teacher-force the base draft under
adapter-on vs adapter-off; measure whether bridge-entity tokens are elevated in the
next-token distribution anywhere in the draft (sub-emission memory-guided drafting).
Null control: same metric with 8 other-task entity sets per task.
Out results/inmind/token_probe.json"""
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
full = json.load(open(os.path.join(HERE, "results", "inmind", "answers_full.json")))
frow = {r["task_id"]: r for r in full["rows"]}

STOP = set("about would could there their which where thing things really something "
           "someone always never these those after before other every".split())

def content_words(t):
    ws = set(re.findall(r"[a-zA-Z]{5,}", (t.get("entity_1") or "") + " " + t["user_message"]))
    return {w for w in ws if w.lower() not in STOP}

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(HERE, "results", "inmind",
                                 "adapter_full.pt"), map_location=DEV), strict=False)
model.eval()

def token_ids_for(words):
    ids = set()
    for w in words:
        for v in (w, " " + w, w.lower(), " " + w.lower(), w.capitalize(), " " + w.capitalize()):
            enc = tok(v, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(enc[0])
    return sorted(ids)

ANSWER = "{q}\n\nAnswer the user helpfully and concisely."
ELICIT = ("Question: {q}\n\nBefore answering, what single keyword about the user's "
          "personal situation should be checked in the memory log? Keyword:")

@torch.no_grad()
def elicit_probe(t, ids):
    p = tok.apply_chat_template([{"role": "user", "content": ELICIT.format(q=t["query"])}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    out = {}
    for name, use in (("on", True), ("off", False)):
        if use:
            lg = model(**inp).logits[0, -1].float()
            g = model.generate(**inp, max_new_tokens=10, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        else:
            with model.disable_adapter():
                lg = model(**inp).logits[0, -1].float()
                g = model.generate(**inp, max_new_tokens=10, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
        lp = torch.log_softmax(lg, -1)
        top10 = torch.topk(lp, 10).indices.tolist()
        out[name] = {"bridge_lp_max": float(lp[ids].max()) if ids else None,
                     "bridge_in_top10": bool(set(ids) & set(top10)),
                     "top10_tokens": [tok.decode([i]) for i in top10],
                     "gen": tok.decode(g[0][inp["input_ids"].shape[1]:],
                                       skip_special_tokens=True)}
    return out

@torch.no_grad()
def draft_logits(t):
    q = t["query"]
    p = tok.apply_chat_template([{"role": "user", "content": ANSWER.format(q=q)}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    draft = frow[t["task_id"]]["draft"]
    full_txt = p + draft
    inp = tok(full_txt, return_tensors="pt").to(DEV)
    npre = tok(p, return_tensors="pt")["input_ids"].shape[1]
    lg_on = model(**inp).logits[0].float()
    with model.disable_adapter():
        lg_off = model(**inp).logits[0].float()
    lp_on = torch.log_softmax(lg_on[npre - 1:-1], -1)
    lp_off = torch.log_softmax(lg_off[npre - 1:-1], -1)
    return lp_on, lp_off  # [n_draft_positions, vocab]

def probe(lp_on, lp_off, ids):
    if not ids:
        return None
    on = lp_on[:, ids]
    off = lp_off[:, ids]
    dmax = float((on - off).max())
    k = 10
    top_on = torch.topk(lp_on, k, dim=-1).indices
    top_off = torch.topk(lp_off, k, dim=-1).indices
    idt = torch.tensor(ids, device=lp_on.device)
    in_on = (top_on.unsqueeze(-1) == idt).any(-1).any(-1)
    in_off = (top_off.unsqueeze(-1) == idt).any(-1).any(-1)
    entries = int((in_on & ~in_off).sum())
    return dmax, entries

rows = []
NULLS = 8
for i, t in enumerate(tasks):
    words = content_words(t)
    draft_words = set(re.findall(r"[a-zA-Z]{5,}", frow[t["task_id"]]["draft"].lower()))
    if {w.lower() for w in words} & draft_words:
        rows.append({"task_id": t["task_id"], "excluded": "bridge-word-in-draft"})
        print(f"[tokprobe] {i:3d} EXCLUDED", flush=True)
        continue
    ids_own = token_ids_for(words)
    lp_on, lp_off = draft_logits(t)
    own = probe(lp_on, lp_off, ids_own)
    if own is None:
        rows.append({"task_id": t["task_id"], "excluded": "no-bridge-token-ids"})
        print(f"[tokprobe] {i:3d} EXCLUDED no-ids", flush=True)
        continue
    elic = elicit_probe(t, ids_own)
    nulls = []
    for j in range(NULLS):
        ot = tasks[(i + 13 * (j + 1)) % len(tasks)]
        if ot["task_id"] == t["task_id"]:
            continue
        r = probe(lp_on, lp_off, token_ids_for(content_words(ot)))
        if r:
            nulls.append(r)
    rows.append({"task_id": t["task_id"], "own_dmax": own[0], "own_top10_entries": own[1],
                 "null_dmax": [n[0] for n in nulls],
                 "null_top10_entries": [n[1] for n in nulls],
                 "elicit": elic})
    print(f"[tokprobe] {i:3d} own_dmax={own[0]:.3f} entries={own[1]} "
          f"null_med={sorted(n[0] for n in nulls)[len(nulls)//2]:.3f}", flush=True)

json.dump(rows, open(os.path.join(HERE, "results", "inmind", "token_probe.json"), "w"),
          indent=1)
ok = [r for r in rows if "own_dmax" in r]
import statistics
own_med = statistics.median(r["own_dmax"] for r in ok)
null_all = [v for r in ok for v in r["null_dmax"]]
null_med = statistics.median(null_all)
null_95 = sorted(null_all)[int(0.95 * len(null_all))]
el_on = sum(1 for r in ok if r["elicit"]["on"]["bridge_in_top10"])
el_off = sum(1 for r in ok if r["elicit"]["off"]["bridge_in_top10"])
el_d = statistics.median((r["elicit"]["on"]["bridge_lp_max"] or -99) -
                         (r["elicit"]["off"]["bridge_lp_max"] or -99) for r in ok)
print(json.dumps({"n": len(ok), "own_dmax_median": round(own_med, 4),
                  "null_dmax_median": round(null_med, 4),
                  "null_dmax_95pct": round(null_95, 4),
                  "own_entries_total": sum(r["own_top10_entries"] for r in ok),
                  "null_entries_mean_per_set": round(sum(v for r in ok for v in
                    r["null_top10_entries"]) / max(sum(len(r["null_top10_entries"])
                    for r in ok), 1), 3),
                  "elicit_bridge_in_top10_on": el_on,
                  "elicit_bridge_in_top10_off": el_off,
                  "elicit_bridge_dlp_median": round(el_d, 4)}, indent=1))
