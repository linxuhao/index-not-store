#!/usr/bin/env python3
"""#23 Amendment 7: recall-tier elicitation (chain29). Fresh LoRA; write a 24-task
subset in TWO forms (note + QA on naive_query) round-robin until bare recall of
naive_query contains a bridge word; then probe the INDIRECT query's elicitation
position top-10 on/off. INMIND_SMOKE=1 -> 6 tasks. Out results/inmind/recall_probe.json"""
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
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
NSEL = 6 if SMOKE else 24
MAX_ROUNDS = 12

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
full = json.load(open(os.path.join(HERE, "results", "inmind", "answers_full.json"))
                 )
frow = {r["task_id"]: r for r in full["rows"]}

STOP = set("about would could there their which where thing things really something "
           "someone always never these those after before other every".split())

def bridge_words(t):
    ws = set(re.findall(r"[a-zA-Z]{5,}", (t.get("entity_1") or "") + " " + t["user_message"]))
    return {w for w in ws if w.lower() not in STOP}

sel = []
for t in tasks:
    ws = bridge_words(t)
    if not ws:
        continue
    draft_words = set(re.findall(r"[a-zA-Z]{5,}", frow[t["task_id"]]["draft"].lower()))
    if {w.lower() for w in ws} & draft_words:
        continue
    sel.append(t)
    if len(sel) >= NSEL:
        break
print(f"[recall-probe] selected {len(sel)} tasks", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)

def forms(t):
    return [f"Note: {t['user_message']}",
            f"Q: {t['naive_query']}\nA: {t['user_message']}"]

def write_step(text):
    inp = tok(text, return_tensors="pt").to(DEV)
    labels = inp["input_ids"].clone()
    npre = max(2, labels.shape[1] // 3)
    labels[:, :npre] = -100
    out = model(**inp, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad()
    return float(out.loss)

@torch.no_grad()
def bare_recall(t):
    model.eval()
    p = tok.apply_chat_template([{"role": "user", "content": t["naive_query"]}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    g = model.generate(**inp, max_new_tokens=60, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
    ans = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    model.train()
    return any(w.lower() in ans for w in bridge_words(t)), ans

# round-robin write to recall criterion
passed = {t["task_id"]: False for t in sel}
steps = {t["task_id"]: 0 for t in sel}
for rnd in range(MAX_ROUNDS):
    todo = [t for t in sel if not passed[t["task_id"]]]
    if not todo:
        break
    for t in todo:
        for f in forms(t):
            write_step(f)
        steps[t["task_id"]] += 2
        ok, _ = bare_recall(t)
        passed[t["task_id"]] = ok
    print(f"[recall-probe] round {rnd}: passed {sum(passed.values())}/{len(sel)}", flush=True)

model.eval()

def token_ids_for(words):
    ids = set()
    for w in words:
        for v in (w, " " + w, w.lower(), " " + w.lower(), w.capitalize(), " " + w.capitalize()):
            enc = tok(v, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(enc[0])
    return sorted(ids)

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
        out[name] = {"bridge_lp_max": float(lp[ids].max()),
                     "bridge_in_top10": bool(set(ids) & set(top10)),
                     "top10_tokens": [tok.decode([i]) for i in top10],
                     "gen": tok.decode(g[0][inp["input_ids"].shape[1]:],
                                       skip_special_tokens=True)}
    return out

THINK = os.environ.get("INMIND_THINK") == "1"

@torch.no_grad()
def think_probe(t, use_adapter):
    p = tok.apply_chat_template([{"role": "user", "content": t["query"]}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    inp = tok(p, return_tensors="pt").to(DEV)
    if use_adapter:
        g = model.generate(**inp, max_new_tokens=1024, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    else:
        with model.disable_adapter():
            g = model.generate(**inp, max_new_tokens=1024, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    txt = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    low = txt.lower()
    hits = [w for w in bridge_words(t) if w.lower() in low]
    pos = min((low.find(w.lower()) for w in hits), default=-1)
    return {"bridge_hits": hits, "first_pos": pos, "len": len(txt), "text": txt[:400]}

rows = []
for t in sel:
    # refresh-then-probe (pre-registered full-run fix): give the trace its best shot
    model.train()
    for f in forms(t):
        write_step(f)
    steps[t["task_id"]] += 2
    model.eval()
    alive, ans = bare_recall(t)
    ids = token_ids_for(bridge_words(t))
    el = elicit_probe(t, ids)
    row = {"task_id": t["task_id"], "steps": steps[t["task_id"]],
           "recall_pass": passed[t["task_id"]], "recall_alive_at_probe": alive,
           "recall_answer": ans[:120], "elicit": el}
    if THINK:
        row["think_on"] = think_probe(t, True)
        row["think_off"] = think_probe(t, False)
    rows.append(row)
    print(f"[recall-probe] task {t['task_id']:3d} steps={steps[t['task_id']]} "
          f"alive={alive} on_top10={el['on']['bridge_in_top10']} "
          f"off_top10={el['off']['bridge_in_top10']}"
          + (f" think_on={len(row['think_on']['bridge_hits'])>0} "
             f"think_off={len(row['think_off']['bridge_hits'])>0}" if THINK else ""),
          flush=True)

alive_rows = [r for r in rows if r["recall_alive_at_probe"]]
n_on = sum(1 for r in alive_rows if r["elicit"]["on"]["bridge_in_top10"])
n_off = sum(1 for r in alive_rows if r["elicit"]["off"]["bridge_in_top10"])
summary = {"n_selected": len(sel), "n_recall_alive": len(alive_rows),
           "bridge_in_top10_on": n_on, "bridge_in_top10_off": n_off,
           "delta_fraction": round((n_on - n_off) / max(len(alive_rows), 1), 3)}
if THINK:
    summary["think_bridge_on"] = sum(1 for r in alive_rows if r["think_on"]["bridge_hits"])
    summary["think_bridge_off"] = sum(1 for r in alive_rows if r["think_off"]["bridge_hits"])
# phase 2: certificate-tier adapter_full thinking arms (7b comparison)
if THINK and not SMOKE:
    model.load_state_dict(torch.load(os.path.join(HERE, "results", "inmind",
                                     "adapter_full.pt"), map_location=DEV), strict=False)
    model.eval()
    cert = []
    for t in sel:
        con = think_probe(t, True)
        coff = think_probe(t, False)
        cert.append({"task_id": t["task_id"], "think_on": con, "think_off": coff})
        print(f"[recall-probe] cert task {t['task_id']:3d} "
              f"on={len(con['bridge_hits'])>0} off={len(coff['bridge_hits'])>0}", flush=True)
    summary["cert_think_bridge_on"] = sum(1 for c in cert if c["think_on"]["bridge_hits"])
    summary["cert_think_bridge_off"] = sum(1 for c in cert if c["think_off"]["bridge_hits"])
else:
    cert = []
tag = "smoke" if SMOKE else "full"
json.dump({"summary": summary, "rows": rows, "cert_think": cert},
          open(os.path.join(HERE, "results", "inmind", f"recall_probe_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
