#!/usr/bin/env python3
"""chain34 (pre-reg 2026-08-01): WIDE TOKEN FAN retrieval — the user's original
design. Per task: one forward at the ELICIT position (adapter on/off), top-100
tokens, filter to whole content words, embed EACH, score log lines by max cosine
over the token set, top-6. Readouts: gold_in ON/OFF, instance-word presence in
the token set, overlap vs A3a16. Out results/inmind/tokenfan_arm.json"""
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

STOP = set("""about would could there their which where thing things really something someone
always never these those after before other every should might known level check personal
information situation memory relevant question answer keyword user special specific important
please based certain making doing having getting recent recently history""".split())

def content_tokens(ids, tok):
    out = []
    for i in ids:
        w = tok.decode([i]).strip().lower()
        if re.fullmatch(r"[a-z]{4,}", w) and w not in STOP:
            out.append(w)
    return list(dict.fromkeys(out))

def bridge_words(t):
    ws = set(re.findall(r"[a-zA-Z]{5,}", (t.get("entity_1") or "") + " " + t["user_message"]))
    return {w.lower() for w in ws if w.lower() not in STOP}

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

ELICIT = ("Question: {q}\n\nBefore answering, what single keyword about the user's "
          "personal situation should be checked in the memory log? Keyword:")

@torch.no_grad()
def fan(t, use_adapter):
    p = tok.apply_chat_template([{"role": "user", "content": ELICIT.format(q=t["query"])}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    if use_adapter:
        lg = model(**inp).logits[0, -1].float()
    else:
        with model.disable_adapter():
            lg = model(**inp).logits[0, -1].float()
    top100 = torch.topk(torch.log_softmax(lg, -1), 100).indices.tolist()
    words = content_tokens(top100, tok)
    if not words:
        return words, []
    wv = embed(words)                      # [m, d]
    scores = (LOGV @ wv.T).max(dim=1).values   # per line: max cosine over token set
    order = torch.argsort(scores, descending=True)[:6].tolist()
    return words, [LOG[i] for i in order]

rows = []
for i, t in enumerate(tasks):
    w_on, r_on = fan(t, True)
    w_off, r_off = fan(t, False)
    gold_ln = LOG[i]
    b = bridge_words(t)
    inst_on = any(any(bw[:5] in w for bw in b) for w in w_on)
    inst_off = any(any(bw[:5] in w for bw in b) for w in w_off)
    row = {"task_id": t["task_id"], "domain": t["domain"],
           "n_words_on": len(w_on), "n_words_off": len(w_off),
           "words_on": w_on[:25], "words_off": w_off[:25],
           "instance_in_fan": {"ON": bool(inst_on), "OFF": bool(inst_off)},
           "gold_in": {"ON": int(gold_ln in r_on), "OFF": int(gold_ln in r_off)},
           "ret_on": r_on, "ret_off": r_off}
    rows.append(row)
    print(f"[fan] {i:3d} ON={row['gold_in']['ON']} OFF={row['gold_in']['OFF']} "
          f"inst_on={int(inst_on)} inst_off={int(inst_off)} nw={len(w_on)}/{len(w_off)}",
          flush=True)

e16 = {r["task_id"]: r["gold_in"]["A3a16"] for r in full["rows"]}
g_on = sum(r["gold_in"]["ON"] for r in rows)
g_off = sum(r["gold_in"]["OFF"] for r in rows)
summary = {"n": N, "gold_in_ON": g_on / N, "gold_in_OFF": g_off / N,
           "delta": (g_on - g_off) / N, "net": g_on - g_off,
           "instance_in_fan_ON": sum(r["instance_in_fan"]["ON"] for r in rows),
           "instance_in_fan_OFF": sum(r["instance_in_fan"]["OFF"] for r in rows),
           "ON_new_over_embed16": sum(1 for r in rows if r["gold_in"]["ON"] == 1
                                      and e16[r["task_id"]] == 0)}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", "tokenfan_arm.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
