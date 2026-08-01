#!/usr/bin/env python3
"""chain35 (pre-reg 2026-08-01): CANONICAL-PROTOCOL frozen-harness arms.
Per the benchmark's protocol.md: memory state per task = 486 LME-s background user
turns + the ONE injected target fact (injected after the 41st cumulative user turn).
Frozen arms only (no in-weight writes — in-weight claims stay in the accumulated
setting by design):
  C_A1   embed(question) top-4 over the 487-line store -> base answer
  C_FZ   base draft (reused from answers_full) -> embed(draft) top-16 -> base answer
gold_in per arm. (C_A5 dropped: 487-line prompt OOMs the 24GB card; full-store
stuffing is not one of the benchmark's own baselines.) INMIND_SMOKE=1 -> 10 tasks. Out results/inmind/canonical_{tag}.json"""
import json, os, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
HERE = os.path.dirname(__file__)
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
tag = "smoke" if SMOKE else "full"

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
if SMOKE:
    tasks = tasks[:10]
bg = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "evaluation",
                                               "background", "lme_s_background.jsonl"))]
BG = [b["content"] for b in bg if b["role"] == "user"]
INJECT_AT = 41  # after the 41st cumulative background user turn (protocol.md)
print(f"[canon] background user turns: {len(BG)}; inject at {INJECT_AT}", flush=True)

full = json.load(open(os.path.join(HERE, "results", "inmind", "answers_full.json")))
frow = {r["task_id"]: r for r in full["rows"]}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
model.eval()
etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()

@torch.no_grad()
def embed(texts, bs=64):
    vs = []
    for i in range(0, len(texts), bs):
        b = etok(texts[i:i + bs], padding=True, truncation=True,
                 return_tensors="pt").to(DEV)
        vs.append(torch.nn.functional.normalize(emb(**b).last_hidden_state[:, 0], dim=-1))
    return torch.cat(vs)

BGV = embed(BG)  # cached background embeddings, shared across tasks

def chat(content, max_new=300):
    msgs = [{"role": "user", "content": content}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

RAG = ("Here are entries from the user's memory log:\n{log}\n\n{q}\n\n"
       "Answer the user helpfully and concisely, taking any relevant personal facts "
       "from the log into account.")

rows = []
for i, t in enumerate(tasks):
    gold_ln = f"[turn {INJECT_AT:03d}+] {t['user_message']}"
    store = [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[:INJECT_AT])] + [gold_ln] + \
            [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[INJECT_AT:], start=INJECT_AT + 1)]
    gv = embed([t["user_message"]])
    storev = torch.cat([BGV[:INJECT_AT], gv, BGV[INJECT_AT:]])
    q = t["query"]
    draft = frow[t["task_id"]]["draft"]

    def etop(text, k):
        qv = embed([text])
        idx = torch.argsort((storev @ qv.T).squeeze(1), descending=True)[:k].tolist()
        return [store[j] for j in idx]

    r_a1 = etop(q, 4)
    a_a1 = chat(RAG.format(log="\n".join(r_a1), q=q))
    r_fz = etop(draft, 16)
    a_fz = chat(RAG.format(log="\n".join(r_fz), q=q))
    row = {"task_id": t["task_id"], "domain": t["domain"],
           "arms": {"C_A1": {"ans": a_a1, "ret": r_a1},
                    "C_FZ": {"ans": a_fz, "ret": r_fz}},
           "gold_in": {"C_A1": int(gold_ln in r_a1), "C_FZ": int(gold_ln in r_fz)}}
    rows.append(row)
    print(f"[canon] {i:3d} A1={row['gold_in']['C_A1']} FZ={row['gold_in']['C_FZ']}",
          flush=True)

summary = {"n": len(rows),
           "gold_in": {k: sum(r["gold_in"][k] for r in rows) / len(rows)
                       for k in ["C_A1", "C_FZ"]}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"canonical_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
