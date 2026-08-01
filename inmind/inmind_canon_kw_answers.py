#!/usr/bin/env python3
"""chain36: C_KW end-to-end under canonical protocol — keyword(base) top-6 retrieval
over the 487-line store -> base answer. Out results/inmind/answers_canonkw.json"""
import json, os, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
HERE = os.path.dirname(os.path.abspath(__file__))
tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
bg = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "evaluation",
                                               "background", "lme_s_background.jsonl"))]
BG = [b["content"] for b in bg if b["role"] == "user"]
INJECT_AT = 41
kw = {r["task_id"]: r["kw_off"] for r in
      json.load(open(os.path.join(HERE, "results", "inmind", "keyword_arm.json")))["rows"]}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B",
                                             torch_dtype=torch.bfloat16).to(DEV)
model.eval()
etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()

@torch.no_grad()
def embed(ts, bs=64):
    out = []
    for i in range(0, len(ts), bs):
        b = etok(ts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(DEV)
        out.append(torch.nn.functional.normalize(emb(**b).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out)

BGV = embed(BG)

def chat(content, max_new=300):
    p = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
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
    storev = torch.cat([BGV[:INJECT_AT], embed([t["user_message"]]), BGV[INJECT_AT:]])
    qv = embed([kw[t["task_id"]] or t["query"]])
    idx = torch.argsort((storev @ qv.T).squeeze(1), descending=True)[:6].tolist()
    ret = [store[j] for j in idx]
    ans = chat(RAG.format(log="\n".join(ret), q=t["query"]))
    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "arms": {"C_KW": {"ans": ans, "ret": ret}},
                 "gold_in": {"C_KW": int(gold_ln in ret)}})
    print(f"[ckw] {i:3d} gold_in={rows[-1]['gold_in']['C_KW']}", flush=True)

summary = {"n": len(rows),
           "gold_in": sum(r["gold_in"]["C_KW"] for r in rows) / len(rows)}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", "answers_canonkw.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
