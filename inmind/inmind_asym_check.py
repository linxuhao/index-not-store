#!/usr/bin/env python3
"""chain49: asymmetric-encoder query control. Query arm re-scored with BGE query
instruction prefix; store unchanged. Out results/inmind/asym_check.json"""
import json, os, sys
import torch
from transformers import AutoModel, AutoTokenizer
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
PREFIX = "Represent this sentence for searching relevant passages: "
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
res = {"plain": {4: 0, 6: 0}, "prefixed": {4: 0, 6: 0}}
for t in tasks:
    storev = torch.cat([BGV[:INJECT_AT], embed([t["user_message"]]), BGV[INJECT_AT:]])
    for arm, q in (("plain", t["query"]), ("prefixed", PREFIX + t["query"])):
        qv = embed([q])
        order = torch.argsort((storev @ qv.T).squeeze(1), descending=True).tolist()
        rank = order.index(INJECT_AT) + 1
        for k in (4, 6):
            res[arm][k] += int(rank <= k)
summary = {a: {f"gold_in@{k}": v / len(tasks) for k, v in d.items()} for a, d in res.items()}
json.dump(summary, open(os.path.join(HERE, "results", "inmind", "asym_check.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
