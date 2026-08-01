#!/usr/bin/env python3
"""C_KW / C_FAN (pre-reg 2026-08-01): retrieval-layer test of keyword & token-fan
queries on the CANONICAL store (486 background turns + 1 gold). CPU-only (bge-small),
reuses stored base keywords (keyword_arm.json kw_off) and base fan word-sets
(tokenfan_arm.json words_off). gold_in@{4,6,16} per query type, vs C_FZ/C_A1 from
the running chain35. Out results/inmind/canon_kwfan.json"""
import json, os, torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
bg = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "evaluation",
                                               "background", "lme_s_background.jsonl"))]
BG = [b["content"] for b in bg if b["role"] == "user"]
INJECT_AT = 41
kw = {r["task_id"]: r["kw_off"] for r in
      json.load(open(os.path.join(HERE, "results", "inmind", "keyword_arm.json")))["rows"]}
fan = {r["task_id"]: r["words_off"] for r in
       json.load(open(os.path.join(HERE, "results", "inmind", "tokenfan_arm.json")))["rows"]}

etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

@torch.no_grad()
def embed(ts, bs=32):
    out = []
    for i in range(0, len(ts), bs):
        b = etok(ts[i:i + bs], padding=True, truncation=True, return_tensors="pt")
        out.append(torch.nn.functional.normalize(emb(**b).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out)

BGV = embed(BG)
print(f"[kwfan] background embedded: {len(BG)}", flush=True)

rows = []
for i, t in enumerate(tasks):
    gv = embed([t["user_message"]])
    storev = torch.cat([BGV[:INJECT_AT], gv, BGV[INJECT_AT:]])
    gold_idx = INJECT_AT

    def rank_of(qv_scores):
        order = torch.argsort(qv_scores, descending=True).tolist()
        return order.index(gold_idx) + 1

    k_q = kw[t["task_id"]] or t["query"]
    kv = embed([k_q])
    r_kw = rank_of((storev @ kv.T).squeeze(1))
    words = fan[t["task_id"]]
    if words:
        wv = embed(words)
        r_fan = rank_of((storev @ wv.T).max(dim=1).values)
    else:
        r_fan = 488
    rows.append({"task_id": t["task_id"], "kw": k_q[:30], "rank_kw": r_kw,
                 "rank_fan": r_fan, "n_fan_words": len(words)})
    if i % 25 == 0:
        print(f"[kwfan] {i}: kw_rank={r_kw} fan_rank={r_fan}", flush=True)

def gi(key, k):
    return sum(1 for r in rows if r[key] <= k) / len(rows)

summary = {"n": len(rows),
           "C_KW": {f"gold_in@{k}": round(gi("rank_kw", k), 3) for k in (4, 6, 16)},
           "C_FAN": {f"gold_in@{k}": round(gi("rank_fan", k), 3) for k in (4, 6, 16)},
           "median_rank": {"kw": sorted(r["rank_kw"] for r in rows)[62],
                           "fan": sorted(r["rank_fan"] for r in rows)[62]}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", "canon_kwfan.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
