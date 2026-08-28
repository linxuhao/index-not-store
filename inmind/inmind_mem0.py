#!/usr/bin/env python3
"""chain52 (pre-reg 2026-08-25): Mem0 (open-source mem0ai) under the inmind protocol.
ARM=van   : hits = mem0.search(task["query"])
ARM=instr : hits = mem0.search(QS[task_id])  (results/inmind/qemb_questions.json)
Per task: fresh user_id, 241-line store IN STREAM ORDER = BG[:41] + [task
user_message] + BG[41:240], each line mem0.add'ed as a user message with mem0's
default inference pipeline ON. Answer: frozen Qwen3.5-9B via localhost:8001
(vLLM, no-think template), temperature 0, max_tokens 300, canonical RAG prompt
over the top-6 returned memory texts. Gold heuristic: bridge_words/stem_hit
copied from inmind_rs_ctrl.py. INMIND_SMOKE=1 -> 8 tasks; TASK_LIMIT=n -> n.
Out: results/inmind/mem0_{ARM}_{tag}.json"""
import importlib.metadata
import json
import os
import re
import shutil
import time

os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

import requests
from mem0 import Memory

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
ARM = os.environ.get("ARM", "van")
assert ARM in ("van", "instr"), ARM
tag = "smoke" if SMOKE else "full"
TOPK = 6
INJECT_AT = 41
VLLM = "http://localhost:8001/v1"
MODEL = "Qwen/Qwen3.5-9B"
MEM0_VERSION = importlib.metadata.version("mem0ai")

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
if SMOKE:
    tasks = tasks[:8]
if os.environ.get("TASK_LIMIT"):
    tasks = tasks[:int(os.environ["TASK_LIMIT"])]
bg = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "evaluation",
                                               "background", "lme_s_background.jsonl"))]
BG = [b["content"] for b in bg if b["role"] == "user"]
QS = {int(k): v for k, v in json.load(open(os.path.join(
    HERE, "results", "inmind", "qemb_questions.json"))).items()}

STOP = set("about would could there their which where thing things really something "
           "someone always never these those after before other every".split())

def bridge_words(t):
    ws = set(re.findall(r"[a-zA-Z]{5,}", (t.get("entity_1") or "") + " " + t["user_message"]))
    return {w for w in ws if w.lower() not in STOP}

def stem_hit(text, words):
    low = (text or "").lower()
    return any(w.lower()[:5] in low for w in words)

QDRANT_PATH = os.path.join(HERE, "results", "inmind", f"mem0_qdrant_{ARM}_{tag}")
if os.path.exists(QDRANT_PATH):
    shutil.rmtree(QDRANT_PATH)
CONFIG = {
    "llm": {"provider": "openai",
            "config": {"model": MODEL, "temperature": 0.0, "max_tokens": 2000,
                       "api_key": "dummy", "openai_base_url": VLLM}},
    "embedder": {"provider": "huggingface",
                 "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384}},
    "vector_store": {"provider": "qdrant",
                     "config": {"collection_name": "inmind_mem0",
                                "embedding_model_dims": 384,
                                "path": QDRANT_PATH, "on_disk": True}},
}
mem = Memory.from_config(CONFIG)

RAG = ("Here are entries from the user's memory log:\n{memories}\n\n{query}\n\n"
       "Answer the user helpfully and concisely, taking any relevant personal facts "
       "from the log into account.")

def answer(prompt):
    r = requests.post(VLLM + "/chat/completions", json={
        "model": MODEL, "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

rows = []
total_add_fail = 0
for i, t in enumerate(tasks):
    uid = f"t{t['task_id']}"
    store = BG[:INJECT_AT] + [t["user_message"]] + BG[INJECT_AT:240]
    assert len(store) == 241, len(store)
    add_fail = 0
    t0 = time.time()
    for line in store:
        for attempt in (0, 1):
            try:
                mem.add(messages=[{"role": "user", "content": line}], user_id=uid)
                break
            except Exception as e:
                if attempt == 1:
                    add_fail += 1
                    print(f"[mem0:{ARM}] {i:3d} ADD FAILED after retry: "
                          f"{type(e).__name__}: {e}", flush=True)
                else:
                    time.sleep(2)
    add_s = time.time() - t0
    total_add_fail += add_fail

    query = t["query"] if ARM == "van" else QS[t["task_id"]]
    t1 = time.time()
    hits = mem.search(query=query, top_k=TOPK, filters={"user_id": uid})
    search_s = time.time() - t1
    results = hits["results"] if isinstance(hits, dict) else hits
    ret = [h.get("memory", "") for h in results]

    b = bridge_words(t)
    gold_in = int(any(stem_hit(m_, b) for m_ in ret))
    allm = mem.get_all(filters={"user_id": uid}, top_k=10000)
    n_store = len(allm["results"] if isinstance(allm, dict) else allm)
    ans = answer(RAG.format(memories="\n".join(ret), query=t["query"]))
    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "search_query": query, "ret": ret, "gold_in_returned": gold_in,
                 "n_store": n_store, "add_fail": add_fail, "ans": ans,
                 "add_s": round(add_s, 2), "search_s": round(search_s, 3)})
    print(f"[mem0:{ARM}] {i:3d} id={t['task_id']} store={n_store} in={gold_in} "
          f"add={add_s:.1f}s search={search_s:.2f}s fail={add_fail}", flush=True)

n = len(rows)
summary = {"arm": ARM, "tag": tag, "n": n,
           "gold_in_returned": sum(r["gold_in_returned"] for r in rows) / n,
           "mean_add_s": sum(r["add_s"] for r in rows) / n,
           "mean_search_s": sum(r["search_s"] for r in rows) / n,
           "mean_n_store": sum(r["n_store"] for r in rows) / n,
           "add_failures": total_add_fail,
           "mem0_version": MEM0_VERSION, "config": CONFIG,
           "backbone": MODEL, "inject_at": INJECT_AT, "topk": TOPK}
out = os.path.join(HERE, "results", "inmind", f"mem0_{ARM}_{tag}.json")
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
