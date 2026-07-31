#!/usr/bin/env python3
"""#23 InMind judge: applies the benchmark's own judge_application prompt (binary,
two conditions: context_recall AND answer_warning) to answers_{tag}.json via the
local vllm OpenAI-compatible API (qwen3 = Qwen3.6-27B, localhost:8000). Deviation
from paper (GPT-5-mini) noted in ledger. INMIND_SMOKE=1 -> smoke tag."""
import json, os, re, sys, urllib.request

SMOKE = os.environ.get("INMIND_SMOKE") == "1"
tag = os.environ.get("INMIND_TAG") or ("smoke" if SMOKE else "full")
HERE = os.path.dirname(__file__)
API = os.environ.get("INMIND_JUDGE_API", "http://localhost:8000/v1/chat/completions")
JMODEL = os.environ.get("INMIND_JUDGE_MODEL", "qwen3")

tasks = {t["task_id"]: t for t in
         (json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                   "dataset", "inmind.jsonl")))}
JUDGE = open(os.path.join(HERE, "inmind_bench", "evaluation", "prompts",
             "judge_application.txt")).read()
data = json.load(open(os.path.join(HERE, "results", "inmind", f"answers_{tag}.json")))

def call(prompt):
    body = json.dumps({"model": JMODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1600,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    err = "no-parse"
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r'"score"\s*:\s*([01])', out)
            if m:
                # scoring stays FIRST-match for comparability with the full run;
                # store the full output so extraction is verifiable post-hoc
                return int(m.group(1)), out
        except Exception as e:
            err = str(e)[:80]
    return None, err

results = {}
for row in data["rows"]:
    t = tasks[row["task_id"]]
    for arm, a in row["arms"].items():
        if arm == "naive_A1":
            continue
        ctx = "\n".join(a["ret"]) if a["ret"] != ["<all>"] else "\n".join(
            f"[session {i:03d}] {tt['user_message']}" for i, tt in enumerate(tasks.values()))
        prompt = (JUDGE + "\n\n"
                  f"user_message: {t['user_message']}\n"
                  f"context: {ctx}\n"
                  f"query: {t['query']}\n"
                  f"explanation: {t['explanation']}\n"
                  f"answer: {a['ans']}\n")
        s, reason = call(prompt)
        results.setdefault(arm, []).append({"task_id": row["task_id"], "score": s,
                                            "reason": reason})
        print(f"[judge] task {row['task_id']:3d} {arm:6s} -> {s}", flush=True)

summary = {arm: {"n": len(v),
                 "indirect_application": round(sum(r["score"] or 0 for r in v) / len(v), 3),
                 "null": sum(1 for r in v if r["score"] is None)}
           for arm, v in results.items()}
odir = os.path.join(HERE, "results", "inmind")
json.dump({"summary": summary, "results": results},
          open(os.path.join(odir, f"judged_{tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
