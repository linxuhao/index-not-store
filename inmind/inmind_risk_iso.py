#!/usr/bin/env python3
"""#23 Amendment 5: risk-route isolation (chain27). Completes the A3c 2x2.
  RISK_BB  BASE risk-terms -> grep+embed(<=6) -> BASE reader
  RISK_AB  adapter risk-terms (REUSED from answers_full) -> same retrieval -> BASE reader
Base drafts reused from answers_full. Out results/inmind/answers_risk.json"""
import json, os, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
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
RISK = ("A user asked: {q}\n\nA draft answer:\n{draft}\n\nList up to 5 kinds of "
        "personal attributes, conditions, or constraints a user might have that would "
        "make this answer wrong, risky, or in need of adjustment. Reply with short "
        "keywords only, one per line.")

def risk_retrieve(terms, draft):
    hits = [ln for ln in LOG if any(len(w) > 3 and w in ln.lower() for w in terms)][:4]
    return list(dict.fromkeys(hits + etop(" ".join(terms) or draft, 4)))[:6]

rows = []
for i, t in enumerate(tasks):
    q = t["query"]
    fr = frow[t["task_id"]]
    draft = fr["draft"]
    risks_b = chat(RISK.format(q=q, draft=draft), max_new=80)
    terms_b = [w.strip("-• ").lower() for w in risks_b.split("\n") if w.strip()][:5]
    r_bb = risk_retrieve(terms_b, draft)
    a_bb = chat(RAG.format(log="\n".join(r_bb), q=q))
    terms_a = fr["risk_terms"]
    r_ab = risk_retrieve(terms_a, draft)
    a_ab = chat(RAG.format(log="\n".join(r_ab), q=q))
    gold_ln = LOG[i]
    row = {"task_id": t["task_id"], "domain": t["domain"],
           "risk_terms_base": terms_b, "risk_terms_adapter": terms_a,
           "arms": {"RISK_BB": {"ans": a_bb, "ret": r_bb},
                    "RISK_AB": {"ans": a_ab, "ret": r_ab}},
           "gold_in": {"RISK_BB": int(any(gold_ln == r for r in r_bb)),
                       "RISK_AB": int(any(gold_ln == r for r in r_ab))}}
    rows.append(row)
    print(f"[risk-iso] {i:3d} gold_in BB={row['gold_in']['RISK_BB']} "
          f"AB={row['gold_in']['RISK_AB']}", flush=True)

summary = {"n": N, "target_recall": {k: sum(r["gold_in"][k] for r in rows) / N
                                     for k in rows[0]["gold_in"]}}
odir = os.path.join(HERE, "results", "inmind")
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(odir, "answers_risk.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
