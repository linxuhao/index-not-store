#!/usr/bin/env python3
"""#23 InMind arms (CLAIMS #23 Track-A + Amendment 1). Loads the write-phase adapter
and the same task subset; produces answers per arm:
  A1  naive-RAG: embed(query) top-4 over LOG -> base reader
  A3a answer-side embed: base draft -> embed(draft) top-k (k=4,16) -> revise (adapter on)
  A3b answer-side SCAN: draft -> presented-mode dlp of EVERY log line under draft
      (neutral-subtracted) -> top-4 -> revise
  A3c risk-extraction: adapter model lists risk attributes for the draft -> grep+embed -> revise
  A4  in-context ceiling: gold fact in context (base)
  A5  full-log in context (base)
A2 (interactive agent) runs at full scale only, separate script.
INMIND_SMOKE=1 -> first 15 tasks + adapter_smoke.pt. Out results/inmind/answers_{tag}.json"""
import json, os, re, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
tag = "smoke" if SMOKE else "full"

tasks = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__),
         "inmind_bench", "benchmark", "dataset", "inmind.jsonl"))]
if SMOKE:
    tasks = tasks[:15]
N = len(tasks)
LOG = [f"[session {i:03d}] {t['user_message']}" for i, t in enumerate(tasks)]

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "results",
                                 "inmind", f"adapter_{tag}.pt"), map_location=DEV), strict=False)
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

def chat(content, use_adapter, max_new=300):
    msgs = [{"role": "user", "content": content}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        if use_adapter:
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        else:
            with model.disable_adapter():
                g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def dlp_norm(prefix, val):
    def lp(use):
        full = tok(prefix + " " + val, return_tensors="pt").to(DEV)
        npre = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
        if use:
            lg = model(**full).logits[0]
        else:
            with model.disable_adapter():
                lg = model(**full).logits[0]
        s = torch.log_softmax(lg[:-1].float(), -1)
        ids = full["input_ids"][0]
        return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(len(ids) - npre, 1)
    return lp(True) - lp(False)

NEUTRAL = "Note:"
print("[inmind-arms] caching neutral dlp for log lines...", flush=True)
NEU = [dlp_norm(NEUTRAL, t["user_message"]) for t in tasks]

def scan_against(text, k=4):
    pre = f"Context: {text[:600]}\nRelevant user fact:"
    scores = [dlp_norm(pre, t["user_message"]) - NEU[i] for i, t in enumerate(tasks)]
    order = sorted(range(N), key=lambda i: -scores[i])[:k]
    return [LOG[i] for i in order], [round(scores[i], 4) for i in order]

ANSWER = ("{q}\n\nAnswer the user helpfully and concisely.")
RAG = ("Here are entries from the user's memory log:\n{log}\n\n{q}\n\n"
       "Answer the user helpfully and concisely, taking any relevant personal facts "
       "from the log into account.")
REVISE = ("The user asked: {q}\n\nA draft answer:\n{draft}\n\nEntries from the user's "
          "memory log that may be relevant:\n{log}\n\nRevise the draft into a final "
          "answer. If any log entry is relevant to the user's situation, take it into "
          "account and include any needed warning or adjustment. Reply with the final "
          "answer only.")
RISK = ("A user asked: {q}\n\nA draft answer:\n{draft}\n\nList up to 5 kinds of "
        "personal attributes, conditions, or constraints a user might have that would "
        "make this answer wrong, risky, or in need of adjustment. Reply with short "
        "keywords only, one per line.")

rows = []
for i, t in enumerate(tasks):
    q = t["query"]
    draft = chat(ANSWER.format(q=q), use_adapter=False)
    a1 = chat(RAG.format(log="\n".join(etop(q, 4)), q=q), use_adapter=False)
    r3a4 = etop(draft, 4)
    a3a4 = chat(RAG.format(log="\n".join(r3a4), q=q), use_adapter=True)
    r3a16 = etop(draft, 16)
    a3a16 = chat(RAG.format(log="\n".join(r3a16), q=q), use_adapter=True)
    r3b, sc3b = scan_against(draft, 4)
    a3b = chat(RAG.format(log="\n".join(r3b), q=q), use_adapter=True)
    risks = chat(RISK.format(q=q, draft=draft), use_adapter=True, max_new=80)
    terms = [w.strip("-• ").lower() for w in risks.split("\n") if w.strip()][:5]
    hits = [ln for ln in LOG if any(len(w) > 3 and w in ln.lower() for w in terms)][:4]
    r3c = list(dict.fromkeys(hits + etop(" ".join(terms) or draft, 4)))[:6]
    a3c = chat(RAG.format(log="\n".join(r3c), q=q), use_adapter=True)
    a4 = chat(RAG.format(log=LOG[i], q=q), use_adapter=False)
    a5 = chat(RAG.format(log="\n".join(LOG), q=q), use_adapter=False)
    naive = chat(RAG.format(log="\n".join(etop(t["naive_query"], 4)), q=t["naive_query"]),
                 use_adapter=False)
    gold_ln = LOG[i]
    row = {"task_id": t["task_id"], "domain": t["domain"], "draft": draft,
           "risk_terms": terms,
           "arms": {"A1": {"ans": a1, "ret": etop(q, 4)},
                    "A3a4": {"ans": a3a4, "ret": r3a4},
                    "A3a16": {"ans": a3a16, "ret": r3a16},
                    "A3b": {"ans": a3b, "ret": r3b, "scores": sc3b},
                    "A3c": {"ans": a3c, "ret": r3c},
                    "A4": {"ans": a4, "ret": [gold_ln]},
                    "A5": {"ans": a5, "ret": ["<all>"]},
                    "naive_A1": {"ans": naive, "ret": etop(t["naive_query"], 4)}},
           "gold_in": {k: int(any(gold_ln == r for r in v["ret"])) for k, v in
                       {"A1": {"ret": etop(q, 4)}, "A3a4": {"ret": r3a4},
                        "A3a16": {"ret": r3a16}, "A3b": {"ret": r3b},
                        "A3c": {"ret": r3c}}.items()}}
    rows.append(row)
    print(f"[inmind-arms] {i:3d} gold_in A1={row['gold_in']['A1']} a4={row['gold_in']['A3a4']} "
          f"a16={row['gold_in']['A3a16']} b={row['gold_in']['A3b']} c={row['gold_in']['A3c']}",
          flush=True)

summary = {"n": N, "target_recall": {k: sum(r["gold_in"][k] for r in rows) / N
                                     for k in rows[0]["gold_in"]}}
odir = os.path.join(os.path.dirname(__file__), "results", "inmind")
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(odir, f"answers_{tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
