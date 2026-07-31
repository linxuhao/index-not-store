#!/usr/bin/env python3
"""#19a post-hoc binding gate (CLAIMS): can F6 v3.1's centered presented-mode bind,
applied to the VALUE THE AGENT ANSWERED at the question slot, separate correct answers
from wrong/fabricated ones in #18's rows? If yes -> a cheap final-check closes the ref
fabrication hole (0.235) without a new agent loop. No new behavior; pure re-measurement
of e2e/agent_9b.json answers."""
import json, os, sys

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L
_argv = sys.argv; sys.argv = ["phase5_accum.py"]
from phase5_accum import make_facts
sys.argv = _argv

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
ADAPTER = "results/e2e9b/adapters/accum_replay_Pmiss_MQwen359B_Dbf16_n48_pe6_s1234.pt"

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), ADAPTER),
                                 map_location=DEV), strict=False)
model.eval()

facts = make_facts(48, 1234)
VALUES = [f["answer"] for f in facts]
NEUTRAL = "Q: Tell me something.\nA:"

@torch.no_grad()
def lp_value(prefix, val, use_adapter):
    full = tok(prefix + " " + val + ".", return_tensors="pt").to(DEV)
    npre = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
    if use_adapter:
        lg = model(**full).logits[0]
    else:
        with model.disable_adapter():
            lg = model(**full).logits[0]
    s = torch.log_softmax(lg[:-1].float(), -1)
    ids = full["input_ids"][0]
    return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(len(ids) - npre, 1)

def dlp_norm(prefix, val):
    return lp_value(prefix, val, True) - lp_value(prefix, val, False)

f6 = json.load(open(os.path.join(os.path.dirname(__file__), "results", "ragfail", "f6v3_9b.json")))
COLMEAN = np.array([r["bv"] for r in f6["rows"]]).mean(0)
print("[gate] caching neutral dlp...", flush=True)
NEU = {v: dlp_norm(NEUTRAL, v) for v in VALUES}

agent = json.load(open(os.path.join(os.path.dirname(__file__), "results", "e2e", "agent_9b.json")))
rows = []
for r in agent["rows"]:
    if r["final"] in ("UNKNOWN", "TURN_EXHAUST") or r["final"] is None:
        continue
    v = next((x for x in VALUES if L.contains_match_ci(x, r["final"])), None)
    if v is None:
        rows.append({"fid": r["fid"], "form": r["form"], "cls": "off-inventory",
                     "final": r["final"][:24], "rank": None, "bind": None}); continue
    pre = f"Q: {r['q']}\nA:"
    bv = np.array([dlp_norm(pre, x) - NEU[x] for x in VALUES]) - COLMEAN
    vi = VALUES.index(v)
    rank = int(np.argsort(-bv).tolist().index(vi)) + 1
    cls = ("fabricated" if r["unwritten"] else ("correct" if r["acc"] else "wrong"))
    rows.append({"fid": r["fid"], "form": r["form"], "cls": cls, "final": v,
                 "rank": rank, "bind": round(float(bv[vi]), 4),
                 "z": round(float((bv[vi] - bv.mean()) / max(bv.std(), 1e-9)), 3)})
    print(f"[gate] {r['form']} {'U' if r['unwritten'] else 'W'}{r['fid']:2d} cls={cls:10s} "
          f"rank={rank:2d} bind={bv[vi]:+.3f}", flush=True)

def auc(pos, neg):
    if not pos or not neg: return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

ok = [r for r in rows if r["cls"] == "correct"]
bad = [r for r in rows if r["cls"] in ("wrong", "fabricated")]
summary = {"n_correct": len(ok), "n_wrong_or_fab": len(bad),
           "n_off_inventory": sum(1 for r in rows if r["cls"] == "off-inventory"),
           "auc_bind": auc([r["bind"] for r in ok], [r["bind"] for r in bad]),
           "auc_negrank": auc([-r["rank"] for r in ok], [-r["rank"] for r in bad]),
           "rank_med_correct": sorted(r["rank"] for r in ok)[len(ok) // 2] if ok else None,
           "rank_med_bad": sorted(r["rank"] for r in bad)[len(bad) // 2] if bad else None,
           "gate_rank1": {"keep_correct": sum(1 for r in ok if r["rank"] == 1) / max(len(ok), 1),
                          "block_bad": sum(1 for r in bad if r["rank"] != 1) / max(len(bad), 1)},
           "gate_rank4": {"keep_correct": sum(1 for r in ok if r["rank"] <= 4) / max(len(ok), 1),
                          "block_bad": sum(1 for r in bad if r["rank"] > 4) / max(len(bad), 1)}}
odir = os.path.join(os.path.dirname(__file__), "results", "e2e")
json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, "bindgate.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
