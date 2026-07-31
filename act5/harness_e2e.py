#!/usr/bin/env python3
"""E2E memory-harness smoke (CLAIMS queue 16). Read pipeline on the validated ws1+missreplay
substrate. Arms: 1 no-memory / 2 index-only / 3 passive RAG / 4 active (adapter reads) /
4b active (base reads). 48 written QA + 12 unwritten fabrication probes."""
import argparse, json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L
_argv = sys.argv; sys.argv = ["phase5_accum.py"]
from phase5_accum import make_facts, question_form
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--adapter", default="results/ws1replay/adapters/accum_ewcreplay_l300_Pmiss_n48_pe2_s1234.pt")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--topk", type=int, default=30)
ap.add_argument("--dev", default="cuda:1")
ap.add_argument("--dtype", choices=["fp32","bf16"], default="fp32")
ap.add_argument("--out-tag", default="s1234")
args = ap.parse_args()
DEV = args.dev
L.check_env()

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.dtype=="bf16" else torch.float32).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), args.adapter),
                                 map_location=DEV), strict=False)
model.eval()

facts = make_facts(48, args.seed)
unwritten = make_facts(60, args.seed + 77001)[48:60]  # 12 never-written probes
for f in unwritten:
    f["unwritten"] = True

def attr_of(f):
    return f["statement"][len("The user's "):f["statement"].find(" is ")]

LOG = [f"[2026-07-{(f['fid'] % 27) + 1:02d}] {f['statement']}" for f in facts]

def chat(content, use_adapter, max_new=24):
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

QTMPL = ("{q} Answer with only the value. If you do not know, answer exactly UNKNOWN.")
RTMPL = ("Here are excerpts from your own memory log:\n{log}\n\n{q} Answer with only the "
         "value. If the log does not contain the answer, answer exactly UNKNOWN.")

@torch.no_grad()
def keys_for(f):
    qp = question_form(f)
    ids = tok(qp, return_tensors="pt")["input_ids"][0].to(DEV)
    logits = model(input_ids=ids[None]).logits[0, -1]
    cands = []
    for t in torch.argsort(logits, descending=True)[: args.topk].tolist():
        seq = torch.cat([ids, torch.tensor([t], device=DEV)])
        outp = [t]
        for _ in range(8):
            nxt = int(model(input_ids=seq[None]).logits[0, -1].argmax())
            if nxt == tok.eos_token_id:
                break
            seq = torch.cat([seq, torch.tensor([nxt], device=DEV)])
            outp.append(nxt)
            d = tok.decode([nxt])
            if "." in d or "\n" in d:
                break
        c = tok.decode(outp, skip_special_tokens=True).strip().rstrip(".")
        if c and c not in cands:
            cands.append(c)
    return cands

def grep(keys, attr):
    hits = [ln for ln in LOG if attr in ln]
    for ln in LOG:
        if ln in hits:
            continue
        val = ln.rsplit(" is ", 1)[-1].rstrip(".")
        if any(L.contains_match_ci(val, c) for c in keys):
            hits.append(ln)
    return hits[:12]

rows = []
for f in facts + unwritten:
    gold = f["answer"]
    un = f.get("unwritten", False)
    q = question_form(f).replace("Q: ", "").replace("\nA:", "")
    a1 = chat(QTMPL.format(q=q), use_adapter=False)
    a2 = chat(QTMPL.format(q=q), use_adapter=True)
    attr = attr_of(f)
    r3 = [ln for ln in LOG if attr in ln][:12]
    a3 = chat(RTMPL.format(log="\n".join(r3) or "(no matches)", q=q), use_adapter=False)
    ks = keys_for(f)
    r4 = grep(ks, attr)
    a4 = chat(RTMPL.format(log="\n".join(r4) or "(no matches)", q=q), use_adapter=True)
    a4b = chat(RTMPL.format(log="\n".join(r4) or "(no matches)", q=q), use_adapter=False)
    row = {"fid": f["fid"], "unwritten": un, "n_retrieved": len(r4),
           "gold_in_r4": any(gold in ln for ln in r4)}
    for k, a in (("a1", a1), ("a2", a2), ("a3", a3), ("a4", a4), ("a4b", a4b)):
        row[k] = a[:40]
        row[k + "_hit"] = int(L.contains_match_ci(gold, a))
        row[k + "_abstain"] = int("UNKNOWN" in a.upper())
    rows.append(row)
    print(f"[e2e] {'U' if un else 'W'}{f['fid']:2d} r4={len(r4)} "
          + " ".join(f"{k}={row[k+'_hit']}/{row[k+'_abstain']}" for k in ("a1","a2","a3","a4","a4b")),
          flush=True)

W = [r for r in rows if not r["unwritten"]]
U = [r for r in rows if r["unwritten"]]
summary = {"adapter": os.path.basename(args.adapter), "topk": args.topk,
           "acc_written": {k: sum(r[k + "_hit"] for r in W) / len(W)
                          for k in ("a1", "a2", "a3", "a4", "a4b")},
           "fabricate_unwritten": {k: sum(1 - r[k + "_abstain"] for r in U) / len(U)
                                   for k in ("a1", "a2", "a3", "a4", "a4b")},
           "gold_in_r4": sum(r["gold_in_r4"] for r in W) / len(W)}
odir = os.path.join(os.path.dirname(__file__), "results", "e2e")
os.makedirs(odir, exist_ok=True)
json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, f"e2e_{args.out_tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
