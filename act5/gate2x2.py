#!/usr/bin/env python3
"""#19c gate 2x2 smoke: is EXPECT polarity driven by PROMPT or by MEMORY?
Known cells: active-prompt+adapter=NO, active-prompt+base=NO, passive-prompt+base=YES.
This runs the missing cell (passive-prompt + adapter) plus a same-run re-check of
passive-prompt+base, turn-1 EXPECT only (no tool loop). 8W+4U dir smoke probes."""
import json, os, re, sys

import torch
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
unwritten = make_facts(72, 1234 + 77001)[48:72]
def attr_of(f): return f["statement"][len("The user's "):f["statement"].find(" is ")]

SYS_PASSIVE = """You are an assistant with a memory notebook about one user: a timestamped log of facts. Facts never written are NOT in the log. You have no other memory of this user.
TOOLS you may call, one per line, exactly:
GREP: <word or phrase> -- search the log for that text, returns matching lines.
EMBED -- semantic search: returns the 4 log entries most similar to the current question. Use it when GREP fails or you are unsure what to search.

PROTOCOL for every question:
- The FIRST line of your first reply must be exactly "EXPECT: YES" (you believe this fact is in the log) or "EXPECT: NO" (you believe it was never written). Judge before any tool.
- Then either call ONE tool (one line), or finish immediately.
- After each TOOL RESULT you may call another tool or finish.
- To finish: output "ANSWER: <exact value>" if the log gives it, or "UNKNOWN" if the fact is not in the log. Values look like short code words (e.g. ZedKimWol).
- Never invent a value that you cannot ground in the log."""

def gen1(q, use_adapter):
    msgs = [{"role": "system", "content": SYS_PASSIVE}, {"role": "user", "content": q}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    inp = tok(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        if use_adapter:
            g = model.generate(**inp, max_new_tokens=800, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        else:
            with model.disable_adapter():
                g = model.generate(**inp, max_new_tokens=800, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
    out = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    think, vis = ("", out) if "</think>" not in out else out.split("</think>", 1)
    m = re.search(r"EXPECT:\s*(YES|NO)", vis, re.I)
    return (m.group(1).upper() if m else None), think.replace("<think>", "").strip()[:600]

probes = [(f, False) for f in facts[:8]] + [(f, True) for f in unwritten[:4]]
rows = []
for f, un in probes:
    q = f"What is the user's {attr_of(f)}?"
    ea, ta = gen1(q, True)    # missing cell: passive prompt + adapter
    eb, tb = gen1(q, False)   # re-check: passive prompt + base
    rows.append({"fid": f["fid"], "unwritten": un, "exp_adapter": ea, "exp_base": eb,
                 "think_adapter": ta, "think_base": tb})
    print(f"[2x2] {'U' if un else 'W'}{f['fid']:2d} passiveprompt+adapter={ea} +base={eb}", flush=True)

def yr(rs, k): return sum(1 for r in rs if r[k] == "YES") / max(len(rs), 1)
W = [r for r in rows if not r["unwritten"]]; U = [r for r in rows if r["unwritten"]]
summary = {"passiveprompt_adapter_yes_W": yr(W, "exp_adapter"), "passiveprompt_adapter_yes_U": yr(U, "exp_adapter"),
           "passiveprompt_base_yes_W": yr(W, "exp_base"), "passiveprompt_base_yes_U": yr(U, "exp_base")}
odir = os.path.join(os.path.dirname(__file__), "results", "e2e")
json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, "gate2x2_smoke.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
