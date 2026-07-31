#!/usr/bin/env python3
"""F1b referential opacity (CLAIMS #17): query describes the referent via world-knowledge
relation ("the one I fly on off-days" -> weekend kite); adjective also synonymized -> zero
lexical overlap AND no direct synonym; resolving the reference requires inference.
Arms: term-grep+base / embed-grep+base / active-keys+adapter (9B)."""
import json, os, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
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

DESC = {"kite": "the one I fly in the sky", "boat": "the one I sail on water",
        "car": "the one I drive around", "song": "the one I hum along to",
        "book": "the one I read page by page", "dish": "the one I cook and eat",
        "drink": "the one I sip from a glass", "shoe": "the ones I wear on my feet",
        "pet": "the little companion I keep at home", "lamp": "the one that lights my room",
        "clock": "the one that tells me the hour", "phone": "the one I make calls with",
        "coat": "the one I put on when it gets cold", "chair": "the one I sit on",
        "plant": "the green one I water", "tool": "the one I fix things with",
        "movie": "the one I watch on screen", "game": "the one I play for fun",
        "river": "the flowing water I walk along", "park": "the green space where I stroll",
        "snack": "the little bite I eat between meals", "ring": "the one I wear on my finger"}
ASYN = {"favorite": "most-liked", "childhood": "youth-era", "secret": "undisclosed",
        "backup": "fallback", "lucky": "fortune-bringing", "old": "aged", "new": "recent",
        "spare": "extra", "morning": "daybreak", "evening": "dusk", "summer": "hot-season",
        "winter": "cold-season", "weekend": "off-day", "travel": "journey", "study": "learning",
        "work": "job", "home": "household", "early": "first-hours", "late": "last-hours",
        "north": "northern", "south": "southern", "first": "initial", "second": "runner-up",
        "third": "number-three", "best": "top", "worst": "bottom", "main": "primary"}

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), ADAPTER),
                                 map_location=DEV), strict=False)
model.eval()
etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()

@torch.no_grad()
def embed(texts):
    b = etok(texts, padding=True, truncation=True, return_tensors="pt").to(DEV)
    return torch.nn.functional.normalize(emb(**b).last_hidden_state[:, 0], dim=-1)

facts = make_facts(48, 1234)
def attr_of(f): return f["statement"][len("The user's "):f["statement"].find(" is ")]
LOG = [f"[2026-07-{(f['fid'] % 27) + 1:02d}] {f['statement']}" for f in facts]
LOGV = embed(LOG)

def chat(content, use_adapter):
    msgs = [{"role": "user", "content": content}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        if use_adapter:
            g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        else:
            with model.disable_adapter():
                g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def keys_for(q):
    ids = tok(f"Q: {q}\nA:", return_tensors="pt")["input_ids"][0].to(DEV)
    logits = model(input_ids=ids[None]).logits[0, -1]
    cands = []
    for t in torch.argsort(logits, descending=True)[:30].tolist():
        seq = torch.cat([ids, torch.tensor([t], device=DEV)]); outp = [t]
        for _ in range(8):
            nxt = int(model(input_ids=seq[None]).logits[0, -1].argmax())
            if nxt == tok.eos_token_id: break
            seq = torch.cat([seq, torch.tensor([nxt], device=DEV)]); outp.append(nxt)
            d = tok.decode([nxt])
            if "." in d or "\n" in d: break
        c = tok.decode(outp, skip_special_tokens=True).strip().rstrip(".")
        if c and c not in cands: cands.append(c)
    return cands

RT = ("Here are excerpts from your own memory log:\n{log}\n\n{q} Answer with only the value. "
      "If the log does not contain the answer, answer exactly UNKNOWN.")
STOP = {"user's", "user", "what", "which", "name"}
rows = []
todo = []
for f in facts:
    adj, noun = attr_of(f).rsplit(" ", 1)
    if noun in DESC and adj in ASYN:
        q = f"You keep a {ASYN[adj]} one of these: {DESC[noun]}. Which is it -- give its name."
        qw = {w for w in q.lower().replace("?", "").replace("--", " ").replace("-", " ").replace(":", "").replace(".", "").split() if len(w) > 3} - STOP
        lw = {w.strip(".").lower() for w in LOG[f["fid"]].split() if len(w.strip(".")) > 3} - STOP
        if not (qw & lw):
            todo.append((f, q))
todo = todo[:24]
for f, q in todo:
    gold = f["answer"]
    words = [w for w in q.lower().replace("?", "").split() if len(w) > 3]
    scored = sorted([(sum(w in ln.lower() for w in words), ln) for ln in LOG], key=lambda x: -x[0])
    tg = [ln for s, ln in scored if s > 0][:4]
    qv = embed([q]); eg = [LOG[i] for i in torch.argsort((LOGV @ qv.T).squeeze(1), descending=True)[:4].tolist()]
    ks = keys_for(q)
    ag = [ln for ln in LOG if any(L.contains_match_ci(ln.rsplit(" is ", 1)[-1].rstrip("."), c) for c in ks)][:6]
    a_t = chat(RT.format(log="\n".join(tg) or "(no matches)", q=q), False)
    a_e = chat(RT.format(log="\n".join(eg) or "(no matches)", q=q), False)
    a_a = chat(RT.format(log="\n".join(ag) or "(no matches)", q=q), True)
    rows.append({"fid": f["fid"],
                 "hit_t": int(any(gold in ln for ln in tg)), "hit_e": int(any(gold in ln for ln in eg)),
                 "hit_a": int(any(gold in ln for ln in ag)),
                 "acc_t": int(L.contains_match_ci(gold, a_t)), "acc_e": int(L.contains_match_ci(gold, a_e)),
                 "acc_a": int(L.contains_match_ci(gold, a_a))})
    print(f"[F1b] f{f['fid']:2d} hit={rows[-1]['hit_t']}{rows[-1]['hit_e']}{rows[-1]['hit_a']} "
          f"acc={rows[-1]['acc_t']}{rows[-1]['acc_e']}{rows[-1]['acc_a']}", flush=True)

summary = {"n": len(rows),
           "hit": {k: sum(r["hit_" + k] for r in rows) / max(len(rows), 1) for k in ("t", "e", "a")},
           "acc": {k: sum(r["acc_" + k] for r in rows) / max(len(rows), 1) for k in ("t", "e", "a")}}
odir = os.path.join(os.path.dirname(__file__), "results", "ragfail")
json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, "f1b_9b.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
