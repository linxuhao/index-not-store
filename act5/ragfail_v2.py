#!/usr/bin/env python3
"""RAG-fail v2 (CLAIMS #17, rebuilt per v1 autopsy). Reader = Qwen3.5-9B + pure-replay adapter.
F1 indirect: FULL paraphrase (adj+noun), verified zero content-word overlap with the log line.
   Arms: term-grep+base / embed-grep+base / active-keys+adapter. Retrieval-hit logged.
F2 ambiguity, MATCHED context: same 4 lines (gold + 3 stale never-written distractors) to all
   arms; vary READ side only: base / adapter / adapter+Δlp-annotations (the self-test tool).
T3 absence, MATCHED deceptive context: same near-miss lines to all arms; base vs adapter vs
   adapter+emit-note. Abstain measured."""
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

NSYN = {"color": "shade", "city": "town", "dish": "meal", "drink": "beverage", "movie": "film",
        "book": "novel", "river": "waterway", "gadget": "device", "snack": "treat",
        "song": "tune", "park": "garden", "shoe": "footwear", "plant": "flora",
        "tool": "implement", "street": "avenue", "car": "automobile", "phone": "handset",
        "chair": "seat", "coat": "jacket", "clock": "timepiece", "boat": "vessel",
        "pet": "creature", "lamp": "lantern", "hobby": "pastime"}
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
unwritten = make_facts(72, 1234 + 77001)[48:72]
def attr_of(f): return f["statement"][len("The user's "):f["statement"].find(" is ")]
LOG = [f"[2026-07-{(f['fid'] % 27) + 1:02d}] {f['statement']}" for f in facts]
LOGV = embed(LOG)
syl = ["zor", "vex", "lun", "qua", "mip", "tar", "nye", "blu", "gro", "fen", "wix", "dap",
       "sol", "kee", "ral", "tun", "vop", "jiz", "mol", "pez", "fyx", "gub", "hox", "lid"]

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
def dlp(prompt, ans):
    def lp(use):
        full = tok(prompt + " " + ans + ".", return_tensors="pt").to(DEV)
        npre = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        if use:
            lg = model(**full).logits[0]
        else:
            with model.disable_adapter():
                lg = model(**full).logits[0]
        s = torch.log_softmax(lg[:-1].float(), -1)
        ids = full["input_ids"][0]
        return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())
    return lp(True) - lp(False)

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
rows = []

# ---- F1: full paraphrase, verified zero overlap
f1 = []
for f in facts:
    adj, noun = attr_of(f).rsplit(" ", 1)
    if adj in ASYN and noun in NSYN:
        q = f"What is the user's {ASYN[adj]} {NSYN[noun]}?"
        STOP = {"user's", "user", "what", "the"}
        qw = {w for w in q.lower().replace("?", "").replace("-", " ").split() if len(w) > 3} - STOP
        lw = {w.strip(".").lower() for w in LOG[f["fid"]].split() if len(w.strip(".")) > 3} - STOP
        if not (qw & lw):
            f1.append((f, q))
f1 = f1[:24]
for f, q in f1:
    gold = f["answer"]
    words = [w for w in q.lower().replace("?", "").split() if len(w) > 3]
    tg = [ln for _, ln in sorted([(sum(w in ln.lower() for w in words), ln) for ln in LOG],
                                 key=lambda x: -x[0]) if _ > 0][:4] if any(
        sum(w in ln.lower() for w in words) for ln in LOG) else []
    qv = embed([q]); eg = [LOG[i] for i in torch.argsort((LOGV @ qv.T).squeeze(1), descending=True)[:4].tolist()]
    ks = keys_for(q)
    ag = [ln for ln in LOG if any(L.contains_match_ci(ln.rsplit(" is ", 1)[-1].rstrip("."), c) for c in ks)][:6]
    a_t = chat(RT.format(log="\n".join(tg) or "(no matches)", q=q), False)
    a_e = chat(RT.format(log="\n".join(eg) or "(no matches)", q=q), False)
    a_a = chat(RT.format(log="\n".join(ag) or "(no matches)", q=q), True)
    rows.append({"t": "F1", "fid": f["fid"],
                 "hit_t": int(any(gold in ln for ln in tg)), "hit_e": int(any(gold in ln for ln in eg)),
                 "hit_a": int(any(gold in ln for ln in ag)),
                 "acc_t": int(L.contains_match_ci(gold, a_t)), "acc_e": int(L.contains_match_ci(gold, a_e)),
                 "acc_a": int(L.contains_match_ci(gold, a_a))})
    print(f"[F1] f{f['fid']:2d} hit={rows[-1]['hit_t']}{rows[-1]['hit_e']}{rows[-1]['hit_a']} "
          f"acc={rows[-1]['acc_t']}{rows[-1]['acc_e']}{rows[-1]['acc_a']}", flush=True)

if os.environ.get("F1_ONLY"):
    F1r = [r for r in rows if r["t"] == "F1"]
    summary = {"F1_n": len(F1r),
        "F1_hit": {k: sum(r["hit_"+k] for r in F1r)/max(len(F1r),1) for k in ("t","e","a")},
        "F1_acc": {k: sum(r["acc_"+k] for r in F1r)/max(len(F1r),1) for k in ("t","e","a")}}
    odir = os.path.join(os.path.dirname(__file__), "results", "ragfail")
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, "v2_9b_f1.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1)); sys.exit(0)

# ---- F2: ambiguity, matched context, read-side arms
import random as R
for f in facts[:24]:
    gold = f["answer"]; attr = attr_of(f)
    r = R.Random(555 + f["fid"])
    seen = {x["answer"] for x in facts}
    ds = []
    while len(ds) < 3:
        v = "".join(s.capitalize() for s in r.sample(syl, 3))
        if v not in seen: seen.add(v); ds.append(v)
    lines = [f"[2026-07-{d:02d}] The user's {attr} is {v}." for d, v in zip((3, 9, 15), ds)]
    lines.append(f"[2026-07-21] The user's {attr} is {gold}.")
    r.shuffle(lines)
    ctx = "\n".join(lines)
    q = f"What is the user's {attr}? Multiple log entries conflict; give the value that is actually true."
    a_b = chat(RT.format(log=ctx, q=q), False)
    a_m = chat(RT.format(log=ctx, q=q), True)
    ann_lines = []
    cloze_prefix = "The user's " + attr + " is"
    for ln in lines:
        v = ln.rsplit(" is ", 1)[-1].rstrip(".")
        score = dlp(cloze_prefix, v)
        ann_lines.append(ln + f"  [self-familiarity: {score:.1f}]")
    ann = "\n".join(ann_lines)
    a_n = chat(RT.format(log=ann, q=q), True)
    rows.append({"t": "F2", "fid": f["fid"],
                 "acc_b": int(L.contains_match_ci(gold, a_b)), "acc_m": int(L.contains_match_ci(gold, a_m)),
                 "acc_n": int(L.contains_match_ci(gold, a_n))})
    print(f"[F2] f{f['fid']:2d} base={rows[-1]['acc_b']} adpt={rows[-1]['acc_m']} annot={rows[-1]['acc_n']}", flush=True)

# ---- T3v2: absence, matched deceptive context
for f in unwritten:
    attr = attr_of(f)
    words = [w for w in attr.lower().split() if len(w) > 3]
    dec = [ln for _, ln in sorted([(sum(w in ln.lower() for w in words), ln) for ln in LOG],
                                  key=lambda x: -x[0])][:4]
    ctx = "\n".join(dec)
    q = f"What is the user's {attr}?"
    a_b = chat(RT.format(log=ctx, q=q), False)
    a_m = chat(RT.format(log=ctx, q=q), True)
    rows.append({"t": "T3", "fid": f["fid"],
                 "ab_b": int("UNKNOWN" in a_b.upper()), "ab_m": int("UNKNOWN" in a_m.upper())})
    print(f"[T3] f{f['fid']:2d} abstain base={rows[-1]['ab_b']} adpt={rows[-1]['ab_m']}", flush=True)

F1r = [r for r in rows if r["t"] == "F1"]; F2r = [r for r in rows if r["t"] == "F2"]
T3r = [r for r in rows if r["t"] == "T3"]
summary = {
    "F1_n": len(F1r),
    "F1_hit": {k: sum(r["hit_" + k] for r in F1r) / max(len(F1r), 1) for k in ("t", "e", "a")},
    "F1_acc": {k: sum(r["acc_" + k] for r in F1r) / max(len(F1r), 1) for k in ("t", "e", "a")},
    "F2_n": len(F2r),
    "F2_acc": {k: sum(r["acc_" + k] for r in F2r) / max(len(F2r), 1) for k in ("b", "m", "n")},
    "T3_n": len(T3r),
    "T3_abstain": {k: sum(r["ab_" + k] for r in T3r) / max(len(T3r), 1) for k in ("b", "m")}}
odir = os.path.join(os.path.dirname(__file__), "results", "ragfail")
os.makedirs(odir, exist_ok=True)
json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, "v2_9b.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
