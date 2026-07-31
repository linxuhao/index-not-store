#!/usr/bin/env python3
"""F6 v3 (CLAIMS #17): presented-mode binding trigger. Lesson from v1/v2 (5th
appearance): binding lives ONLY in presented-mode comparison. Instrument:
bind(v|q) = [dlp(v | question slot) - dlp(v | neutral slot)] / ntok over the FULL
log-value inventory (48). Neutral-slot subtraction cancels content authentication
(equal for all written values); the residual is question-binding. No generation,
no write-form cloze, no gold leak. Trigger = peak statistic (z, top1-top2 margin)
of the bind vector: written -> concentrated peak, unwritten -> flat.
Recovery = argmax bind == gold, esp. on pass-1 (keys+grep, identical to v2) misses.
Positive control: direct-form questions (transparent reference) must separate
strongly or the instrument is broken. F6_SMOKE=1 -> 8 ref + 4 unwritten + 6 direct."""
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
SMOKE = os.environ.get("F6_SMOKE") == "1"

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

facts = make_facts(48, 1234)
unwritten = make_facts(72, 1234 + 77001)[48:72]
def attr_of(f): return f["statement"][len("The user's "):f["statement"].find(" is ")]
LOG = [f"[2026-07-{(f['fid'] % 27) + 1:02d}] {f['statement']}" for f in facts]
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
    ntok = len(ids) - npre
    return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(ntok, 1)

def dlp_norm(prefix, val):
    return lp_value(prefix, val, True) - lp_value(prefix, val, False)

@torch.no_grad()
def keys_for(q, k=30):
    ids = tok(f"Q: {q}\nA:", return_tensors="pt")["input_ids"][0].to(DEV)
    logits = model(input_ids=ids[None]).logits[0, -1]
    cands = []
    for t in torch.argsort(logits, descending=True)[:k].tolist():
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

def qref(f):
    adj, noun = attr_of(f).rsplit(" ", 1)
    if noun in DESC and adj in ASYN:
        return f"You keep a {ASYN[adj]} one of these: {DESC[noun]}. Which is it -- give its name."
    return None

def qdirect(f):
    return f"What is the user's {attr_of(f)}?"

print("[F6v3] caching neutral-slot dlp for 48 values...", flush=True)
NEU = {v: dlp_norm(NEUTRAL, v) for v in VALUES}
print(f"[F6v3] neutral dlp: mean={sum(NEU.values())/len(NEU):.3f} "
      f"min={min(NEU.values()):.3f} max={max(NEU.values()):.3f}", flush=True)

def bind_vector(q):
    pre = f"Q: {q}\nA:"
    return [dlp_norm(pre, v) - NEU[v] for v in VALUES]

def auc(pos, neg):
    if not pos or not neg: return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))

ref_w = [f for f in facts if qref(f)]
ref_u = [f for f in unwritten if qref(f)]
if SMOKE:
    probes = ([("ref", f, False) for f in ref_w[:8]] + [("ref", f, True) for f in ref_u[:4]]
              + [("dir", f, False) for f in facts[:6]] + [("dir", f, True) for f in unwritten[:3]])
else:
    probes = ([("ref", f, False) for f in ref_w] + [("ref", f, True) for f in ref_u]
              + [("dir", f, False) for f in facts] + [("dir", f, True) for f in unwritten])

rows = []
for form, f, un in probes:
    q = qref(f) if form == "ref" else qdirect(f)
    gold = f["answer"]
    ks = keys_for(q)
    r1 = [ln for ln in LOG if any(L.contains_match_ci(ln.rsplit(" is ", 1)[-1].rstrip("."), c) for c in ks)][:6]
    hit1 = int(any(gold in ln for ln in r1))
    bv = bind_vector(q)
    order = sorted(range(len(bv)), key=lambda i: -bv[i])
    top1, top2 = bv[order[0]], bv[order[1]]
    rest = [bv[i] for i in order[1:]]
    mu = sum(rest) / len(rest)
    sd = (sum((x - mu) ** 2 for x in rest) / len(rest)) ** 0.5
    z = (top1 - mu) / max(sd, 1e-6)
    margin = top1 - top2
    grank = (order.index(VALUES.index(gold)) + 1) if not un else None
    sel = int(not un and VALUES[order[0]] == gold)
    rows.append({"fid": f["fid"], "form": form, "unwritten": un, "hit1": hit1,
                 "sel": sel, "grank": grank, "z": round(z, 3), "margin": round(margin, 4),
                 "top3": [VALUES[i] for i in order[:3]], "gold": gold,
                 "bv": [round(x, 4) for x in bv]})
    print(f"[F6v3] {form} {'U' if un else 'W'}{f['fid']:2d} hit1={hit1} sel={sel} "
          f"grank={grank} z={z:5.2f} margin={margin:6.3f} top1={VALUES[order[0]][:20]}", flush=True)

import numpy as np
BV = np.array([r["bv"] for r in rows])
nrow = len(rows)
LOO = BV - (BV.mean(0, keepdims=True) * nrow - BV) / (nrow - 1)
for i, r in enumerate(rows):
    oc = np.argsort(-LOO[i])
    r["z_c"] = round(float((LOO[i][oc[0]] - LOO[i][oc[1:]].mean()) / max(LOO[i][oc[1:]].std(), 1e-9)), 3)
    r["margin_c"] = round(float(LOO[i][oc[0]] - LOO[i][oc[1]]), 4)
    if not r["unwritten"]:
        g = VALUES.index(r["gold"])
        r["grank_c"] = int(np.where(oc == g)[0][0]) + 1
        r["sel_c"] = int(r["grank_c"] == 1)
        r["hit4_c"] = int(r["grank_c"] <= 4)
        r["pw_c"] = round(float((LOO[i][g] > np.delete(LOO[i], g)).mean()), 3)

summary = {"smoke": SMOKE}
for form in ("dir", "ref"):
    W = [r for r in rows if r["form"] == form and not r["unwritten"]]
    U = [r for r in rows if r["form"] == form and r["unwritten"]]
    m1 = [r for r in W if not r["hit1"]]
    def med(k, rs): return sorted(r[k] for r in rs)[len(rs) // 2] if rs else None
    def avg(k, rs): return round(sum(r[k] for r in rs) / len(rs), 3) if rs else None
    summary[form] = {
        "n_written": len(W), "n_unwritten": len(U), "n_pass1_miss": len(m1),
        "pass1_hit": avg("hit1", W),
        "raw": {"sel_acc": avg("sel", W), "grank_median": med("grank", W),
                "auc_z": auc([r["z"] for r in W], [r["z"] for r in U]),
                "auc_margin": auc([r["margin"] for r in W], [r["margin"] for r in U])},
        "cent": {"sel_acc": avg("sel_c", W), "grank_median": med("grank_c", W),
                 "hit4": avg("hit4_c", W), "pairwise": avg("pw_c", W),
                 "sel_on_miss": avg("sel_c", m1), "hit4_on_miss": avg("hit4_c", m1),
                 "grank_median_miss": med("grank_c", m1),
                 "auc_z": auc([r["z_c"] for r in W], [r["z_c"] for r in U]),
                 "auc_z_miss": auc([r["z_c"] for r in m1], [r["z_c"] for r in U]) if m1 else None,
                 "auc_margin": auc([r["margin_c"] for r in W], [r["margin_c"] for r in U])},
    }
odir = os.path.join(os.path.dirname(__file__), "results", "ragfail")
tag = "f6v3_smoke" if SMOKE else "f6v3_9b"
json.dump({"summary": summary, "rows": rows}, open(os.path.join(odir, f"{tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
