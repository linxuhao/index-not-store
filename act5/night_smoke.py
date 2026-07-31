#!/usr/bin/env python3
"""#20 SMOKE (CLAIMS): day+night on the NEW substrate, single adapter, and the
implicit->explicit diversity hypothesis. Day: ws1 writes (1 step/fact, write policy
lr) of 24 facts -> snapshot. Nights (2): equal budget arms from the same snapshot —
A verbatim (write-form x4 steps/fact/night), B diverse (4 UNSEEN paraphrase templates
per fact per night, never repeated across nights), C no-sleep. Probes per arm:
write-form recall, q-form recall (EXPLICIT primary), 2-AFC dlp write+q form (implicit,
predict B~A), EXPECT verbal gate w/ thinking (24W+12U; sharpest prediction: B creates
W-YES where C has none). Deterministic templates (no LLM paraphraser at smoke)."""
import copy, json, os, re, sys

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
LR = 3e-5
NIGHTS = 2
STEPS_PER_FACT_NIGHT = 4

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.eval()

facts = make_facts(24, 777)
unwritten = make_facts(48, 777 + 55)[24:36]
def attr_of(f): return f["statement"][len("The user's "):f["statement"].find(" is ")]

TPL = ["The user's {a} is {v}.",                          # write form (day + arm A)
       "{v} is the user's {a}.",
       "When asked about their {a}, the user said {v}.",
       "The user mentioned that their {a} is {v}.",
       "Note on the user's {a}: {v}.",
       "For the user, the {a} goes by the name {v}.",
       "User {a}: {v}.",
       "Their {a}? That would be {v}.",
       "In the log, the {a} of the user appears as {v}."]

def train_step(text, val):
    b = tok(text, return_tensors="pt").to(DEV)
    pos = text.find(val)
    npre = tok(text[:pos], return_tensors="pt")["input_ids"].shape[1] if pos > 0 else 0
    labels = b["input_ids"].clone(); labels[:, :npre] = -100  # value-token loss, policy-matched
    model.train()
    out = model(**b, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainables(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()

def trainables():
    return [p for p in model.parameters() if p.requires_grad]

# ---- DAY: load adapter produced by phase5_accum itself (policy-faithful: 1step+missreplay) ----
DAY_ADAPTER = os.environ["DAY_ADAPTER"]
sd0 = torch.load(os.path.join(os.path.dirname(__file__), DAY_ADAPTER), map_location=DEV)
model.load_state_dict(sd0, strict=False)
print(f"[night] DAY adapter loaded: {DAY_ADAPTER}", flush=True)
day_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k}
opt = torch.optim.AdamW(trainables(), lr=LR)

@torch.no_grad()
def dlp(prefix, val):
    def lp(use):
        full = tok(prefix + " " + val + ".", return_tensors="pt").to(DEV)
        npre = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
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
def greedy(prefix, max_new=10):
    inp = tok(prefix, return_tensors="pt").to(DEV)
    g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

GATE_SYS = ("You are an assistant with a personal memory system about one user: facts were "
            "previously written into your weights. You may feel you know an answer; impressions "
            "can be vague or wrong.\nFor the question below, reply with EXACTLY one line: "
            "\"EXPECT: YES\" if you believe this fact is in your memory, or \"EXPECT: NO\" "
            "if you believe it was never written. Judge from your impressions only.")

@torch.no_grad()
def expect_gate(q):
    msgs = [{"role": "system", "content": GATE_SYS}, {"role": "user", "content": q}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    inp = tok(p, return_tensors="pt").to(DEV)
    g = model.generate(**inp, max_new_tokens=500, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
    out = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    vis = out.split("</think>", 1)[-1]
    m = re.search(r"EXPECT:\s*(YES|NO)", vis, re.I)
    return m.group(1).upper() if m else None

def probe(arm):
    rows = []
    others = [f["answer"] for f in facts]
    for i, f in enumerate(facts):
        a, gold = attr_of(f), f["answer"]
        wcloze = f["statement"][:f["statement"].find(" is ") + 3]
        qcloze = f"Q: What is the user's {a}?\nA: The user's {a} is"
        dist = others[(i + 7) % len(others)]
        rows.append({"fid": f["fid"],
                     "recall_w": int(L.contains_match_ci(gold, greedy(wcloze))),
                     "recall_q": int(L.contains_match_ci(gold, greedy(qcloze))),
                     "afc_w": int(dlp(wcloze, gold) > dlp(wcloze, dist)),
                     "afc_q": int(dlp(qcloze, gold) > dlp(qcloze, dist)),
                     "expect": expect_gate(f"What is the user's {a}?")})
    urows = [{"fid": u["fid"], "expect": expect_gate(f"What is the user's {attr_of(u)}?")}
             for u in unwritten]
    n = len(rows)
    s = {"recall_w": sum(r["recall_w"] for r in rows) / n,
         "recall_q": sum(r["recall_q"] for r in rows) / n,
         "afc_w": sum(r["afc_w"] for r in rows) / n,
         "afc_q": sum(r["afc_q"] for r in rows) / n,
         "expect_yes_W": sum(1 for r in rows if r["expect"] == "YES") / n,
         "expect_yes_U": sum(1 for r in urows if r["expect"] == "YES") / len(urows)}
    print(f"[night] arm {arm}: {json.dumps({k: round(v, 3) for k, v in s.items()})}", flush=True)
    return s, rows, urows

def night_texts(arm, night):
    for f in facts:
        a, v = attr_of(f), f["answer"]
        if arm == "A":
            for _ in range(STEPS_PER_FACT_NIGHT):
                yield TPL[0].format(a=a, v=v), v
        else:
            for t in range(STEPS_PER_FACT_NIGHT):
                yield TPL[1 + night * STEPS_PER_FACT_NIGHT + t].format(a=a, v=v), v

results = {}
for arm in ("C", "A", "B"):
    sd = model.state_dict()
    sd.update({k: v.clone() for k, v in day_state.items()})
    model.load_state_dict(sd, strict=False)
    opt = torch.optim.AdamW(trainables(), lr=LR)
    if arm != "C":
        for night in range(NIGHTS):
            print(f"[night] arm {arm} night {night}...", flush=True)
            for text, val in night_texts(arm, night):
                train_step(text, val)
    s, rows, urows = probe(arm)
    results[arm] = {"summary": s, "rows": rows, "unwritten": urows}

odir = os.path.join(os.path.dirname(__file__), "results", "night")
os.makedirs(odir, exist_ok=True)
json.dump(results, open(os.path.join(odir, "night_smoke3.json"), "w"), indent=1)
print(json.dumps({a: results[a]["summary"] for a in results}, indent=1))
