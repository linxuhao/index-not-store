#!/usr/bin/env python3
"""chain50 (pre-reg 2026-08-13): RS-arm attribution controls.
ARM=chatter : fresh LoRA per task; write 41 pre + 10 post chatter turns (Note,
              1 step each, chain44 stream order); NO gold writes. Statement
              probe with adapter. Register without the fact.
ARM=frozen  : zero writes; statement probe from the frozen base.
Both: embed statement -> top-6 over canonical store (gold ALWAYS in the log)
-> frozen-base cold answer (for later judging). Anchors: RS_STMT 0.576/0.448,
Q_EMB 0.384/0.312. Lines (margin 0.04): CHATTER ret >=0.536 kills the write's
residue; CHATTER <=0.464 AND FROZEN <=0.424 vindicates it. SANITY: chatter
instance flag ~0 (fact never written) else VOID.
INMIND_SMOKE=1 -> 8 tasks. Out results/inmind/rs_ctrl_{arm}_{tag}.json"""
import json, os, re, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
ARM = os.environ.get("ARM", "chatter")
assert ARM in ("chatter", "frozen"), ARM
tag = "smoke" if SMOKE else "full"
TOPK = 6
N_PRE = 41
N_POST = 10

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
if SMOKE:
    tasks = tasks[:8]
bg = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "evaluation",
                                               "background", "lme_s_background.jsonl"))]
BG = [b["content"] for b in bg if b["role"] == "user"]
INJECT_AT = 41

STOP = set("about would could there their which where thing things really something "
           "someone always never these those after before other every".split())

def bridge_words(t):
    ws = set(re.findall(r"[a-zA-Z]{5,}", (t.get("entity_1") or "") + " " + t["user_message"]))
    return {w for w in ws if w.lower() not in STOP}

def stem_hit(text, words):
    low = (text or "").lower()
    return any(w.lower()[:5] in low for w in words)

BG_WRITTEN = BG[:N_PRE] + BG[INJECT_AT:INJECT_AT + N_POST]
BG_WORDS = set()
for c in BG_WRITTEN:
    BG_WORDS |= {w.lower() for w in re.findall(r"[a-zA-Z]{5,}", c)} - STOP

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B",
                                            torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
INIT = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()

@torch.no_grad()
def embed(ts, bs=64):
    out = []
    for i in range(0, len(ts), bs):
        b = etok(ts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(DEV)
        out.append(torch.nn.functional.normalize(emb(**b).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out)

BGV = embed(BG)

PERSONA = ("You are the user's personal assistant. You may remember things about the user "
           "from past conversations. If you remember, answer directly from memory; if you do "
           "not remember, say exactly 'I don't remember.' Do not guess.")
REFUSAL_PAT = re.compile(r"don't have access|do not have access|no access to|don't store|"
                         r"privacy|personal information")
RECALL_Q = ("The user just asked: {q}\n\n"
            "Before answering, write your best one-sentence guess about a personal fact "
            "of this user (from your memory of past conversations) that changes how this "
            "question should be answered. Write it as a specific fact about the user, "
            "not a topic label. Guess even if unsure.\n"
            "My best guess about the user:")
RAG = ("Here are entries from the user's memory log:\n{log}\n\n{q}\n\n"
       "Answer the user helpfully and concisely, taking any relevant personal facts "
       "from the log into account.")

def reset_adapter():
    sd = model.state_dict()
    for k, v in INIT.items():
        sd[k].copy_(v.to(sd[k].device))

def write_step(opt, form):
    inp = tok(form, return_tensors="pt").to(DEV)
    labels = inp["input_ids"].clone()
    labels[:, :max(2, labels.shape[1] // 3)] = -100
    out = model(**inp, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad()

@torch.no_grad()
def gen_chat(msgs, max_new=60, adapter=True):
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    if adapter:
        g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    else:
        with model.disable_adapter():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def flags(s, t):
    b = bridge_words(t)
    binst = stem_hit(s, b)
    low = (s or "").lower()
    bgw = BG_WORDS - {w.lower() for w in b}
    return {"instance": int(binst),
            "refused": int(bool(REFUSAL_PAT.search(low)) or "don't remember" in low),
            "bg_leak": int((not binst) and any(w in low for w in bgw))}

rows = []
for i, t in enumerate(tasks):
    use_adapter = ARM == "chatter"
    if use_adapter:
        reset_adapter()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        model.train()
        for c in BG[:N_PRE]:                       # stream: pre-injection chatter
            write_step(opt, "Note: " + c)
        for c in BG[INJECT_AT:INJECT_AT + N_POST]:  # stream: post-gold chatter (no gold)
            write_step(opt, "Note: " + c)
        model.eval()

    gold_idx = INJECT_AT
    gold_ln = f"[turn {INJECT_AT:03d}+] {t['user_message']}"
    store = [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[:INJECT_AT])] + [gold_ln] + \
            [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[INJECT_AT:], start=INJECT_AT + 1)]
    storev = torch.cat([BGV[:INJECT_AT], embed([t["user_message"]]), BGV[INJECT_AT:]])

    stmt = gen_chat([{"role": "system", "content": PERSONA},
                     {"role": "user", "content": RECALL_Q.format(q=t["query"])}],
                    max_new=48, adapter=use_adapter)
    qv = embed([stmt or t["query"]])
    order = torch.argsort((storev @ qv.T).squeeze(1), descending=True).tolist()
    rank = order.index(gold_idx) + 1
    ret = [store[j] for j in order[:TOPK]]
    ans = gen_chat([{"role": "user",
                     "content": RAG.format(log="\n".join(ret), q=t["query"])}],
                   max_new=300, adapter=False)   # serve cold, always frozen
    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "stmt": stmt[:200], "stmt_rank": rank,
                 "stmt_gold_in": int(rank <= TOPK), "stmt_flags": flags(stmt, t),
                 "ret": ret, "ans": ans})
    print(f"[rsctrl:{ARM}] {i:3d} rank={rank} in={int(rank <= TOPK)} "
          f"inst={rows[-1]['stmt_flags']['instance']} "
          f"ref={rows[-1]['stmt_flags']['refused']}", flush=True)

n = len(rows)
summary = {"arm": ARM, "n": n,
           "stmt_gold_in": sum(r["stmt_gold_in"] for r in rows) / n,
           "median_rank": sorted(r["stmt_rank"] for r in rows)[n // 2],
           "instance": sum(r["stmt_flags"]["instance"] for r in rows) / n,
           "refused": sum(r["stmt_flags"]["refused"] for r in rows) / n,
           "bg_leak": sum(r["stmt_flags"]["bg_leak"] for r in rows) / n,
           "anchors": {"rs_stmt_ret": 0.576, "rs_stmt_ia": 0.448,
                       "q_emb_ret": 0.384, "q_emb_ia": 0.312,
                       "kill_write_if_chatter_ret_ge": 0.536,
                       "vindicate_if_chatter_le_and_frozen_le": [0.464, 0.424]}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"rs_ctrl_{ARM}_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
