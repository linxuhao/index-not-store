#!/usr/bin/env python3
"""chain44 (pre-reg 2026-08-03): REAL-STREAM interference arm (RS).
Per task: fresh LoRA; stream-order writes = 41 pre-injection bg user turns
(Note, 1 step each) -> gold to the RECALL criterion (chain43b protocol, equal
dose) -> 10 post-gold bg turns (1 step each). Probes (recall tier): ba/aa hop
cells + v2 stmt; bb/BASE reused from ptrc_b_full.json offline. New readout:
BG_LEAK (hop answer states a background turn instead of the gold).
INMIND_SMOKE=1 -> 8 tasks. Out results/inmind/rs_{tag}.json"""
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
tag = "smoke" if SMOKE else "full"
MAX_ROUNDS = 12
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

# background content words (for BG_LEAK detection), excluding gold overlap per task
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
HOP_Q = ("Here is a question a user just asked their personal assistant: {q}\n\n"
         "The assistant may have been told a personal fact earlier that changes the "
         "answer. Write the single question THE USER would ask the assistant to check "
         "that, in the user's own voice, starting with 'Did I tell you' or 'What did I "
         "say about'. Reply with just that question.\nUser's question:")

def reset_adapter():
    sd = model.state_dict()
    for k, v in INIT.items():
        sd[k].copy_(v.to(sd[k].device))

def chat_qa_text(t):
    return tok.apply_chat_template([{"role": "system", "content": PERSONA},
                                    {"role": "user", "content": t["naive_query"]},
                                    {"role": "assistant", "content": t["user_message"]}],
                                   tokenize=False)

def write_step(opt, form, t=None):
    if form == "CHATQA":
        full_txt = chat_qa_text(t)
        pre = tok.apply_chat_template([{"role": "system", "content": PERSONA},
                                       {"role": "user", "content": t["naive_query"]}],
                                      tokenize=False, add_generation_prompt=True)
        inp = tok(full_txt, return_tensors="pt").to(DEV)
        labels = inp["input_ids"].clone()
        npre = min(tok(pre, return_tensors="pt")["input_ids"].shape[1], labels.shape[1] - 2)
        labels[:, :npre] = -100
    else:
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

def bare_recall(t):
    model.eval()
    ans = gen_chat([{"role": "system", "content": PERSONA},
                    {"role": "user", "content": t["naive_query"]}])
    model.train()
    refused = bool(REFUSAL_PAT.search(ans.lower()))
    return (stem_hit(ans, bridge_words(t)) and not refused), ans

def rank_of(text, t, storev, gold_idx):
    qv = embed([text or t["query"]])
    order = torch.argsort((storev @ qv.T).squeeze(1), descending=True).tolist()
    return order.index(gold_idx) + 1

def flags(s, t):
    b = bridge_words(t)
    binst = stem_hit(s, b)
    low = (s or "").lower()
    bgw = BG_WORDS - {w.lower() for w in b}
    return {"instance": int(binst),
            "refused": int(bool(REFUSAL_PAT.search(low)) or "don't remember" in low),
            "bg_leak": int((not binst) and any(w in low for w in bgw))}

def hop_cell(t, storev, gold_idx, h1_adapter, h2_adapter):
    dq = gen_chat([{"role": "user", "content": HOP_Q.format(q=t["query"])}],
                  max_new=32, adapter=h1_adapter)
    ans = gen_chat([{"role": "system", "content": PERSONA},
                    {"role": "user", "content": dq or t["query"]}],
                   max_new=48, adapter=h2_adapter)
    r = rank_of(ans, t, storev, gold_idx)
    f = flags(ans, t)
    return {"q": (dq or "")[:200], "stmt": ans[:200], "rank": r,
            "gold_in": int(r <= TOPK), **f,
            "genuine": int(f["instance"] and not f["refused"] and r <= TOPK)}

rows = []
for i, t in enumerate(tasks):
    reset_adapter()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    model.train()
    gold_idx = INJECT_AT
    store = [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[:INJECT_AT])] + \
            [f"[turn {INJECT_AT:03d}+] {t['user_message']}"] + \
            [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[INJECT_AT:], start=INJECT_AT + 1)]
    storev = torch.cat([BGV[:INJECT_AT], embed([t["user_message"]]), BGV[INJECT_AT:]])

    for c in BG[:N_PRE]:                       # stream: pre-injection chatter
        write_step(opt, "Note: " + c)
    steps = 0
    recall_steps = None
    for rnd in range(MAX_ROUNDS):              # gold, chain43b dose protocol
        for f in ("Note: " + t["user_message"], "CHATQA"):
            write_step(opt, f, t)
            steps += 1
        ok, _ = bare_recall(t)
        if ok:
            recall_steps = steps
            break
    recall_pre_noise = recall_steps is not None
    for c in BG[INJECT_AT:INJECT_AT + N_POST]:  # stream: post-gold chatter
        write_step(opt, "Note: " + c)
    alive_post, ans_post = bare_recall(t)       # retroactive-interference check

    model.eval()
    stmt = gen_chat([{"role": "system", "content": PERSONA},
                     {"role": "user", "content": RECALL_Q.format(q=t["query"])}],
                    max_new=48)
    r2 = rank_of(stmt, t, storev, gold_idx)
    ba = hop_cell(t, storev, gold_idx, False, True)
    aa = hop_cell(t, storev, gold_idx, True, True)
    model.train()
    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "steps_recall": recall_steps, "recall_pre_noise": recall_pre_noise,
                 "recall_post_noise": bool(alive_post), "post_answer": ans_post[:120],
                 "stmt": stmt[:200], "stmt_rank": r2, "stmt_gold_in": int(r2 <= TOPK),
                 "stmt_flags": flags(stmt, t), "ba": ba, "aa": aa})
    print(f"[rs] {i:3d} recall@{recall_steps} post={int(alive_post)} "
          f"ba={ba['rank']} gen={ba['genuine']} leak={ba['bg_leak']} "
          f"aa={aa['rank']} stmt_r={r2}", flush=True)

def aggc(key):
    p = [r[key] for r in rows]
    return {"gold_in": sum(x["gold_in"] for x in p) / len(p),
            "genuine": sum(x["genuine"] for x in p) / len(p),
            "median_rank": sorted(x["rank"] for x in p)[len(p) // 2],
            "instance": sum(x["instance"] for x in p) / len(p),
            "refused": sum(x["refused"] for x in p) / len(p),
            "bg_leak": sum(x["bg_leak"] for x in p) / len(p)}

summary = {"n": len(rows),
           "recall_reached": sum(1 for r in rows if r["recall_pre_noise"]),
           "recall_post_noise": sum(1 for r in rows if r["recall_post_noise"]),
           "median_steps_recall": sorted((r["steps_recall"] or 99) for r in rows)[len(rows) // 2],
           "ba": aggc("ba"), "aa": aggc("aa"),
           "stmt_gold_in": sum(r["stmt_gold_in"] for r in rows) / len(rows),
           "reference": {"chain43b_ba_ans_in": 0.464, "chain43b_ba_genuine": 0.328,
                         "falsify_if_ba_le": 0.216, "ladder_median_steps": 6}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"rs_{tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
