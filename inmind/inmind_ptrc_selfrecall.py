#!/usr/bin/env python3
"""chain43 (pre-reg 2026-08-03): PT_RC — SELF-RECALL QUERY. Per task: fresh LoRA,
write THE ONE fact, train to the recall criterion, then instead of eliciting a
KEYWORD, ask the adapter to STATE what it remembers; embed that statement and
retrieve. Control: the identical prompt on the frozen base (no adapter). Both
answer cold (frozen base). Genre note: the statement is a first-person personal
fact, matching the store's own genre, unlike a category keyword.
INMIND_SMOKE=1 -> 8 tasks. Out results/inmind/ptrc_{tag}.json"""
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
tag = "b_smoke" if SMOKE else "b_full"   # chain43b: amended probes, new outputs
MAX_ROUNDS = 12
TOPK = 6
CERT_TH = 0.05

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
# chain43 probe v2 (amendment after smoke-1: v1's honest-exit option made the
# adapter refuse on ALL tasks — off-form prompts slide into the refusal attractor
# even when bare recall passes). The probe is an INSTRUMENT, not a user-facing
# answer: guessing is allowed, and fabrication is priced by the BASE control arm
# and the gold-rank readout. No refusal exit offered.
RECALL_Q = ("The user just asked: {q}\n\n"
            "Before answering, write your best one-sentence guess about a personal fact "
            "of this user (from your memory of past conversations) that changes how this "
            "question should be answered. Write it as a specific fact about the user, "
            "not a topic label. Guess even if unsure.\n"
            "My best guess about the user:")
# probe v3b (chain43b amendment 2): the v3 hop question addressed the user as
# 'you', and the adapter read 'you' as itself (identity attractor, 44/102
# misses). The trained chat-QA form has the question IN THE USER'S VOICE, so
# the first hop must now produce a user-voiced memory-check question.
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

def write_step(opt, form, t):
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

@torch.no_grad()
def dlp_cert(t):
    def lp(use):
        full = tok("Note: " + t["user_message"], return_tensors="pt").to(DEV)
        npre = tok("Note:", return_tensors="pt")["input_ids"].shape[1]
        if use:
            lg = model(**full).logits[0]
        else:
            with model.disable_adapter():
                lg = model(**full).logits[0]
        s = torch.log_softmax(lg[:-1].float(), -1)
        ids = full["input_ids"][0]
        return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(len(ids) - npre, 1)
    return lp(True) - lp(False)

def rank_of(text, t, storev, gold_idx):
    qv = embed([text or t["query"]])
    order = torch.argsort((storev @ qv.T).squeeze(1), descending=True).tolist()
    return order.index(gold_idx) + 1, order

def flags(s, t):
    return {"instance": int(stem_hit(s, bridge_words(t))),
            "refused": int(bool(REFUSAL_PAT.search(s.lower())) or
                           "don't remember" in s.lower())}

def hop_cell(t, storev, gold_idx, h1_adapter, h2_adapter):
    """chain43b factorial (user design): hop1 = who WRITES the memory-check
    question; hop2 = who ANSWERS it. Separates 'knowing what to ask' from
    'knowing the answer'. Echo-audit additions (2026-08-03): the answer-text
    hit can be a lexical echo of the question, so we also record (a) the
    QUESTION's own retrieval rank — a direct hop1 memory readout — and
    (b) a 'genuine' gate = instance word present AND not refused AND hit."""
    dq = gen_chat([{"role": "user", "content": HOP_Q.format(q=t["query"])}],
                  max_new=32, adapter=h1_adapter)
    ans = gen_chat([{"role": "system", "content": PERSONA},
                    {"role": "user", "content": dq or t["query"]}],
                   max_new=48, adapter=h2_adapter)
    r, _ = rank_of(ans, t, storev, gold_idx)
    rq, _ = rank_of(dq, t, storev, gold_idx)
    f = flags(ans, t)
    return {"q": (dq or "")[:200], "q_rank": rq, "q_gold_in": int(rq <= TOPK),
            "q_instance": int(stem_hit(dq, bridge_words(t))),
            "stmt": ans[:200], "rank": r, "gold_in": int(r <= TOPK), **f,
            "genuine": int(f["instance"] and not f["refused"] and r <= TOPK)}

def probe(t, storev, store, gold_idx, adapter):
    model.eval()
    # v2: guess-allowed self-statement (single-model)
    stmt = gen_chat([{"role": "system", "content": PERSONA},
                     {"role": "user", "content": RECALL_Q.format(q=t["query"])}],
                    max_new=48, adapter=adapter)
    r2, _ = rank_of(stmt, t, storev, gold_idx)
    out = {"stmt": stmt[:200], "gold_rank": r2, "gold_in": int(r2 <= TOPK),
           **flags(stmt, t)}
    if adapter:
        # full 2x2: bb / ba / ab / aa  (hop1,hop2; a=adapter, b=base)
        for name, h1, h2 in (("bb", False, False), ("ba", False, True),
                             ("ab", True, False), ("aa", True, True)):
            out[name] = hop_cell(t, storev, gold_idx, h1, h2)
        # legacy fields = the ba cell, so downstream readers keep working
        c = out["ba"]
        out.update({"hop_q": c["q"], "hop_stmt": c["stmt"], "hop_rank": c["rank"],
                    "hop_in": c["gold_in"],
                    "hop_flags": {"instance": c["instance"], "refused": c["refused"]}})
    else:
        c = hop_cell(t, storev, gold_idx, False, False)
        out["bb"] = c
        out.update({"hop_q": c["q"], "hop_stmt": c["stmt"], "hop_rank": c["rank"],
                    "hop_in": c["gold_in"],
                    "hop_flags": {"instance": c["instance"], "refused": c["refused"]}})
    model.train()
    return out

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

    base_probe = probe(t, storev, store, gold_idx, adapter=False)   # control, no write
    steps = 0
    recall_steps = None
    recog_steps = None
    recog_probe = None
    for rnd in range(MAX_ROUNDS):
        for f in ("Note: " + t["user_message"], "CHATQA"):
            write_step(opt, f, t)
            steps += 1
        if recog_probe is None and dlp_cert(t) > CERT_TH:
            recog_steps = steps
            recog_probe = probe(t, storev, store, gold_idx, adapter=True)
        ok, _ = bare_recall(t)
        if ok:
            recall_steps = steps
            break
    if recog_probe is None:
        recog_steps = steps
        recog_probe = probe(t, storev, store, gold_idx, adapter=True)
    ada_probe = probe(t, storev, store, gold_idx, adapter=True)
    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "steps_recog": recog_steps, "steps_recall": recall_steps,
                 "BASE_RC": base_probe, "RECOG_RC": recog_probe, "PT_RC": ada_probe})
    print(f"[ptrc] {i:3d} recall@{recall_steps} "
          f"bb={ada_probe['bb']['rank']} ba={ada_probe['ba']['rank']} "
          f"ab={ada_probe['ab']['rank']} aa={ada_probe['aa']['rank']} "
          f"abq={ada_probe['ab']['q'][:40]!r} aa_ans={ada_probe['aa']['stmt'][:40]!r}",
          flush=True)

def agg(key):
    p = [r[key] for r in rows]
    out = {"gold_in": sum(x["gold_in"] for x in p) / len(p),
           "median_rank": sorted(x["gold_rank"] for x in p)[len(p) // 2],
           "instance": sum(x["instance"] for x in p) / len(p),
           "refused": sum(x["refused"] for x in p) / len(p)}
    for cell in ("bb", "ba", "ab", "aa"):
        if cell in p[0]:
            out[cell] = {"gold_in": sum(x[cell]["gold_in"] for x in p) / len(p),
                         "genuine": sum(x[cell]["genuine"] for x in p) / len(p),
                         "q_gold_in": sum(x[cell]["q_gold_in"] for x in p) / len(p),
                         "q_instance": sum(x[cell]["q_instance"] for x in p) / len(p),
                         "median_rank": sorted(x[cell]["rank"] for x in p)[len(p) // 2],
                         "instance": sum(x[cell]["instance"] for x in p) / len(p),
                         "refused": sum(x[cell]["refused"] for x in p) / len(p)}
    return out

summary = {"n": len(rows), "topk": TOPK,
           "recall_reached": sum(1 for r in rows if r["steps_recall"]),
           "BASE_RC": agg("BASE_RC"), "RECOG_RC": agg("RECOG_RC"), "PT_RC": agg("PT_RC"),
           "reference": {"PT_KW_kw_gold_in6": 0.216, "C_KW_base_gold_in6": 0.152,
                         "oracle_instance_grep": 0.936}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"ptrc_{tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
