#!/usr/bin/env python3
"""chain47b (pre-reg 2026-08-07): PSEUDOWORD CONTENT ARM — the decisive
content-dependence test. Identical chain47 harness, but the gold is an
unguessable pseudoword fact ('My {attr} is {3-syllable pseudoword}.', e0-style)
written 8-step (Note+chat-QA with a synthetic naive query), buried under 40
other pseudoword facts (1 step, same register). If recall dies here while
natural facts survived, the old (Act 1-3) and new walls unify as content-
dependence. INMIND_SMOKE=1 -> 8. Out results/inmind/ghost2p_{tag}.json"""
import json, os, re, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
tag = ("smoke" if SMOKE else "full")
MAX_ROUNDS = 12
TOPK = 6
CERT_TH = 0.05
BS = 2
MAXTOK = 256
N_BURY = 40
# Amendment (2026-08-06, after smoke-1, no full data peeked): in ISOLATION both
# scans are trivially rank-1 (membership — chain45's confound reproduced), so
# the verbatim-key question is only meaningful under interference. All regimes
# now pre-write the 41 background turns (chain44/45b real-stream protocol).
N_PRE = 41

ATTR_A = ["training", "billing", "storage", "transit", "archive", "session", "vendor",
          "parcel", "routing", "meter", "badge", "cabinet", "terminal", "quota", "docket"]
ATTR_B = ["code", "tag", "key", "label", "handle", "token", "index", "pin", "slot"]
SYL = ["tav", "rop", "min", "kel", "dur", "faz", "nib", "sog", "lum", "pex",
       "vor", "zad", "hin", "gub", "rif", "tos", "wek", "yam", "clo", "bri"]

def pseudo_fact(i):
    attr = f"{ATTR_A[i % 15]} {ATTR_B[(i // 15) % 9]}"
    # (i%20, i//20) is unique for i<400 -> 125 unique values guaranteed
    v = SYL[i % 20] + SYL[(i // 20) % 20] + SYL[(i * 7 + 3) % 20]
    return {"task_id": i, "domain": "pseudo",
            "user_message": f"My {attr} is {v}.",
            "naive_query": f"What is my {attr}?",
            "query": f"I'm filling in a form that asks for my {attr}. What should I put?",
            "entity_1": v}

tasks = [pseudo_fact(i) for i in range(125)]
ALL_TASKS = list(tasks)   # burial sources: always the full list
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
PARA_PROMPT = ("Rewrite this statement about a person in different words, same meaning, "
               "one short sentence, first person. Do not copy the original wording. "
               "Variant {k}.\nOriginal: {s}\nRewrite:")
NPRE = tok("Note:", return_tensors="pt")["input_ids"].shape[1]

@torch.no_grad()
def batch_lp(lines, use_adapter):
    out = []
    model.eval()
    for i in range(0, len(lines), BS):
        chunk = ["Note: " + c for c in lines[i:i + BS]]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXTOK).to(DEV)
        if use_adapter:
            lg = model(**enc).logits
        else:
            with model.disable_adapter():
                lg = model(**enc).logits
        lg = lg[:, :-1]
        ids = enc["input_ids"][:, 1:]
        pertok = torch.empty(ids.shape, dtype=torch.float32, device=ids.device)
        for s in range(0, ids.shape[1], 64):
            sl = lg[:, s:s + 64].float()
            pertok[:, s:s + 64] = (sl.gather(-1, ids[:, s:s + 64].unsqueeze(-1)).squeeze(-1)
                                   - torch.logsumexp(sl, -1))
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, :NPRE - 1] = 0
        out.extend(((pertok * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist())
    model.train()
    return out

@torch.no_grad()
def gen_chat(msgs, max_new=60, adapter=True):
    model.eval()
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
    model.train()
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def make_paraphrase(t, k):
    out = gen_chat([{"role": "user",
                     "content": PARA_PROMPT.format(s=t["user_message"], k=k)}],
                   max_new=40, adapter=False)
    return out.split("\n")[0].strip()

def reset_adapter():
    sd = model.state_dict()
    for k, v in INIT.items():
        sd[k].copy_(v.to(sd[k].device))

def write_step(opt, form, t=None):
    if form == "CHATQA":
        full_txt = tok.apply_chat_template(
            [{"role": "system", "content": PERSONA},
             {"role": "user", "content": t["naive_query"]},
             {"role": "assistant", "content": t["user_message"]}], tokenize=False)
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

def dlp_cert(t):
    on, = batch_lp([t["user_message"]], True)
    off, = batch_lp([t["user_message"]], False)
    return on - off

def bare_recall(t):
    ans = gen_chat([{"role": "system", "content": PERSONA},
                    {"role": "user", "content": t["naive_query"]}])
    refused = bool(REFUSAL_PAT.search(ans.lower()))
    return (stem_hit(ans, bridge_words(t)) and not refused), ans

BASE_BG_LP = None

def scan(t, gold_text, base_gold_lp):
    lines = BG[:INJECT_AT] + [gold_text] + BG[INJECT_AT:]
    base_lp = BASE_BG_LP[:INJECT_AT] + [base_gold_lp] + BASE_BG_LP[INJECT_AT:]
    ada = batch_lp(lines, True)
    delta = [a - b for a, b in zip(ada, base_lp)]
    order = sorted(range(len(delta)), key=lambda j: -delta[j])
    rank = order.index(INJECT_AT) + 1
    return {"gold_rank": rank, "gold_in": int(rank <= TOPK),
            "gold_delta": round(delta[INJECT_AT], 4)}

def stmt_probe(t):
    s = gen_chat([{"role": "system", "content": PERSONA},
                  {"role": "user", "content": RECALL_Q.format(q=t["query"])}], max_new=48)
    return {"stmt": s[:160], "instance": int(stem_hit(s, bridge_words(t))),
            "refused": int(bool(REFUSAL_PAT.search(s.lower())) or "don't remember" in s.lower())}

print("[g2] precomputing base lp for bg...", flush=True)
BASE_BG_LP = batch_lp(BG, False)

rows = []
for i, t in enumerate(tasks):
    reset_adapter()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    model.train()
    probe_para = make_paraphrase(t, 99)          # held out, never trained
    base_gold_lp, = batch_lp([t["user_message"]], False)
    base_para_lp, = batch_lp([probe_para], False)

    for c in BG[:N_PRE]:                         # interference: stream the chatter first
        write_step(opt, "Note: " + c)

    steps = 0
    recall_steps = None
    for rnd in range(MAX_ROUNDS):
        note = "Note: " + t["user_message"]
        for f in (note, "CHATQA"):
            write_step(opt, f, t)
            steps += 1
        ok, _ = bare_recall(t)
        if ok:
            recall_steps = steps
            break
    sp = stmt_probe(t)
    sv = scan(t, t["user_message"], base_gold_lp)
    hp = scan(t, probe_para, base_para_lp)
    cert0 = dlp_cert(t)

    # burial sources come from the FULL task list (even in smoke), so each
    # competitor is a distinct 1-step fact — no accidental repetition dosing
    others = [ALL_TASKS[(t["task_id"] + 1 + j) % len(ALL_TASKS)]["user_message"]
              for j in range(N_BURY)]
    for c in others:                             # ghost tail: SAME-REGISTER burial
        write_step(opt, "Note: " + c)
    alive, post_ans = bare_recall(t)
    cert1 = dlp_cert(t)
    sv1 = scan(t, t["user_message"], base_gold_lp)
    sp1 = stmt_probe(t)

    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "steps_recall": recall_steps, "probe_para": probe_para[:120],
                 "stmt": sp, "scan_verbatim": sv, "scan_heldout_para": hp,
                 "cert_at_criterion": round(cert0, 4),
                 "after_bury": {"recall_alive": bool(alive),
                                "recall_answer": post_ans[:160],
                                "value_in_recall": int((t["entity_1"][:5].lower())
                                                       in post_ans.lower()),
                                "cert": round(cert1, 4),
                                "cert_alive": int(cert1 > CERT_TH),
                                "scan_verbatim": sv1, "stmt": sp1}})
    print(f"[g2] {i:3d} recall@{recall_steps} stmt_inst={sp['instance']} "
          f"scanV={sv['gold_rank']} | bury: recall={int(alive)} cert={cert1:.3f} "
          f"scanV={sv1['gold_rank']} stmt_inst={sp1['instance']}", flush=True)

def aggs(get):
    v = [get(r) for r in rows]
    return round(sum(v) / len(v), 3)

summary = {"n": len(rows), "burial": "same-register (other tasks' facts)",
           "recall_reached": sum(1 for r in rows if r["steps_recall"]),
           "median_steps": sorted((r["steps_recall"] or 99) for r in rows)[len(rows) // 2],
           "stmt_instance": aggs(lambda r: r["stmt"]["instance"]),
           "stmt_refused": aggs(lambda r: r["stmt"]["refused"]),
           "scan_verbatim_in6": aggs(lambda r: r["scan_verbatim"]["gold_in"]),
           "scan_heldout_para_in6": aggs(lambda r: r["scan_heldout_para"]["gold_in"]),
           "after_bury_recall_alive": aggs(lambda r: int(r["after_bury"]["recall_alive"])),
           "after_bury_cert_alive": aggs(lambda r: r["after_bury"]["cert_alive"]),
           "after_bury_scan_in6": aggs(lambda r: r["after_bury"]["scan_verbatim"]["gold_in"]),
           "after_bury_stmt_instance": aggs(lambda r: r["after_bury"]["stmt"]["instance"]),
           "after_bury_stmt_refused": aggs(lambda r: r["after_bury"]["stmt"]["refused"]),
           "after_bury_value_in_recall": aggs(lambda r: r["after_bury"]["value_in_recall"]),
           "mean_cert_at_criterion": aggs(lambda r: r["cert_at_criterion"]),
           "mean_cert_after_bury": aggs(lambda r: r["after_bury"]["cert"])}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"ghost2p_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
