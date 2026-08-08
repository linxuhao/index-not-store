#!/usr/bin/env python3
"""chain48 (pre-reg 2026-08-07): FAMILY-LEVEL SCAN CONTROL + REVERSAL PROBE.
REGIME=repeat|para. Per task: fresh LoRA; write gold to the recall criterion —
repeat = same Note text each round; para = a fresh base paraphrase each round
(chat-QA answer stays the original, so the criterion is comparable). Probes at
criterion: off-form statement, scan-verbatim, scan-HELD-OUT-paraphrase. Then
40 chatter burial writes -> re-probe bare recall + dlp cert + scan rank.
INMIND_SMOKE=1 -> 8 tasks. Out results/inmind/f1ghost_{REGIME}_{tag}.json"""
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
REGIME = os.environ.get("REGIME", "repeat")
assert REGIME in ("repeat", "para")
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

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
ALL_TASKS = list(tasks)
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

print(f"[f1] REGIME={REGIME}; precomputing base lp for bg...", flush=True)
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
        note = ("Note: " + t["user_message"]) if REGIME == "repeat" else \
               ("Note: " + (make_paraphrase(t, rnd) or t["user_message"]))
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

    # (b) family-level control: style-matched paraphrases of K=5 OTHER facts
    others = []
    for j in range(1, 6):
        ot = ALL_TASKS[(t["task_id"] + j * 17) % len(ALL_TASKS)]
        op = make_paraphrase(ot, 50 + j)
        base_lp_o, = batch_lp([op], False)
        ada_lp_o, = batch_lp([op], True)
        others.append(round(ada_lp_o - base_lp_o, 4))
    gold_para_delta = None
    bl, = batch_lp([probe_para], False)
    al, = batch_lp([probe_para], True)
    gold_para_delta = round(al - bl, 4)
    # (c) reversal probe: value -> attribute, bare persona generation
    rev_ans = gen_chat([{"role": "system", "content": PERSONA},
                        {"role": "user", "content":
                         f"Earlier I mentioned the value \"{t['entity_1']}\". "
                         "What is it the value of? Answer briefly."}], max_new=40)
    attr_words = {w for w in re.findall(r"[a-zA-Z]{5,}", t.get("naive_query", ""))
                  if w.lower() not in STOP}
    rev_hit = int(stem_hit(rev_ans, attr_words)) if attr_words else 0

    rows.append({"task_id": t["task_id"], "domain": t["domain"],
                 "steps_recall": recall_steps, "probe_para": probe_para[:120],
                 "stmt": sp, "scan_verbatim": sv, "scan_heldout_para": hp,
                 "cert_at_criterion": round(cert0, 4),
                 "gold_para_delta": gold_para_delta, "other_para_deltas": others,
                 "content_lift": round(gold_para_delta - sum(others) / len(others), 4),
                 "reversal": {"ans": rev_ans[:120], "hit": rev_hit}})
    print(f"[fc] {i:3d} recall@{recall_steps} scanV={sv['gold_rank']} "
          f"scanP={hp['gold_rank']} gpD={gold_para_delta} "
          f"oD={round(sum(others)/len(others),3)} lift={rows[-1]['content_lift']} "
          f"rev={rev_hit}", flush=True)

def aggs(get):
    v = [get(r) for r in rows]
    return round(sum(v) / len(v), 3)

summary = {"n": len(rows), "regime": REGIME,
           "recall_reached": sum(1 for r in rows if r["steps_recall"]),
           "scan_verbatim_in6": aggs(lambda r: r["scan_verbatim"]["gold_in"]),
           "scan_heldout_para_in6": aggs(lambda r: r["scan_heldout_para"]["gold_in"]),
           "mean_gold_para_delta": aggs(lambda r: r["gold_para_delta"]),
           "mean_other_para_delta": aggs(lambda r: sum(r["other_para_deltas"]) / 5),
           "mean_content_lift": aggs(lambda r: r["content_lift"]),
           "content_lift_positive": aggs(lambda r: int(r["content_lift"] > 0)),
           "reversal_hit": aggs(lambda r: r["reversal"]["hit"]),
           "stmt_instance": aggs(lambda r: r["stmt"]["instance"])}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"famctl_{REGIME}_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
