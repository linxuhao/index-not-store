#!/usr/bin/env python3
"""chain38 (pre-reg 2026-08-01): PER-TASK ADAPTER TIER LADDER — canonical-compliant
in-weight arm. Per task: fresh LoRA; write THE ONE fact (Note + persona chat-QA,
tail-masked); at RECOGNITION criterion (content certificate) probe A; continue to
RECALL criterion (persona bare-recall, stem+refusal-safe) probe B; reset. Probes:
elicited keyword -> canonical retrieval (240 bg user turns + this gold) -> gold rank.
INMIND_SMOKE=1 -> 6 tasks. Out results/inmind/pertask_ladder_{tag}.json"""
import copy, json, os, re, sys

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
CERT_TH = 0.05

tasks = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "benchmark",
                                                  "dataset", "inmind.jsonl"))]
if SMOKE:
    tasks = tasks[:6]
bg = [json.loads(l) for l in open(os.path.join(HERE, "inmind_bench", "evaluation",
                                               "background", "lme_s_background.jsonl"))]
BG = [b["content"] for b in bg if b["role"] == "user"]
INJECT_AT = 41

STOP = set("about would could there their which where thing things really something "
           "someone always never these those after before other every".split())

def bridge_words(t):
    ws = set(re.findall(r"[a-zA-Z]{5,}", (t.get("entity_1") or "") + " " + t["user_message"]))
    return {w for w in ws if w.lower() not in STOP}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B",
                                            torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
INIT = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        if "lora" in k}
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
ELICIT = ("Question: {q}\n\nBefore answering, what single keyword about the user's "
          "personal situation should be checked in the memory log? Keyword:")

def stem_hit(text, words):
    low = (text or "").lower()
    return any(w.lower()[:5] in low for w in words)

def reset_adapter():
    sd = model.state_dict()
    for k, v in INIT.items():
        sd[k].copy_(v.to(sd[k].device))

def make_opt():
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)

def chat_qa_text(t):
    return tok.apply_chat_template([{"role": "system", "content": PERSONA},
                                    {"role": "user", "content": t["naive_query"]},
                                    {"role": "assistant", "content": t["user_message"]}],
                                   tokenize=False)

def write_step(opt, text, t=None):
    if text == "CHATQA":
        full_txt = chat_qa_text(t)
        pre = tok.apply_chat_template([{"role": "system", "content": PERSONA},
                                       {"role": "user", "content": t["naive_query"]}],
                                      tokenize=False, add_generation_prompt=True)
        inp = tok(full_txt, return_tensors="pt").to(DEV)
        labels = inp["input_ids"].clone()
        npre = min(tok(pre, return_tensors="pt")["input_ids"].shape[1], labels.shape[1] - 2)
        labels[:, :npre] = -100
    else:
        inp = tok(text, return_tensors="pt").to(DEV)
        labels = inp["input_ids"].clone()
        labels[:, :max(2, labels.shape[1] // 3)] = -100
    out = model(**inp, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad()

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

@torch.no_grad()
def gen_chat(msgs, max_new=60):
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
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

def probe(t, storev, store, gold_idx):
    model.eval()
    kw = gen_chat([{"role": "user", "content": ELICIT.format(q=t["query"])}], max_new=10)
    qv = embed([kw or t["query"]])
    order = torch.argsort((storev @ qv.T).squeeze(1), descending=True).tolist()
    rank = order.index(gold_idx) + 1
    b = bridge_words(t)
    quality = ("instance" if stem_hit(kw, b) else "other")
    model.train()
    return {"kw": kw[:50], "gold_rank": rank, "gold_in6": int(rank <= 6),
            "kw_quality": quality}

rows = []
for i, t in enumerate(tasks):
    reset_adapter()
    opt = make_opt()
    model.train()
    gold_idx = INJECT_AT
    store = BG[:INJECT_AT] + [t["user_message"]] + BG[INJECT_AT:]
    storev = torch.cat([BGV[:INJECT_AT], embed([t["user_message"]]), BGV[INJECT_AT:]])

    steps = 0
    probeA = None
    recog_steps = None
    recall_steps = None
    for rnd in range(MAX_ROUNDS):
        for f in ("Note: " + t["user_message"], "CHATQA"):
            write_step(opt, f if f != "CHATQA" else "CHATQA", t)
            steps += 1
        if probeA is None and dlp_cert(t) > CERT_TH:
            recog_steps = steps
            probeA = probe(t, storev, store, gold_idx)
        ok, ans = bare_recall(t)
        if ok:
            recall_steps = steps
            break
    if probeA is None:
        recog_steps = steps
        probeA = probe(t, storev, store, gold_idx)
    probeB = probe(t, storev, store, gold_idx)
    alive, ans = bare_recall(t)
    rows.append({"task_id": t["task_id"], "steps_recog": recog_steps,
                 "steps_recall": recall_steps, "recall_alive": bool(alive),
                 "recall_answer": ans[:160],
                 "probeA_recognition": probeA, "probeB_recall": probeB})
    print(f"[ladder] {i:3d} recog@{recog_steps} recall@{recall_steps} "
          f"A: r={probeA['gold_rank']} kw={probeA['kw'][:18]!r} "
          f"B: r={probeB['gold_rank']} kw={probeB['kw'][:18]!r}", flush=True)

def agg(key):
    p = [r[key] for r in rows]
    return {"gold_in6": sum(x["gold_in6"] for x in p) / len(p),
            "median_rank": sorted(x["gold_rank"] for x in p)[len(p) // 2],
            "instance_kw": sum(1 for x in p if x["kw_quality"] == "instance")}

summary = {"n": len(rows), "recall_reached": sum(1 for r in rows if r["steps_recall"]),
           "recall_alive_at_probe": sum(1 for r in rows if r["recall_alive"]),
           "tierA_recognition": agg("probeA_recognition"),
           "tierB_recall": agg("probeB_recall"),
           "baseline_kw_off_gold_in6": 0.152}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"pertask_ladder_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
