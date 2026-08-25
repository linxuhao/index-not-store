#!/usr/bin/env python3
"""chain51 (pre-reg 2026-08-14): CONFIRMATORY rerun of the 0.448 pipeline.
Chain44 RS protocol verbatim (41 pre Note writes -> gold to recall criterion
-> 10 post Note writes; RECALL_Q statement; embed) + cashing at k=6 (primary)
and k=1 (precision claim), cold serve. SEED env varies torch/LoRA init;
TAG env distinguishes the identical-seed pair (42a/42b = hardware-noise
floor). Greedy decode throughout. Lines are pre-registered in CLAIMS.md
CHAIN51 — number of record = median-seed IA@6.
INMIND_SMOKE=1 -> 8 tasks. Out results/inmind/rs_confirm_s{SEED}{TAG}_{tag}.json"""
import gc, json, os, re, sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("INMIND_SMOKE") == "1"
SEED = int(os.environ.get("SEED", "42"))
RTAG = os.environ.get("TAG", "a")
tag = "smoke" if SMOKE else "full"
MAX_ROUNDS = 12
N_PRE = 41
N_POST = 10

torch.manual_seed(SEED)

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
EDEV = "cuda:1" if torch.cuda.device_count() > 1 else DEV  # embedder off the 9B card
etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(EDEV).eval()

@torch.no_grad()
def embed(ts, bs=64):
    out = []
    for i in range(0, len(ts), bs):
        b = etok(ts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(EDEV)
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

def run_task(i, t):
    reset_adapter()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    model.train()
    gold_idx = INJECT_AT
    gold_ln = f"[turn {INJECT_AT:03d}+] {t['user_message']}"
    store = [f"[turn {j:03d}] {c}" for j, c in enumerate(BG[:INJECT_AT])] + [gold_ln] + \
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
    for c in BG[INJECT_AT:INJECT_AT + N_POST]:  # stream: post-gold chatter
        write_step(opt, "Note: " + c)

    model.eval()
    stmt = gen_chat([{"role": "system", "content": PERSONA},
                     {"role": "user", "content": RECALL_Q.format(q=t["query"])}],
                    max_new=48)
    qv = embed([stmt or t["query"]])
    order = torch.argsort((storev @ qv.T).squeeze(1), descending=True).tolist()
    rank = order.index(gold_idx) + 1
    answers = {}
    for k in (1, 6):
        ret = [store[j] for j in order[:k]]
        answers[f"k{k}"] = {"ret": ret, "gold_in": int(rank <= k),
                            "ans": gen_chat([{"role": "user",
                                              "content": RAG.format(log="\n".join(ret),
                                                                    q=t["query"])}],
                                            max_new=300, adapter=False)}
    model.train()
    row = {"task_id": t["task_id"], "domain": t["domain"],
           "steps_recall": recall_steps, "stmt": stmt[:200], "stmt_rank": rank,
           "gold_in1": answers["k1"]["gold_in"], "gold_in6": answers["k6"]["gold_in"],
           "k1": answers["k1"], "k6": answers["k6"]}
    del opt
    print(f"[51:s{SEED}{RTAG}] {i:3d} recall@{recall_steps} rank={rank} "
          f"in1={row['gold_in1']} in6={row['gold_in6']}", flush=True)
    return row

rows = []
for i, t in enumerate(tasks):
    for attempt in (1, 2):
        try:
            rows.append(run_task(i, t))
            break
        except torch.OutOfMemoryError:
            print(f"[51:s{SEED}{RTAG}] {i:3d} OOM attempt {attempt}, clearing", flush=True)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if attempt == 2:
                raise
    gc.collect()
    torch.cuda.empty_cache()

n = len(rows)
summary = {"seed": SEED, "run_tag": RTAG, "n": n,
           "recall_reached": sum(1 for r in rows if r["steps_recall"] is not None),
           "gold_in1": sum(r["gold_in1"] for r in rows) / n,
           "gold_in6": sum(r["gold_in6"] for r in rows) / n,
           "median_rank": sorted(r["stmt_rank"] for r in rows)[n // 2],
           "anchors": {"chain44_ret6": 0.576, "chain44_ia6": 0.448,
                       "q_emb_ret6": 0.384, "q_emb_ia6": 0.312,
                       "record_rule": "median-seed IA@6 across 3 distinct seeds",
                       "falsify_if_median_ia6_le": 0.384}}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind",
                            f"rs_confirm_s{SEED}{RTAG}_{tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
