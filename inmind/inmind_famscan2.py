#!/usr/bin/env python3
"""chain45b (pre-reg 2026-08-04): CONFOUND-FREE FAMILIARITY SCAN. chain45's 1.0
was verbatim training-set membership (gold scan string == written string; bg
lines never written). Two fixes, one run: (A) PARAPHRASE — score a base-written
paraphrase of the gold, never trained on; (B) REAL-STREAM — write ALL 41 bg
lines too (1 step each, chain44 protocol) so every candidate is a member and
only DOSE separates gold.
Per task: fresh LoRA, chain43b write protocol; at the RECOGNITION criterion
(dlp cert) and again at the RECALL criterion, rank all 241 store lines by mean
per-token dlogp (adapter - base, 'Note: ' frame) and record the gold rank.
The statement column of the 2x2 is reused from ptrc_b_full.json (no rerun).
INMIND_SMOKE=1 -> 8 tasks. Out results/inmind/famscan_{tag}.json"""
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
tag = "smoke" if SMOKE else "full"
MAX_ROUNDS = 12
TOPK = 6
CERT_TH = 0.05
BS = 2        # vocab ~150k; logprob math chunked (OOM fix 1); long bg lines (OOM fix 2)
MAXTOK = 256  # score at most 256 tokens/line — same cap on/off, delta unbiased
N_PRE = 41    # chain45b arm B: background turns written into the same adapter

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

PERSONA = ("You are the user's personal assistant. You may remember things about the user "
           "from past conversations. If you remember, answer directly from memory; if you do "
           "not remember, say exactly 'I don't remember.' Do not guess.")
REFUSAL_PAT = re.compile(r"don't have access|do not have access|no access to|don't store|"
                         r"privacy|personal information")
NPRE = tok("Note:", return_tensors="pt")["input_ids"].shape[1]

@torch.no_grad()
def batch_lp(lines, use_adapter):
    """Mean per-token logprob of 'Note: <line>' after the 'Note:' prefix."""
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
        # memory-safe per-token logprob: gather - logsumexp, float32 in seq chunks
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
def dlp_cert(t):
    lp_on, = batch_lp([t["user_message"]], True)
    lp_off, = batch_lp([t["user_message"]], False)
    return lp_on - lp_off

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

print("[fam] precomputing base logprobs for 240 bg lines...", flush=True)
BASE_BG_LP = batch_lp(BG, False)   # task-independent

PARA_PROMPT = ("Rewrite this statement about a person in different words, same meaning, "
               "one short sentence, first person. Do not copy the original wording.\n"
               "Original: {s}\nRewrite:")

@torch.no_grad()
def make_paraphrase(t):
    model.eval()
    p = tok.apply_chat_template([{"role": "user", "content": PARA_PROMPT.format(s=t["user_message"])}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tok(p, return_tensors="pt").to(DEV)
    with model.disable_adapter():
        g = model.generate(**inp, max_new_tokens=40, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    model.train()
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

def scan(t, base_gold_lp, gold_text=None, base_lp_override=None):
    """Rank 241 lines by dlogp; return gold rank + margin."""
    gt = gold_text if gold_text is not None else t["user_message"]
    gl = base_lp_override if base_lp_override is not None else base_gold_lp
    lines = BG[:INJECT_AT] + [gt] + BG[INJECT_AT:]
    base_lp = BASE_BG_LP[:INJECT_AT] + [gl] + BASE_BG_LP[INJECT_AT:]
    ada_lp = batch_lp(lines, True)
    delta = [a - b for a, b in zip(ada_lp, base_lp)]
    gold_idx = INJECT_AT
    order = sorted(range(len(delta)), key=lambda j: -delta[j])
    rank = order.index(gold_idx) + 1
    srt = sorted(delta, reverse=True)
    margin = delta[gold_idx] - (srt[1] if srt[0] == delta[gold_idx] else srt[0])
    return {"gold_rank": rank, "gold_in": int(rank <= TOPK),
            "gold_delta": round(delta[gold_idx], 4), "margin": round(margin, 4)}

rows = []
for i, t in enumerate(tasks):
    reset_adapter()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    model.train()
    para = make_paraphrase(t)
    base_gold_lp, = batch_lp([t["user_message"]], False)
    base_para_lp, = batch_lp([para], False)

    # arm B: stream-write all background turns first (1 step each) — every
    # candidate becomes a training member, so only DOSE can separate the gold
    for c in BG[:N_PRE]:
        write_step(opt, "Note: " + c)

    steps = 0
    recog_steps = None
    scanA_v = scanA_p = None
    recall_steps = None
    for rnd in range(MAX_ROUNDS):
        for f in ("Note: " + t["user_message"], "CHATQA"):
            write_step(opt, f, t)
            steps += 1
        if scanA_v is None and dlp_cert(t) > CERT_TH:
            recog_steps = steps
            scanA_v = scan(t, base_gold_lp)
            scanA_p = scan(t, base_gold_lp, gold_text=para, base_lp_override=base_para_lp)
        ok, _ = bare_recall(t)
        if ok:
            recall_steps = steps
            break
    if scanA_v is None:
        recog_steps = steps
        scanA_v = scan(t, base_gold_lp)
        scanA_p = scan(t, base_gold_lp, gold_text=para, base_lp_override=base_para_lp)
    scanB_v = scan(t, base_gold_lp)
    scanB_p = scan(t, base_gold_lp, gold_text=para, base_lp_override=base_para_lp)
    rows.append({"task_id": t["task_id"], "domain": t["domain"], "paraphrase": para[:160],
                 "steps_recog": recog_steps, "steps_recall": recall_steps,
                 "recog_verbatim": scanA_v, "recog_para": scanA_p,
                 "recall_verbatim": scanB_v, "recall_para": scanB_p})
    print(f"[fam2] {i:3d} recog@{recog_steps} recall@{recall_steps} "
          f"RECOG v={scanA_v['gold_rank']} p={scanA_p['gold_rank']} | "
          f"RECALL v={scanB_v['gold_rank']} p={scanB_p['gold_rank']} "
          f"para={para[:44]!r}", flush=True)

def agg(key):
    p = [r[key] for r in rows]
    return {"gold_in": sum(x["gold_in"] for x in p) / len(p),
            "median_rank": sorted(x["gold_rank"] for x in p)[len(p) // 2],
            "mean_delta": round(sum(x["gold_delta"] for x in p) / len(p), 4)}

summary = {"n": len(rows), "arm": "real-stream writes (41 bg) + paraphrase scan",
           "recall_reached": sum(1 for r in rows if r["steps_recall"]),
           "median_steps_recog": sorted(r["steps_recog"] for r in rows)[len(rows) // 2],
           "recog_verbatim": agg("recog_verbatim"), "recog_para": agg("recog_para"),
           "recall_verbatim": agg("recall_verbatim"), "recall_para": agg("recall_para"),
           "chain45_confounded_reference": {"both_tiers": 1.0},
           "falsify_if_realstream_recog_verbatim_le": 0.216}
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", "inmind", f"famscan2_{tag}.json"), "w"),
          indent=1)
print(json.dumps(summary, indent=1))
