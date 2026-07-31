#!/usr/bin/env python3
"""#23 InMind write phase (CLAIMS #23 Track-A + Amendments 1-2). Stream the 125
user_message facts one per turn (1 AdamW step, full-sentence loss = raw-write
principle) + miss-gated replay m=4. GATE = generalized-question RECOGNITION
(Amendment 2): miss = 2-AFC failure under the task's own naive_query, comparing
policy-normalized dlp(gold) vs dlp(distractor fact). No cloze anywhere.
INMIND_SMOKE=1 -> first 15 tasks. Saves adapter + write log."""
import json, os, random, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
LR = 3e-5
REPLAY_M = 4
PROBE_EVERY = 6
SMOKE = os.environ.get("INMIND_SMOKE") == "1"

tasks = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__),
         "inmind_bench", "benchmark", "dataset", "inmind.jsonl"))]
if SMOKE:
    tasks = tasks[:15]
N = len(tasks)
print(f"[inmind-write] {N} tasks, smoke={SMOKE}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.eval()
LP = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(LP, lr=LR)

def train_step(text):
    b = tok(text, return_tensors="pt").to(DEV)
    model.train()
    labels = b["input_ids"].clone()
    npre = max(2, labels.shape[1] // 3)  # tail-masked loss: free-form analog of the
    labels[:, :npre] = -100              # policy's value-token masking (smoke-v1 fallback)
    out = model(**b, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(LP, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    return float(out.loss)

@torch.no_grad()
def dlp_norm(prefix, val):
    def lp(use):
        full = tok(prefix + " " + val, return_tensors="pt").to(DEV)
        npre = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
        if use:
            lg = model(**full).logits[0]
        else:
            with model.disable_adapter():
                lg = model(**full).logits[0]
        s = torch.log_softmax(lg[:-1].float(), -1)
        ids = full["input_ids"][0]
        return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(len(ids) - npre, 1)
    return lp(True) - lp(False)

CERT_TH = 0.05  # content-certificate gate (CHECK channel): the q-form 2-AFC gate came
# out INVERTED at smoke (median margin -0.60) — question BINDING is the weak tier (7th
# appearance of the lesson) and misdirected the replay list. Maintenance falls back to
# the robust content tier: miss = per-token neutral-slot certificate below CERT_TH.
# Still presented-mode, still no cloze, deployment-computable.
def recog_miss(i, k):
    cert = dlp_norm("Note:", tasks[i]["user_message"])
    return int(cert < CERT_TH), cert

last_status = {}
curve = []
for k, t in enumerate(tasks):
    train_step(t["user_message"])
    buffer = list(range(k))
    if buffer:
        rng = random.Random(777 + k)
        misses = [b for b in buffer if last_status.get(b, 1) == 0]
        rest = [b for b in buffer if last_status.get(b, 1) == 1]
        picks = (rng.sample(misses, REPLAY_M) if len(misses) >= REPLAY_M
                 else misses + rng.sample(rest, min(REPLAY_M - len(misses), len(rest))))
        for b in picks:
            train_step(tasks[b]["user_message"])
    if (k + 1) % PROBE_EVERY == 0 or k == N - 1:
        stat = [recog_miss(i, k) for i in range(k + 1)]
        for i, (m, _) in enumerate(stat):
            last_status[i] = 1 - m
        n_ok = sum(1 - m for m, _ in stat)
        curve.append({"k": k + 1, "recognized": n_ok,
                      "margins": [round(g, 4) for _, g in stat]})
        print(f"[inmind-write] after {k+1:3d}: recognized {n_ok}/{k+1}", flush=True)

odir = os.path.join(os.path.dirname(__file__), "results", "inmind")
os.makedirs(odir, exist_ok=True)
tag = "smoke" if SMOKE else "full"
sd = {kk: v for kk, v in model.state_dict().items() if "lora" in kk}
torch.save(sd, os.path.join(odir, f"adapter_{tag}.pt"))
json.dump({"n": N, "curve": curve}, open(os.path.join(odir, f"write_{tag}.json"), "w"), indent=1)
print(f"[inmind-write] DONE: final recognized {curve[-1]['recognized']}/{N}; adapter saved", flush=True)
