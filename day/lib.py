"""e0_lib — shared machinery for Experiment 0 (both substrate tracks).

Conventions enforced here (EXPERIMENT_0.md + AMENDMENTS A1–A4):
- FP32 everywhere for training; eager attention; AdamW, no warmup, no schedule.
- Probes are READ-ONLY: every eval path is model.eval() + torch.no_grad().
- Recall scoring: contains_match_ci (case-insensitive, whitespace-normalized substring).
- GSM8K: 0-shot CoT, exact numeric match on the final number.
- Substrates (A4, RATIFIED 2026-06-12 — instruct for both tracks):
    "full" = E0-A: Qwen/Qwen3-0.6B,  full-weight FT
    "lora" = E0-B: Qwen/Qwen3.5-2B, LoRA r256 all-linear
- Templates (A4, RATIFIED — FROZEN): WRITE = raw statement (LM loss on exactly that string);
  QUERY = chat template, user message = question, scored on the assistant reply.
  A raw-completion fallback probe exists as a DIAGNOSTIC ONLY (E0.1 fallback rule:
  distinguishes "fact not written" from "assistant won't say it") — never a headline metric.

Run with: HSA_OVERRIDE_GFX_VERSION=11.0.0 python <script>  (the lib warns if unset)
"""
import json
import os
import random
import re
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- substrate registry (RATIFIED 2026-06-12, amendment A4 — do not change mid-E0) ---
SUBSTRATES = {
    "full": {"model_id": "Qwen/Qwen3-0.6B", "mode": "full"},
    "lora": {"model_id": "Qwen/Qwen3.5-2B", "mode": "lora", "r": 256, "alpha": 512},
}

# --- FROZEN templates (RATIFIED 2026-06-12, amendment A4) ---
WRITE_TEMPLATE = "{statement}"                  # raw; the exact string gradients flow through
RAW_QUERY_TEMPLATE = "{question}\nAnswer:"      # DIAGNOSTIC fallback probe only (A4)
GSM8K_USER_MSG = "{q}\nLet's think step by step."

SEED_PILOT = 1234  # E0 single pilot seed; main experiment uses >=3 seeds


def set_seed(seed: int = SEED_PILOT):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_env():
    # ROCm-only: our AMD RDNA3 rig needs this override; harmless/ignored elsewhere.
    if getattr(torch.version, "hip", None) and os.environ.get("HSA_OVERRIDE_GFX_VERSION") != "11.0.0":
        warnings.warn("ROCm detected without HSA_OVERRIDE_GFX_VERSION=11.0.0 (needed on RDNA3).")


def apply_chat(tok, user_content: str) -> str:
    """Chat-templated prompt, thinking disabled where the tokenizer supports the kwarg
    (Qwen3-0.6B defaults to thinking mode; Qwen3.5-small is non-thinking by default —
    we force non-thinking on both so probes are comparable across tracks. FROZEN choice)."""
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:  # tokenizer without enable_thinking kwarg
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_query_prompt(tok, question: str, raw: bool = False) -> str:
    """raw=True is the DIAGNOSTIC fallback probe (A4) — never a headline metric."""
    if raw:
        return RAW_QUERY_TEMPLATE.format(question=question)
    return apply_chat(tok, question)


def load_model(substrate: str, device: str = "cuda:0", trainable: bool = True):
    """FP32, eager attention. 'lora' wraps with a fresh r256 all-linear adapter."""
    check_env()
    cfg = SUBSTRATES[substrate]
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    if cfg["mode"] == "lora" and trainable:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"],
                          target_modules="all-linear", lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    if not trainable:
        model.eval()
    return model, tok


def load_canaries(path: str):
    with open(path) as f:
        d = json.load(f)
    return d["facts"] if isinstance(d, dict) else d


def check_answer_collisions(facts):
    """Meta claims answers are unique w/o substring collisions — verify, don't trust."""
    bad = []
    norm = [(f["id"], " ".join(str(f["answer"]).lower().split())) for f in facts]
    for i, (id_a, a) in enumerate(norm):
        for id_b, b in norm[i + 1:]:
            if a in b or b in a:
                bad.append((id_a, id_b, a, b))
    return bad


def contains_match_ci(answer: str, generation: str) -> bool:
    a = " ".join(str(answer).lower().split())
    g = " ".join(str(generation).lower().split())
    return a in g


@torch.no_grad()
def probe_recall(model, tok, facts, device="cuda:0", max_new_tokens=32, raw=False):
    """READ-ONLY recall probe (chat-templated; raw=True = diagnostic fallback only).
    Returns list of {id, generation, match}."""
    was_training = model.training
    model.eval()
    out = []
    for f in facts:
        prompt = build_query_prompt(tok, f["question"], raw=raw)
        ids = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"id": f["id"], "generation": text,
                    "match": contains_match_ci(f["answer"], text)})
    if was_training:
        model.train()
    return out


def make_optimizer(model, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)  # no warmup, no schedule (per-turn online)


def write_step(model, tok, optimizer, statement: str, device="cuda:0", grad_clip=1.0):
    """One gradient step of LM loss on the RAW statement string (A4: no chat wrapper —
    keeps write→recall a statement→question generalization). Returns loss."""
    model.train()
    text = WRITE_TEMPLATE.format(statement=statement)
    batch = tok(text, return_tensors="pt").to(device)
    out = model(**batch, labels=batch["input_ids"])
    optimizer.zero_grad()
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], grad_clip)
    optimizer.step()
    return float(out.loss)


# ---------------- GSM8K (frozen subset; 0-shot CoT; exact numeric match) -------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str):
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm8k_gold(answer_field: str):
    return answer_field.split("####")[-1].strip().replace(",", "")


@torch.no_grad()
def eval_gsm8k(model, tok, items, device="cuda:0", max_new_tokens=512):
    """READ-ONLY (frozen snapshot). Chat-templated 0-shot CoT (A4), non-thinking.
    max_new_tokens=512 per AMENDMENT A7 (2026-06-12): the prior 256 cap truncated the
    verbose 2B mid-CoT and scored spurious numbers; all pre-A7 GSM8K numbers superseded.
    items: list of {question, answer}. Returns accuracy."""
    was_training = model.training
    model.eval()
    correct = 0
    for it in items:
        prompt = apply_chat(tok, GSM8K_USER_MSG.format(q=it["question"]))
        ids = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if extract_final_number(text) == gsm8k_gold(it["answer"]):
            correct += 1
    if was_training:
        model.train()
    return correct / len(items)


def load_gsm8k_subset(ids_path: str):
    from datasets import load_dataset
    with open(ids_path) as f:
        frozen = json.load(f)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [ds[i] for i in frozen["indices"]]
