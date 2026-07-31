#!/usr/bin/env python3
"""GREP-RESCUE probe (CLAIMS queue 7; pre-registered 2026-07-22, user design).

In-weight memory as INDEX over an external timestamped log:
  top-k first tokens -> greedy candidates (queue-6 machinery) = retrieval KEYS
  grep log (1 gold write + 2 never-trained same-attr distractors per fact, timestamped,
  gold latest; --stale flips half) with keys (index arm) / attr (question arm, ceiling control)
  present retrieved lines in-context -> re-ask cloze -> greedy    = rescued_grep
  Δlp on each retrieved line's value (presented-external regime)  = rescued_auth
Logged controls: adapter-off grep answer (pure-RAG floor), lpA rescoring (queue-6 +1 baseline).
"""
import argparse, json, os, random, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L
_argv = sys.argv; sys.argv = ["phase5_accum.py"]
from phase5_accum import make_facts, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--stale", action="store_true")  # even fids: a distractor gets a LATER timestamp than gold (recency vs familiarity dissociation)
ap.add_argument("--dev", default="cuda:1")
args = ap.parse_args()
DEV = args.dev
L.check_env()

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
sd = torch.load(args.adapter, map_location=DEV)
res = model.load_state_dict(sd, strict=False)
model.eval()
print(f"[grep] adapter loaded unexpected={len(res.unexpected_keys)}", flush=True)

facts = make_facts(args.n, args.seed)
gold_set = {f["answer"] for f in facts}

# --- external log: per fact 1 gold line + 2 never-trained distractor values (same attr) ---
syl = ["zor", "vex", "lun", "qua", "mip", "tar", "nye", "blu", "gro", "fen", "wix", "dap",
       "sol", "kee", "ral", "tun", "vop", "jiz", "mol", "pez", "fyx", "gub", "hox", "lid"]
def attr_of(f):
    s = f["statement"]  # "The user's {attr} is {V}."
    return s[len("The user's "):s.find(" is ")]

log_lines = []  # dicts: fid, value, gold, day
seen_vals = set(gold_set)
for f in facts:
    r = random.Random(9000 + f["fid"] + 100003 * args.seed)
    dvals = []
    while len(dvals) < 2:
        v = "".join(s.capitalize() for s in r.sample(syl, 3))
        if v not in seen_vals:
            seen_vals.add(v); dvals.append(v)
    days = [1, 2, 3]  # distractor1, distractor2, gold(latest)
    if args.stale and f["fid"] % 2 == 0:
        days = [1, 4, 3]  # distractor2 LATER than gold
    log_lines.append({"fid": f["fid"], "value": dvals[0], "gold": 0, "day": days[0]})
    log_lines.append({"fid": f["fid"], "value": dvals[1], "gold": 0, "day": days[1]})
    log_lines.append({"fid": f["fid"], "value": f["answer"], "gold": 1, "day": days[2]})
random.Random(args.seed).shuffle(log_lines)

def render(ln, f_by_id):
    return f"[2026-07-{ln['day']:02d}] The user's {attr_of(f_by_id[ln['fid']])} is {ln['value']}."

f_by_id = {f["fid"]: f for f in facts}

@torch.no_grad()
def slot_logits(prompt):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    return model(**ids).logits[0, -1]

@torch.no_grad()
def greedy_from(prompt, max_new=10, force_first=None, use_adapter=True):
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].to(DEV)
    outp = []
    if force_first is not None:
        ids = torch.cat([ids, torch.tensor([force_first], device=DEV)])
        outp.append(force_first)
    for _ in range(max_new):
        if use_adapter:
            nxt = int(model(input_ids=ids[None]).logits[0, -1].argmax())
        else:
            with model.disable_adapter():
                nxt = int(model(input_ids=ids[None]).logits[0, -1].argmax())
        if nxt == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([nxt], device=DEV)])
        outp.append(nxt)
        if "." in tok.decode([nxt]) or "\n" in tok.decode([nxt]):
            break
    return tok.decode(outp, skip_special_tokens=True).strip().rstrip(".")

@torch.no_grad()
def answer_lp(prompt, ans, use_adapter):
    full = tok(prompt + " " + ans + ".", return_tensors="pt").to(DEV)
    npre = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    if use_adapter:
        logits = model(**full).logits[0]
    else:
        with model.disable_adapter():
            logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())

rows = []
for f in facts:
    p = cloze(f)
    gold = f["answer"]
    gold_first = tok(" " + gold, add_special_tokens=False)["input_ids"][0]
    logits = slot_logits(p)
    order = torch.argsort(logits, descending=True)
    grank = int((order == gold_first).nonzero()[0]) + 1
    g = greedy_from(p)
    hit = int(L.contains_match_ci(gold, g))
    others = [x["answer"] for x in facts if x["fid"] != f["fid"]]
    dis = random.Random(31 * f["fid"] + args.seed).choice(others)
    marg = answer_lp(p, gold, True) - answer_lp(p, dis, True)
    # keys: top-k first tokens -> candidates (+ lpA rescoring baseline, queue-6)
    cands = []
    for t in order[: args.topk].tolist():
        c = greedy_from(p, force_first=t)
        if c and c not in cands:
            cands.append(c)
    gold_in = any(L.contains_match_ci(gold, c) for c in cands)
    sel_lpa = max(cands, key=lambda c: answer_lp(p, c, True)) if cands else ""
    # retrieval: index keys = FULL log value contained in a candidate string
    # (v2 fix: v1 also matched candidate-inside-value; short frankenstein candidates
    #  substring-matched other facts' lines -> polluted context + global Δlp magnets)
    Ri = [ln for ln in log_lines if any(L.contains_match_ci(ln["value"], c) for c in cands)]
    Rq = [ln for ln in log_lines if ln["fid"] == f["fid"]]  # question-attr grep (trivially 3 lines: ceiling control)
    R = list({id(x): x for x in (Rq + Ri)}.values())
    R = sorted(R, key=lambda x: (x["day"], x["value"]))[:12]
    gold_in_Ri = any(ln["gold"] and ln["fid"] == f["fid"] for ln in Ri)
    prec_Ri = (sum(1 for ln in Ri if ln["fid"] == f["fid"]) / len(Ri)) if Ri else None
    excerpt = "\n".join(render(ln, f_by_id) for ln in R)
    # presented-mode answers, two formats (v2: "Using the log, complete:" triggered
    # instruction/think mode on this reasoning-tuned base -> off-policy, the paper-3 lesson):
    # cont = in-distribution continuation (log line for a later day, no instruction language)
    cp = excerpt + f"\n[2026-07-05] The user's {attr_of(f)} is"
    ans_cont = greedy_from(cp, use_adapter=True)
    ans_cont_base = greedy_from(cp, use_adapter=False)  # pure-RAG floor (no in-weight memory)
    # chat = native instruct interface (chat template, thinking off, neutral wording)
    msgs = [{"role": "user", "content": f"Here are log excerpts:\n{excerpt}\n\n"
             f"What is the user's {attr_of(f)}? Answer with only the value."}]
    cprompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    cinp = tok(cprompt, return_tensors="pt").to(DEV)
    with torch.no_grad():
        gout = model.generate(**cinp, max_new_tokens=24, do_sample=False)
        ans_chat = tok.decode(gout[0][cinp["input_ids"].shape[1]:], skip_special_tokens=True)
        with model.disable_adapter():
            gout = model.generate(**cinp, max_new_tokens=24, do_sample=False)
        ans_chat_base = tok.decode(gout[0][cinp["input_ids"].shape[1]:], skip_special_tokens=True)
    # recency-rule variant (v3): timestamp channel made explicit — dissociates from the
    # familiarity channel (they point apart on --stale facts)
    msgs_r = [{"role": "user", "content": f"Here are log excerpts:\n{excerpt}\n\n"
               f"What is the user's {attr_of(f)}? If entries conflict, the most recent entry "
               f"is correct. Answer with only the value."}]
    rprompt = tok.apply_chat_template(msgs_r, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    rinp = tok(rprompt, return_tensors="pt").to(DEV)
    with torch.no_grad():
        gout = model.generate(**rinp, max_new_tokens=24, do_sample=False)
        ans_rec = tok.decode(gout[0][rinp["input_ids"].shape[1]:], skip_special_tokens=True)
        with model.disable_adapter():
            gout = model.generate(**rinp, max_new_tokens=24, do_sample=False)
        ans_rec_base = tok.decode(gout[0][rinp["input_ids"].shape[1]:], skip_special_tokens=True)
    # authentication selector: Δlp over retrieved VALUES on the BARE cloze (no log in context)
    vals = sorted({ln["value"] for ln in R})
    auth = max(vals, key=lambda v: answer_lp(p, v, True) - answer_lp(p, v, False)) if vals else ""
    rows.append({
        "fid": f["fid"], "state": "recalled" if hit else ("recog_only" if marg > 0 else "gone"),
        "grank": grank, "greedy_hit": hit, "gold_in_cands": int(gold_in), "n_cands": len(cands),
        "rescued_lpa": int(L.contains_match_ci(gold, sel_lpa)),
        "gold_in_Ri": int(gold_in_Ri), "n_Ri": len(Ri), "prec_Ri": prec_Ri,
        "rescued_cont": int(L.contains_match_ci(gold, ans_cont)),
        "rescued_cont_base": int(L.contains_match_ci(gold, ans_cont_base)),
        "rescued_chat": int(L.contains_match_ci(gold, ans_chat)),
        "rescued_chat_base": int(L.contains_match_ci(gold, ans_chat_base)),
        "rescued_rec": int(L.contains_match_ci(gold, ans_rec)),
        "rescued_rec_base": int(L.contains_match_ci(gold, ans_rec_base)),
        "rescued_auth": int(L.contains_match_ci(gold, auth)),
        "ans_cont": ans_cont[:40], "ans_chat": ans_chat[:60], "ans_rec": ans_rec[:60],
        "auth": auth[:40]})
    r0 = rows[-1]
    print(f"[grep] f{f['fid']:2d} {r0['state']:10s} rank={grank:4d} hit={hit} inC={int(gold_in)} "
          f"inRi={int(gold_in_Ri)} cont={r0['rescued_cont']}/{r0['rescued_cont_base']} "
          f"chat={r0['rescued_chat']}/{r0['rescued_chat_base']} "
          f"auth={r0['rescued_auth']} lpa={r0['rescued_lpa']}", flush=True)

byst = {}
for s in ("recalled", "recog_only", "gone"):
    v = [r for r in rows if r["state"] == s]
    if v:
        byst[s] = {k: sum(r[k] for r in v) for k in
                   ("gold_in_cands", "gold_in_Ri", "rescued_cont", "rescued_cont_base",
                    "rescued_chat", "rescued_chat_base", "rescued_rec", "rescued_rec_base",
                    "rescued_auth", "rescued_lpa")}
        byst[s]["n"] = len(v)
        byst[s]["median_rank"] = sorted(r["grank"] for r in v)[len(v) // 2]
summary = {"adapter": os.path.basename(args.adapter), "topk": args.topk, "stale": args.stale,
           "n": len(rows), "greedy_recall": sum(r["greedy_hit"] for r in rows),
           "rescued_lpa": sum(r["rescued_lpa"] for r in rows),
           "rescued_cont": sum(r["rescued_cont"] for r in rows),
           "rescued_cont_base": sum(r["rescued_cont_base"] for r in rows),
           "rescued_chat": sum(r["rescued_chat"] for r in rows),
           "rescued_chat_base": sum(r["rescued_chat_base"] for r in rows),
           "rescued_rec": sum(r["rescued_rec"] for r in rows),
           "rescued_rec_base": sum(r["rescued_rec_base"] for r in rows),
           "rescued_auth": sum(r["rescued_auth"] for r in rows),
           "gold_in_Ri": sum(r["gold_in_Ri"] for r in rows), "by_state": byst}
odir = os.path.join(os.path.dirname(__file__), "results", "greprescue")
os.makedirs(odir, exist_ok=True)
tag = "_stale" if args.stale else ""
out = os.path.join(odir, f"greprescue_v3{tag}_{os.path.basename(args.adapter).replace('.pt','')}.json")
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
