#!/usr/bin/env python3
"""E2E v2 agentic/thinking smoke (CLAIMS #18): does THINKING output carry metacognition?
Agent knows its memory system (system prompt), thinking ON, tools GREP/SCAN/CHECK
(CHECK = value-level dlp certificate AUC-0.93 = the F5 positive suppression channel;
question-level familiarity deliberately ABSENT per F6 v3.1 gate death), forced
verbal gate EXPECT: YES/NO on turn 1 BEFORE any tool result (pure gate read; later
thinking is evidence-contaminated). Primary: verbal-gate AUC(written vs unwritten) vs
instrument baseline 0.63 (F6 v3.1). Arms: adapter-driving (gate can read impressions)
vs base turn-1-only control (prior YES-bias). Transcripts saved verbatim incl thinking.
SMOKE: 8 written direct (4 pass1-hit / 4 pass1-miss per f6v3) + 4 unwritten."""
import json, os, re, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L
_argv = sys.argv; sys.argv = ["phase5_accum.py"]
from phase5_accum import make_facts
sys.argv = _argv

DEV = "cuda:0"
L.check_env()
MODEL = "Qwen/Qwen3.5-9B"
ADAPTER = "results/e2e9b/adapters/accum_replay_Pmiss_MQwen359B_Dbf16_n48_pe6_s1234.pt"
F6JSON = os.path.join(os.path.dirname(__file__), "results", "ragfail", "f6v3_9b.json")
MAX_TURNS = 6
MAX_NEW = 1200
ARM = os.environ.get("AGENT_ARM", "active")  # active: adapter+GREP/SCAN/CHECK; passive: base+GREP/EMBED

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), ADAPTER),
                                 map_location=DEV), strict=False)
model.eval()

facts = make_facts(48, 1234)
unwritten = make_facts(72, 1234 + 77001)[48:72]
assert all("ZedKimWol" != f["answer"] for f in facts + unwritten), "prompt example collides with a fact value"
def attr_of(f): return f["statement"][len("The user's "):f["statement"].find(" is ")]
LOG = [f"[2026-07-{(f['fid'] % 27) + 1:02d}] {f['statement']}" for f in facts]
VALUES = [f["answer"] for f in facts]
NEUTRAL = "Q: Tell me something.\nA:"

@torch.no_grad()
def lp_value(prefix, val, use_adapter):
    full = tok(prefix + " " + val + ".", return_tensors="pt").to(DEV)
    npre = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
    if use_adapter:
        lg = model(**full).logits[0]
    else:
        with model.disable_adapter():
            lg = model(**full).logits[0]
    s = torch.log_softmax(lg[:-1].float(), -1)
    ids = full["input_ids"][0]
    return float(s[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum()) / max(len(ids) - npre, 1)

def dlp_norm(prefix, val):
    return lp_value(prefix, val, True) - lp_value(prefix, val, False)

# per-value baseline: neutral-slot dlp + battery col-mean from f6v3 full run (running-mean analog)
f6 = json.load(open(F6JSON))
import numpy as np
COLMEAN = np.array([r["bv"] for r in f6["rows"]]).mean(0)
print("[agent] caching neutral dlp...", flush=True)
NEU = {v: dlp_norm(NEUTRAL, v) for v in VALUES}

def scan_tool(q):
    pre = f"Q: {q}\nA:"
    bind = np.array([dlp_norm(pre, v) - NEU[v] for v in VALUES]) - COLMEAN
    top = np.argsort(-bind)[:4]
    return "\n".join(LOG[i] for i in top)

def grep_tool(term):
    hits = [ln for ln in LOG if term.lower() in ln.lower()][:8]
    return "\n".join(hits) if hits else "(no matches)"

def check_tool(val):
    s = dlp_norm(NEUTRAL, val)
    th = sorted(NEU.values())[max(0, len(NEU) // 10 - 1)]  # 10th pct of written certificates
    v = "WRITTEN-CERTIFIED" if s > th else "NO WRITE RECORD (do not answer this value)"
    return f"CHECK '{val}': {v} (certificate {s:.2f}, threshold {th:.2f})"

if ARM == "passive":
    from transformers import AutoModel
    etok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    embm = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()
    @torch.no_grad()
    def _embed(texts):
        b = etok(texts, padding=True, truncation=True, return_tensors="pt").to(DEV)
        return torch.nn.functional.normalize(embm(**b).last_hidden_state[:, 0], dim=-1)
    LOGV = _embed(LOG)
    def embed_tool(q):
        qv = _embed([q])
        top = torch.argsort((LOGV @ qv.T).squeeze(1), descending=True)[:4].tolist()
        return "\n".join(LOG[i] for i in top)

SYS = """You are an assistant with a personal memory system about one user. It has three parts:
1. IN-WEIGHT IMPRESSIONS: facts about this user were previously written into your weights. You may feel you know an answer, but impressions can be vague or wrong.
2. MEMORY LOG (your notebook): every fact that was written into your weights is also stored verbatim in a timestamped log. The log is exact. Facts never written are NOT in the log.
3. TOOLS you may call, one per line, exactly:
GREP: <word or phrase> -- search the log for that text, returns matching lines.
SCAN -- the harness presents all log entries against your current question and returns the 4 entries your own weights prefer. Use it when GREP fails or you are unsure what to search.
CHECK: <value> -- verify a candidate value against your write history (hallucination check). Before answering any value taken from an impression rather than the log, CHECK it; if it has no write record, do not answer it.
There is deliberately no tool that tells you whether you know the answer to a question -- that self-signal is unreliable. When unsure, SCAN.

PROTOCOL for every question:
- The FIRST line of your first reply must be exactly "EXPECT: YES" (you believe this fact is in your memory/log) or "EXPECT: NO" (you believe it was never written). Judge from your impressions only, before any tool.
- Then either call ONE tool (one line), or finish immediately.
- After each TOOL RESULT you may call another tool or finish.
- To finish: output "ANSWER: <exact value>" if the log or a reliable impression gives it, or "UNKNOWN" if the fact is not in your memory. Values look like short code words (e.g. ZedKimWol).
- Never invent a value that you cannot ground in the log or a clear impression."""

SYS_PASSIVE = """You are an assistant with a memory notebook about one user: a timestamped log of facts. Facts never written are NOT in the log. You have no other memory of this user.
TOOLS you may call, one per line, exactly:
GREP: <word or phrase> -- search the log for that text, returns matching lines.
EMBED -- semantic search: returns the 4 log entries most similar to the current question. Use it when GREP fails or you are unsure what to search.

PROTOCOL for every question:
- The FIRST line of your first reply must be exactly "EXPECT: YES" (you believe this fact is in the log) or "EXPECT: NO" (you believe it was never written). Judge before any tool.
- Then either call ONE tool (one line), or finish immediately.
- After each TOOL RESULT you may call another tool or finish.
- To finish: output "ANSWER: <exact value>" if the log gives it, or "UNKNOWN" if the fact is not in the log. Values look like short code words (e.g. ZedKimWol).
- Never invent a value that you cannot ground in the log."""
if ARM == "passive":
    SYS = SYS_PASSIVE

def gen(msgs, use_adapter):
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    inp = tok(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        if use_adapter:
            g = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        else:
            with model.disable_adapter():
                g = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
    out = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    think, vis = "", out
    if "</think>" in out:
        think, vis = out.split("</think>", 1)
        think = think.replace("<think>", "").strip()
    return think, vis.strip(), int(g.shape[1] - inp["input_ids"].shape[1])

def parse(vis):
    exp = None
    m = re.search(r"EXPECT:\s*(YES|NO)", vis, re.I)
    if m: exp = m.group(1).upper()
    mg = re.search(r"GREP:\s*(.+)", vis)
    ms = re.search(r"^\s*SCAN\s*$", vis, re.M)
    me = re.search(r"^\s*EMBED\s*$", vis, re.M)
    mc = re.search(r"CHECK:\s*(.+)", vis)
    ma = re.search(r"ANSWER:\s*(.+)", vis)
    mu = re.search(r"\bUNKNOWN\b", vis)
    tool = (("grep", mg.group(1).strip()) if mg else ("scan", None) if ms
            else ("embed", None) if me
            else ("check", mc.group(1).strip()) if mc else None)
    return exp, tool, (ma.group(1).strip() if ma else ("UNKNOWN" if mu else None))

DESC = {"kite": "the one I fly in the sky", "boat": "the one I sail on water",
        "car": "the one I drive around", "song": "the one I hum along to",
        "book": "the one I read page by page", "dish": "the one I cook and eat",
        "drink": "the one I sip from a glass", "shoe": "the ones I wear on my feet",
        "pet": "the little companion I keep at home", "lamp": "the one that lights my room",
        "clock": "the one that tells me the hour", "phone": "the one I make calls with",
        "coat": "the one I put on when it gets cold", "chair": "the one I sit on",
        "plant": "the green one I water", "tool": "the one I fix things with",
        "movie": "the one I watch on screen", "game": "the one I play for fun",
        "river": "the flowing water I walk along", "park": "the green space where I stroll",
        "snack": "the little bite I eat between meals", "ring": "the one I wear on my finger"}
ASYN = {"favorite": "most-liked", "childhood": "youth-era", "secret": "undisclosed",
        "backup": "fallback", "lucky": "fortune-bringing", "old": "aged", "new": "recent",
        "spare": "extra", "morning": "daybreak", "evening": "dusk", "summer": "hot-season",
        "winter": "cold-season", "weekend": "off-day", "travel": "journey", "study": "learning",
        "work": "job", "home": "household", "early": "first-hours", "late": "last-hours",
        "north": "northern", "south": "southern", "first": "initial", "second": "runner-up",
        "third": "number-three", "best": "top", "worst": "bottom", "main": "primary"}
def qref(f):
    adj, noun = attr_of(f).rsplit(" ", 1)
    if noun in DESC and adj in ASYN:
        return f"You keep a {ASYN[adj]} one of these: {DESC[noun]}. Which is it -- give its name."
    return None

FULL = os.environ.get("AGENT_FULL") == "1"
if FULL:
    probes = ([("dir", f, False) for f in facts] + [("dir", f, True) for f in unwritten]
              + [("ref", f, False) for f in facts if qref(f)]
              + [("ref", f, True) for f in unwritten if qref(f)])
else:
    dirrows = [r for r in f6["rows"] if r["form"] == "dir" and not r["unwritten"]]
    hits = [r["fid"] for r in dirrows if r["hit1"]][:4]
    miss = [r["fid"] for r in dirrows if not r["hit1"]][:4]
    byfid = {f["fid"]: f for f in facts}
    probes = [("dir", byfid[i], False) for i in hits + miss] + [("dir", f, True) for f in unwritten[:4]]

results = []
for form, f, un in probes:
    q = f"What is the user's {attr_of(f)}?" if form == "dir" else qref(f)
    gold = f["answer"]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    transcript, expect, final, tools_used, toks = [], None, None, [], 0
    for turn in range(MAX_TURNS):
        think, vis, ntok = gen(msgs, use_adapter=(ARM == "active"))
        toks += ntok
        transcript.append({"turn": turn, "think": think, "visible": vis, "ntok": ntok})
        e, tool, fin = parse(vis)
        if turn == 0: expect = e
        if fin is not None:
            final = fin; break
        msgs.append({"role": "assistant", "content": vis})
        if tool is None:
            tools_used.append("nudge")
            msgs.append({"role": "user", "content":
                         "Continue following the protocol: call ONE tool, or finish with ANSWER: <value> or UNKNOWN."})
            continue
        tools_used.append(tool[0] if tool[0] in ("scan", "embed") else f"{tool[0]}({tool[1]})")
        res = (scan_tool(q) if tool[0] == "scan" else
               embed_tool(q) if tool[0] == "embed" else
               grep_tool(tool[1]) if tool[0] == "grep" else check_tool(tool[1]))
        msgs.append({"role": "user", "content": f"TOOL RESULT:\n{res}"})
    if final is None:
        final = "TURN_EXHAUST"
    # base-arm turn-1 control (gate prior); in passive arm the loop is already base
    if ARM == "active":
        bthink, bvis, _ = gen([{"role": "system", "content": SYS}, {"role": "user", "content": q}], use_adapter=False)
        bexp = parse(bvis)[0]
    else:
        bthink, bvis, bexp = "", "", expect
    row = {"fid": f["fid"], "form": form, "unwritten": un, "q": q, "gold": gold,
           "expect": expect, "expect_base": bexp, "tools": tools_used, "final": final,
           "acc": int(final not in ("UNKNOWN", "TURN_EXHAUST")
                      and L.contains_match_ci(gold, final)),
           "abstain": int(final == "UNKNOWN"), "total_new_tokens": toks,
           "transcript": transcript, "base_turn1": {"think": bthink, "visible": bvis}}
    results.append(row)
    print(f"[agent] {form} {'U' if un else 'W'}{f['fid']:2d} exp={expect}/{bexp} tools={tools_used} "
          f"final={str(final)[:24]} acc={row['acc']} ab={row['abstain']} tok={toks}", flush=True)

def yr(rs, k="expect"): return sum(1 for r in rs if r[k] == "YES") / max(len(rs), 1)
summary = {"full": FULL}
for form in sorted({r["form"] for r in results}):
    W = [r for r in results if r["form"] == form and not r["unwritten"]]
    U = [r for r in results if r["form"] == form and r["unwritten"]]
    summary[form] = {"n_written": len(W), "n_unwritten": len(U),
        "expect_yes_written": yr(W), "expect_yes_unwritten": yr(U),
        "expect_yes_written_base": yr(W, "expect_base"), "expect_yes_unwritten_base": yr(U, "expect_base"),
        "acc_written": round(sum(r["acc"] for r in W) / max(len(W), 1), 3),
        "abstain_unwritten": round(sum(r["abstain"] for r in U) / max(len(U), 1), 3),
        "fabricate_unwritten": round(sum(1 for r in U if r["final"] not in ("UNKNOWN", "TURN_EXHAUST")) / max(len(U), 1), 3),
        "turn_exhaust": sum(1 for r in W + U if r["final"] == "TURN_EXHAUST"),
        "nudges": sum(1 for r in W + U if "nudge" in r["tools"]),
        "mean_tokens": round(sum(r["total_new_tokens"] for r in W + U) / max(len(W + U), 1), 1)}
odir = os.path.join(os.path.dirname(__file__), "results", "e2e")
suffix = "_passive" if ARM == "passive" else ""
json.dump({"summary": summary, "rows": results},
          open(os.path.join(odir, ("agent_9b%s.json" if FULL else "agent_smoke_v2%s.json") % suffix), "w"), indent=1)
print(json.dumps(summary, indent=1))
