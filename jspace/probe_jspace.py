#!/usr/bin/env python3
"""J-space smoke (CLAIMS queue 5, user GO 2026-07-19): read vs recognition vs J-projection.

J-lens ANSWER-SLOT VARIANT (documented deviation from transformer-circuits/2026/workspace:
their J_l averages over all t'>=t across ~1k diverse prompts; we take t'=t=last cloze token,
averaged over a reference cloze corpus — template-specific concept directions, which is what
our instrument needs and what 16GB affords).

Phase J1: on the FROZEN BASE (no adapter), for each probe token v (first answer token),
  dir_l[v] = E_prompts[ d logit_v(final, last) / d h_l(last) ]  — one backward per (prompt, v),
  params frozen (no param grads), gradients only w.r.t. captured hidden states.
Phase J2: mount a saved accum adapter ("mem", r64 all-linear); for written(48) + unwritten(48)
  facts, capture h_l at the cloze answer slot, score cos(h_l, dir_l[v_f]); AUC written-vs-
  unwritten per layer, adapter ON vs OFF (base zero-point).
"""
import argparse, json, os, sys

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
ap.add_argument("--corpus", type=int, default=64, help="reference cloze prompts for J averaging")
ap.add_argument("--dev", default="cuda:1")
args = ap.parse_args()
DEV = args.dev
L.check_env()

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)

written = make_facts(args.n, args.seed)
unwritten = make_facts(args.n, args.seed + 77001)
ref = make_facts(args.corpus, args.seed + 90001)

def first_tok(ans):
    return tok(" " + ans, add_special_tokens=False)["input_ids"][0]

probe_toks = sorted({first_tok(f["answer"]) for f in written + unwritten})
# graph must reach the hooks: params stay frozen except embeddings (grad buffer ~1.2GB, zeroed per prompt)
base.get_input_embeddings().weight.requires_grad_(True)

# ---- capture hidden states via hooks on each decoder layer output
layers = base.model.layers
caps = {}
def mk_hook(i):
    def h(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if t.requires_grad:
            t.retain_grad()
        caps[i] = t
        return out
    return h
hooks = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(layers)]

# ---- J1: directions on frozen base
nl = len(layers)
dirs = {v: [torch.zeros(base.config.hidden_size) for _ in range(nl)] for v in probe_toks}
for pi, f in enumerate(ref):
    ids = tok(cloze(f), return_tensors="pt").to(DEV)
    with torch.enable_grad():
        out = base(**ids)
        for v in probe_toks:
            for c in caps.values():
                if c.grad is not None:
                    c.grad = None
            out.logits[0, -1, v].backward(retain_graph=True)
            for i in range(nl):
                dirs[v][i] += caps[i].grad[0, -1].detach().cpu() / len(ref)
        base.get_input_embeddings().weight.grad = None
    caps.clear()
    if pi % 16 == 0:
        print(f"[J1] {pi}/{len(ref)}", flush=True)
print("[J1] directions done", flush=True)

# ---- J2: adapter mounted, project activations
cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules="all-linear",
                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg, adapter_name="mem")
sd = torch.load(args.adapter, map_location=DEV)
res = model.load_state_dict(sd, strict=False)
print(f"[J2] adapter loaded, missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
model.eval()

@torch.no_grad()
def collect(facts, use_adapter):
    ctx = torch.no_grad()
    rows = []
    for f in facts:
        ids = tok(cloze(f), return_tensors="pt").to(DEV)
        caps.clear()
        if use_adapter:
            model(**ids)
        else:
            with model.disable_adapter():
                model(**ids)
        v = first_tok(f["answer"])
        rows.append({"fid": f["fid"], "tok": v, "scores": [
            float(torch.nn.functional.cosine_similarity(
                caps[i][0, -1].detach().cpu(), dirs[v][i], dim=0)) for i in range(nl)]})
    return rows

def auc(pos, neg):
    import itertools
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))

out = {"adapter": args.adapter, "seed": args.seed, "variant": "answer-slot J-lens",
       "corpus": len(ref), "layers": nl}
for state, use in (("adapter_on", True), ("base_off", False)):
    W = collect(written, use); U = collect(unwritten, use)
    aucs = [auc([w["scores"][i] for w in W], [u["scores"][i] for u in U]) for i in range(nl)]
    out[state] = {"auc_per_layer": aucs, "best_layer": int(max(range(nl), key=lambda i: aucs[i])),
                  "best_auc": max(aucs),
                  "written": W if state == "adapter_on" else None}
    print(f"[J2] {state}: best AUC {max(aucs):.3f} @L{out[state]['best_layer']} "
          f"(L0..{nl-1}: {['%.2f' % a for a in aucs[::4]]} every4th)", flush=True)

os.makedirs(os.path.join(os.path.dirname(__file__), "results", "jspace"), exist_ok=True)
name = os.path.basename(args.adapter).replace(".pt", "")
with open(os.path.join(os.path.dirname(__file__), "results", "jspace", f"jspace_smoke_{name}.json"), "w") as f:
    json.dump(out, f, indent=1)
print("[done]", flush=True)
