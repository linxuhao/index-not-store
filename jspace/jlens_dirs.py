#!/usr/bin/env python3
"""Cache J-lens answer-slot directions for the timeline experiment (CLAIMS queue 5).

Computes, on the FROZEN base, dir_l[v] = E_refcloze[ d logit_v(final,last) / d h_l(last) ]
for the 48 written facts' first answer tokens, plus the two pre-registered null controls'
material: (n1) matched-norm random directions (same per-token per-layer norm, seeded);
(n2) mismatch pairing is a fixed derangement stored here (own-vs-other-token projection).
Saves results/jspace/dirs_s<seed>.pt : {"toks": [...], "dirs": {tok: [nl,d]}, "rand": {...},
"derange": {fid: fid'}, "corpus": n, "variant": "answer-slot"}.
"""
import argparse, os, random, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e0"))
import e0_lib as L
_argv = sys.argv; sys.argv = ["phase5_accum.py"]
from phase5_accum import make_facts, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--corpus", type=int, default=64)
ap.add_argument("--dev", default="cuda:1")
args = ap.parse_args()
DEV = args.dev
L.check_env()

tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
base.get_input_embeddings().weight.requires_grad_(True)

facts = make_facts(args.n, args.seed)
ref = make_facts(args.corpus, args.seed + 90001)
def first_tok(ans):
    return tok(" " + ans, add_special_tokens=False)["input_ids"][0]
fact_toks = [first_tok(f["answer"]) for f in facts]
probe_toks = sorted(set(fact_toks))

layers = base.model.layers
nl = len(layers)
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

dirs = {v: torch.zeros(nl, base.config.hidden_size) for v in probe_toks}
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
        print(f"[dirs] {pi}/{len(ref)}", flush=True)

g = torch.Generator().manual_seed(args.seed + 4242)
rand = {v: torch.randn(nl, base.config.hidden_size, generator=g) for v in probe_toks}
for v in probe_toks:  # matched norm per layer (null n1)
    rand[v] = rand[v] / rand[v].norm(dim=1, keepdim=True) * dirs[v].norm(dim=1, keepdim=True)

fids = [f["fid"] for f in facts]
order = fids[:]
rng = random.Random(args.seed + 777)
while True:
    rng.shuffle(order)
    if all(a != b for a, b in zip(fids, order)):
        break
derange = dict(zip(fids, order))  # null n2: project onto another fact's token dir

os.makedirs(os.path.join(os.path.dirname(__file__), "results", "jspace"), exist_ok=True)
out_p = os.path.join(os.path.dirname(__file__), "results", "jspace", f"dirs_s{args.seed}.pt")
torch.save({"toks": probe_toks, "fact_toks": fact_toks, "dirs": dirs, "rand": rand,
            "derange": derange, "corpus": len(ref), "variant": "answer-slot", "nl": nl}, out_p)
print(f"[dirs] saved {out_p}")
