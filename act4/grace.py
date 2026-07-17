#!/usr/bin/env python3
"""GRACE-style lifelong-editing baseline on the day-scale stream (pre-registered 2026-07-13).

Minimal faithful re-implementation of the GRACE mechanism (Hartvigsen et al., NeurIPS 2023):
a discrete key-value codebook grafted onto ONE Linear layer. key = the layer's input activation
at the answer position; value = a learned output offset; at inference the offset is added at any
position whose activation falls within a fixed deferral radius eps of a stored key.
Simplifications vs the official GRACE (noted in the paper): fixed eps (no split/expand), one entry
per fact (our facts are collision-free), value optimized by plain Adam.

Stream/probes identical to accum: same make_facts, same greedy-cloze recall, same 2-AFC
recognition, same probe cadence, same filename discipline. Budgets: --steps 8 (matched to ws=8)
or larger (canonical-ish upper bound).
"""
import argparse, json, os, random, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lib as L

_argv = sys.argv
sys.argv = ["accum.py"]      # accum parses args at import
from accum import make_facts, make_counterfact, cloze
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--n-stream", type=int, default=48)
ap.add_argument("--steps", type=int, default=8)          # value-optimization steps per fact (8 = matched budget)
ap.add_argument("--lr", type=float, default=0.5)         # direct value-vector optimization
ap.add_argument("--eps", type=float, default=None)       # deferral radius; default = auto (median key-dist / 2)
ap.add_argument("--layer-frac", type=float, default=0.75)  # wrapped layer depth (expression gradient: late third)
ap.add_argument("--facts", choices=["synthetic", "counterfact"], default="synthetic")
ap.add_argument("--probe-every", type=int, default=6)
ap.add_argument("--firewall-n", type=int, default=0)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
args = ap.parse_args()
DEV = args.dev

L.check_env(); torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


class GraceLinear(torch.nn.Module):
    """Wraps a Linear; adds stored value offsets where input activation is within eps of a key."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.keys, self.vals = [], []      # lists of tensors [d_in], [d_out]
        self.eps = None
        self.enabled = True
        self.train_val = None              # value tensor being optimized (applied at key match too)
        self.train_idx = None              # codebook index the train_val stands in for

    def forward(self, x):
        y = self.base(x)
        if not self.enabled or (not self.keys and self.train_val is None):
            return y
        flat = x.reshape(-1, x.shape[-1]).float()
        if self.keys:
            K = torch.stack(self.keys)                                    # [n, d_in] fp32
            d = torch.cdist(flat.unsqueeze(0), K.unsqueeze(0)).squeeze(0)  # [T, n]
            mind, argk = d.min(dim=1)
            hit = mind < self.eps
            if hit.any():
                V = torch.stack([self.train_val if (self.train_val is not None and i == self.train_idx)
                                 else self.vals[i] for i in range(len(self.vals))])
                add = torch.zeros_like(y).reshape(-1, y.shape[-1])
                add[hit] = V[argk[hit]].to(y.dtype)
                y = y + add.reshape(y.shape)
        return y


# locate + wrap the target Linear (down_proj of the layer at layer_frac depth)
layers = model.model.layers
li = min(len(layers) - 1, int(len(layers) * args.layer_frac))
target = layers[li].mlp.down_proj
g = GraceLinear(target).to(DEV)
layers[li].mlp.down_proj = g
print(f"[grace] wrapped model.layers[{li}].mlp.down_proj "
      f"(depth {li+1}/{len(layers)}, d_in={target.in_features}, d_out={target.out_features})")

_capture = {}


def key_activation(f):
    """Input activation to the wrapped layer at the FIRST answer token of the statement."""
    full = tok(f["statement"], return_tensors="pt").to(DEV)
    npre = tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]
    feats = {}
    h = g.base.register_forward_pre_hook(lambda m, inp: feats.setdefault("x", inp[0].detach()))
    with torch.no_grad():
        was = g.enabled; g.enabled = False   # key from the CLEAN model (GRACE convention)
        model(**full); g.enabled = was
    h.remove()
    # key = LAST CLOZE TOKEN's activation (the position whose output PREDICTS the first answer
    # token — present during generation; npre itself doesn't exist until the answer is emitted)
    return feats["x"][0, npre - 1].float().clone()   # fp32: ROCm cdist lacks bf16


def write_fact(f):
    k = key_activation(f)
    if g.eps is not None and g.keys:
        d = torch.stack([torch.dist(k, kk) for kk in g.keys])
        if float(d.min()) < g.eps:           # deferral: reuse nearest entry
            idx = int(d.argmin())
            g.train_val = g.vals[idx].clone().requires_grad_(True)
            g.train_idx = idx
            reuse = idx
        else:
            g.keys.append(k); g.vals.append(torch.zeros(g.base.out_features, device=DEV))
            g.train_val = torch.zeros(g.base.out_features, device=DEV, requires_grad=True)
            g.train_idx = len(g.vals) - 1
            reuse = None
    else:
        g.keys.append(k); g.vals.append(torch.zeros(g.base.out_features, device=DEV))
        g.train_val = torch.zeros(g.base.out_features, device=DEV, requires_grad=True)
        g.train_idx = len(g.vals) - 1
        reuse = None
    opt = torch.optim.Adam([g.train_val], lr=args.lr)
    b = tok(f["statement"], return_tensors="pt").to(DEV)
    labels = b["input_ids"].clone()
    labels[:, : tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]] = -100
    for _ in range(args.steps):
        out = model(**b, labels=labels)
        opt.zero_grad(); out.loss.backward(); opt.step()
    g.vals[g.train_idx] = g.train_val.detach()
    g.train_val = None
    g.train_idx = None
    return float(out.loss)


@torch.no_grad()
def answer_lp(f, ans):
    full = tok(cloze(f) + " " + ans + ".", return_tensors="pt").to(DEV)
    npre = tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]
    logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())


@torch.no_grad()
def probe(facts_so_far, all_facts):
    hits, margins = [], []
    for f in facts_so_far:
        ids = tok(cloze(f), return_tensors="pt").to(DEV)
        gen = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        comp = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
        hits.append(int(L.contains_match_ci(f["answer"], comp)))
        others = [x["answer"] for x in all_facts if x["fid"] != f["fid"]]
        dis = random.Random(31 * f["fid"] + args.seed).choice(others)
        margins.append(round(answer_lp(f, f["answer"]) - answer_lp(f, dis), 2))
    return hits, margins


facts = (make_counterfact(args.n_stream, args.seed) if args.facts == "counterfact"
         else make_facts(args.n_stream, args.seed))

# auto eps: half the median pairwise key distance over the first 12 facts (clean model)
if args.eps is None:
    ks = torch.stack([key_activation(f) for f in facts[:12]])
    d = torch.cdist(ks.unsqueeze(0), ks.unsqueeze(0)).squeeze(0)
    off = d[~torch.eye(len(ks), dtype=torch.bool, device=d.device)]
    g.eps = float(off.median()) / 2
    print(f"[grace] auto eps = {g.eps:.2f} (median clean key dist {float(off.median()):.2f})")
else:
    g.eps = args.eps

gs_items, gsm_base, fw = None, None, {}
if args.firewall_n > 0:
    gs_items = L.load_gsm8k_subset(os.path.join(os.path.dirname(__file__),
                                   "gsm8k_pilot_ids.json"))[: args.firewall_n]
    g.enabled = False
    gsm_base = L.eval_gsm8k(model, tok, gs_items, device=DEV)
    g.enabled = True
    print(f"[grace] firewall base GSM8K = {gsm_base:.2f}")

print(f"[grace] model={args.model} stream={args.n_stream} steps={args.steps} lr={args.lr} "
      f"eps={g.eps:.2f} layer={li} seed={args.seed}")

curve = []
for k, f in enumerate(facts):
    loss = write_fact(f)
    if (k + 1) % args.probe_every == 0 or k == len(facts) - 1:
        hits, margins = probe(facts[: k + 1], facts)
        cr = sum(hits) / len(hits)
        curve.append({"k": k + 1, "cumrecall": round(cr, 3), "n_recalled": round(cr * (k + 1), 1),
                      "hits": hits, "margins": margins,
                      "n_recognized": sum(1 for m in margins if m > 0),
                      "n_entries": len(g.keys)})
        print(f"[grace] k={k+1}: recalled {curve[-1]['n_recalled']}/{k+1} "
              f"recog {curve[-1]['n_recognized']} entries {len(g.keys)}")

final = curve[-1]
if args.facts == "counterfact":
    ph = []
    with torch.no_grad():
        for f in facts:
            if not f.get("para"):
                ph.append(None); continue
            ids = tok(f["para"], return_tensors="pt").to(DEV)
            gen = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
            comp = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
            ph.append(int(L.contains_match_ci(f["answer"], comp)))
    final["para_hits"] = ph
    print(f"[grace] paraphrase recall {sum(h for h in ph if h)}/{sum(1 for h in ph if h is not None)}")

if args.firewall_n > 0:
    g.enabled = False
    fw = {"gsm8k_base": gsm_base, "gsm8k_off": L.eval_gsm8k(model, tok, gs_items, device=DEV),
          "n": len(gs_items)}
    g.enabled = True
    print(f"[grace] FIREWALL base={fw['gsm8k_base']:.2f} -> off={fw['gsm8k_off']:.2f}")

out = {"mech": "grace", "model": args.model, "n_stream": args.n_stream, "steps": args.steps,
       "lr": args.lr, "eps": round(g.eps, 3), "layer": li, "layer_frac": args.layer_frac,
       "probe_every": args.probe_every, "seed": args.seed, "facts_src": args.facts,
       "n_entries": len(g.keys), "curve": curve, "firewall": fw,
       "final_n_recalled": final["n_recalled"], "final_n_recognized": final["n_recognized"],
       "final_cumrecall": final["cumrecall"]}
os.makedirs(args.out, exist_ok=True)
tag = f"_st{args.steps}_lr{args.lr:g}"
if args.model != "Qwen/Qwen3.5-2B":
    tag += "_M" + args.model.split("/")[-1].replace("-", "").replace(".", "")[:14]
if args.facts != "synthetic":
    tag += f"_F{args.facts}"
fn = f"accum_grace{tag}_n{args.n_stream}_pe{args.probe_every}_s{args.seed}.json"
json.dump(out, open(os.path.join(args.out, fn), "w"), indent=2)
print(f"[grace] saved {fn}")
