"""accum.py -- the streaming in-weight memory instrument (day scale).

Stream N distinct collision-free facts ONE PER TURN into a LoRA adapter on a frozen base (single-pass
online continual learning). After writing fact k, probe cumulative recall AND 2-AFC recognition of
facts 1..k. Does the recallable count grow (accumulation) or plateau? Compare mechanisms head-to-head:
  naked  : plain online LoRA writes (the floor)
  bf     : Benna-Fusi multi-timescale synaptic cascade (metaplasticity)
  ewc    : accumulated EWC (running summed Fisher + re-anchored theta*)
  local  : sparse per-fact masking (each fact writes a different ~local_frac subspace)
  replay : small replay buffer, uniform or --replay-policy miss (error-gated)
Every run also checks the capability firewall (--firewall-n>0: adapter-off GSM8K vs base).
--selftest-form question gates replay by an interrogative self-test independent of the eval cloze
(the circularity control); --save-adapter archives the end-of-stream LoRA state for the Act-4 probes.

Run:  python accum.py --mech ewcreplay --replay-policy miss --firewall-n 10 --seed 1234 --dev cuda:0
"""
import argparse, json, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--mech", choices=["naked", "bf", "ewc", "local", "replay", "ewcreplay"], default="naked")
ap.add_argument("--firewall-n", type=int, default=0)      # >0: GSM8K adapter-off at start+end (base intact?)
ap.add_argument("--n-stream", type=int, default=48)
ap.add_argument("--ws", type=int, default=8)              # writes per fact (E0 sweet spot)
ap.add_argument("--rank", type=int, default=64)           # day-core sweet spot
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--levels", type=int, default=32)         # BF cascade depth (best from N-sweep)
ap.add_argument("--dt", type=float, default=0.5)
ap.add_argument("--base-g", type=float, default=1.0)
ap.add_argument("--ewc-lambda", type=float, default=300.0)
ap.add_argument("--local-frac", type=float, default=0.15) # fraction of params a fact may write
ap.add_argument("--replay-m", type=int, default=4)        # small buffer: facts replayed per turn
ap.add_argument("--layers", default="all")  # all | early | mid | late (thirds) | "a-b" inclusive layer range
ap.add_argument("--replay-policy", choices=["uniform", "miss"], default="uniform")  # miss: spend replay on last-probe failures (error-gated re-instatement)
ap.add_argument("--selftest-form", choices=["same", "question"], default="same")  # question: gate replay by an interrogative-form self-test, independent of the eval cloze (circularity control)
ap.add_argument("--facts", choices=["synthetic", "counterfact"], default="synthetic")  # counterfact: real-entity counterfactual edits (azhx/counterfact) + free paraphrase probe
ap.add_argument("--probe-every", type=int, default=6)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--save-adapter", action="store_true")   # save end-of-stream LoRA state to results/adapters/ (recovery probes)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
args = ap.parse_args()
DEV = args.dev
N = args.levels
C = [2.0 ** k for k in range(N)]
G = [args.base_g * 2.0 ** (-k) for k in range(max(N - 1, 0))]
LP = None
HIDDEN = None


def make_facts(n, seed):
    adjs = ["favorite", "childhood", "secret", "backup", "lucky", "old", "new", "hidden", "spare",
            "morning", "evening", "summer", "winter", "weekend", "travel", "study", "work", "home",
            "early", "late", "north", "south", "first", "second", "third", "best", "worst", "main"]
    nouns = ["color", "city", "dish", "drink", "band", "movie", "book", "river", "gadget", "hobby",
             "snack", "song", "game", "park", "shoe", "plant", "tool", "street", "pet", "car",
             "phone", "chair", "lamp", "coat", "ring", "clock", "boat", "kite"]
    syl = ["zor", "vex", "lun", "qua", "mip", "tar", "nye", "blu", "gro", "fen", "wix", "dap",
           "sol", "kee", "ral", "tun", "vop", "jiz", "mol", "pez", "fyx", "gub", "hox", "lid"]
    rng = random.Random(seed)
    attrs = [f"{a} {nn}" for a in adjs for nn in nouns]; rng.shuffle(attrs)
    facts, seen = [], set()
    for i in range(n):
        r = random.Random(7000 + i + 100003 * seed)
        while True:
            v = "".join(s.capitalize() for s in r.sample(syl, 3))
            if v not in seen:
                seen.add(v); break
        facts.append({"fid": i, "statement": f"The user's {attrs[i]} is {v}.", "answer": v})
    assert len({f["answer"] for f in facts}) == n
    return facts


def make_counterfact(n, seed):
    """n CounterFact edits as stream facts: write the counterfactual target_new (base cannot
    produce it from prior knowledge -> recall requires the write). Unique subjects + answers;
    first paraphrase prompt kept for the end-of-stream paraphrase probe."""
    from datasets import load_dataset
    ds = load_dataset("azhx/counterfact", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds))); rng.shuffle(idx)
    facts, seen_ans, seen_subj = [], set(), set()
    for i in idx:
        rw = ds[i]["requested_rewrite"]
        subj, ans = rw["subject"], rw["target_new"]["str"]
        prompt = rw["prompt"].format(subj)
        if ans in seen_ans or subj in seen_subj or ans in prompt:
            continue
        para = (ds[i]["paraphrase_prompts"] or [None])[0]
        facts.append({"fid": len(facts), "statement": f"{prompt} {ans}.", "answer": ans, "para": para})
        seen_ans.add(ans); seen_subj.add(subj)
        if len(facts) == n:
            break
    assert len(facts) == n
    return facts


def cloze(f):
    return f["statement"][: f["statement"].find(f["answer"])].rstrip()


def pfx(tok, f):
    return tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]


def grad_step(opt, model, tok, f, ewc=None, mask=None, clip=1.0):
    model.train()
    b = tok(f["statement"], return_tensors="pt").to(DEV)
    labels = b["input_ids"].clone(); labels[:, :pfx(tok, f)] = -100
    out = model(**b, labels=labels); opt.zero_grad(); out.loss.backward()
    if ewc is not None:
        fisher, theta_star, lam = ewc
        for p, fi, ts in zip(LP, fisher, theta_star):
            if p.grad is not None:
                p.grad.add_(lam * fi * (p.detach() - ts))
    if mask is not None:
        for p, m in zip(LP, mask):
            if p.grad is not None:
                p.grad.mul_(m)
    torch.nn.utils.clip_grad_norm_(LP, clip); opt.step()
    return float(out.loss)


@torch.no_grad()
def cascade_step():
    if N < 2:
        return
    for i, p in enumerate(LP):
        u = [p.data] + HIDDEN[i]
        du = []
        for k in range(N):
            flow = None
            if k > 0:
                flow = G[k - 1] * (u[k - 1] - u[k])
            if k < N - 1:
                term = G[k] * (u[k + 1] - u[k])
                flow = term if flow is None else flow + term
            du.append((args.dt / C[k]) * flow)
        for k in range(N):
            u[k].add_(du[k])


def fisher_of(model, tok, f):
    """diagonal Fisher of one fact (value-token loss), mean-normalized to 1."""
    model.train(); model.zero_grad()
    b = tok(f["statement"], return_tensors="pt").to(DEV)
    labels = b["input_ids"].clone(); labels[:, :pfx(tok, f)] = -100
    model(**b, labels=labels).loss.backward()
    fi = [(p.grad.detach() ** 2 if p.grad is not None else torch.zeros_like(p)) for p in LP]
    tot = sum(x.sum() for x in fi); cnt = sum(x.numel() for x in fi)
    mean = (tot / cnt).clamp_min(1e-12)
    model.zero_grad()
    return [x / mean for x in fi]


def make_mask(seed_k):
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed_k)
    return [(torch.rand(p.shape, generator=gen) < args.local_frac).float().to(DEV) for p in LP]


@torch.no_grad()
def answer_lp(model, tok, f, ans):
    """Sum logprob of `ans` tokens as the cloze continuation (recognition meter)."""
    full = tok(cloze(f) + " " + ans + ".", return_tensors="pt").to(DEV)
    npre = tok(cloze(f), return_tensors="pt")["input_ids"].shape[1]
    logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())


def question_form(f):
    """Interrogative self-test form (surface-independent of the declarative eval cloze).
    'The user's favorite color is X.' -> 'Q: What is the user's favorite color?\\nA:'"""
    attr = f["statement"][len("The user's "): f["statement"].find(" is ")]
    return f"Q: What is the user's {attr}?\nA:"


@torch.no_grad()
def selftest_question(model, tok, facts):
    """Question-form hits for replay gating only (never scored as eval)."""
    model.eval(); hits = []
    for f in facts:
        ids = tok(question_form(f), return_tensors="pt").to(DEV)
        g = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        comp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
        hits.append(int(L.contains_match_ci(f["answer"], comp)))
    return hits


@torch.no_grad()
def recall(model, tok, facts, all_facts):
    """Per-fact recall hits (greedy cloze) + recognition margins (2-AFC logprob vs a fixed
    same-pool distractor): 'remembered exactly' vs 'seen'."""
    model.eval(); hits, margins = [], []
    for f in facts:
        ids = tok(cloze(f), return_tensors="pt").to(DEV)
        g = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        comp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]
        hits.append(int(L.contains_match_ci(f["answer"], comp)))
        others = [x["answer"] for x in all_facts if x["fid"] != f["fid"]]
        dis = random.Random(31 * f["fid"] + args.seed).choice(others)
        margins.append(round(answer_lp(model, tok, f, f["answer"]) - answer_lp(model, tok, f, dis), 2))
    return hits, margins


def layer_range(n_layers):
    thirds = {"early": (0, n_layers // 3), "mid": (n_layers // 3, 2 * n_layers // 3),
              "late": (2 * n_layers // 3, n_layers)}
    if args.layers in thirds:
        a, b = thirds[args.layers]; return list(range(a, b))
    a, b = args.layers.split("-"); return list(range(int(a), int(b) + 1))


def cfg(n_layers=None):
    tm, kw = "all-linear", {}
    if args.layers != "all":  # peft forbids layers_to_transform with str target_modules -> explicit list
        tm = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        kw = {"layers_to_transform": layer_range(n_layers), "layers_pattern": "layers"}
    return LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules=tm,
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", **kw)


def main():
    global LP, HIDDEN
    L.check_env(); torch.manual_seed(args.seed)
    facts = (make_counterfact(args.n_stream, args.seed) if args.facts == "counterfact"
             else make_facts(args.n_stream, args.seed))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                attn_implementation="eager").to(DEV)
    model = get_peft_model(base, cfg(base.config.num_hidden_layers), adapter_name="mem")
    LP = [p for _, p in model.named_parameters() if p.requires_grad]
    lora_layers = sorted({int(n.split("layers.")[1].split(".")[0])
                          for n, _ in model.named_parameters() if "lora" in n and "layers." in n})
    print(f"[accum] layers={args.layers} lora on {len(lora_layers)} layers {lora_layers[0]}..{lora_layers[-1]} "
          f"of {base.config.num_hidden_layers} | trainable={sum(p.numel() for p in LP)/1e6:.1f}M")
    opt = torch.optim.AdamW(LP, lr=args.lr)
    if args.mech == "bf":
        HIDDEN = [[p.detach().clone() for _ in range(N - 1)] for p in LP]
    use_ewc = args.mech in ("ewc", "ewcreplay")
    use_replay = args.mech in ("replay", "ewcreplay")
    fisher = [torch.zeros_like(p) for p in LP] if use_ewc else None
    theta_star = [p.detach().clone() for p in LP] if use_ewc else None
    gs_items = None
    if args.firewall_n > 0:
        gs_items = L.load_gsm8k_subset(os.path.join(os.path.dirname(__file__),
                                       "gsm8k_pilot_ids.json"))[: args.firewall_n]
        with model.disable_adapter():
            gsm_base = L.eval_gsm8k(model, tok, gs_items, device=DEV)
        print(f"[accum] firewall base GSM8K = {gsm_base:.2f}")

    print(f"[accum] mech={args.mech} model={args.model} r{args.rank} ws={args.ws} stream={args.n_stream} "
          f"seed={args.seed}" + (f" N={N}" if args.mech == "bf" else "")
          + (f" lam={args.ewc_lambda:g}" if args.mech == "ewc" else "")
          + (f" frac={args.local_frac}" if args.mech == "local" else "")
          + (f" M={args.replay_m}" if args.mech == "replay" else ""))

    buffer, curve, last_status = [], [], {}
    for k, f in enumerate(facts):
        mask = make_mask(args.seed * 100000 + k) if args.mech == "local" else None
        cur_ewc = None
        if use_ewc and k > 0:
            tot = sum(fi.sum() for fi in fisher); cnt = sum(fi.numel() for fi in fisher)
            mean = (tot / cnt).clamp_min(1e-12)
            cur_ewc = ([fi / mean for fi in fisher], theta_star, args.ewc_lambda)
        for _ in range(args.ws):
            grad_step(opt, model, tok, f, ewc=cur_ewc, mask=mask)
        if args.mech == "bf":
            cascade_step()                                    # consolidate AFTER acquiring (once/fact), not mid-write
        if use_ewc:
            fk = fisher_of(model, tok, f)
            for i in range(len(LP)):
                fisher[i] = fisher[i] + fk[i]
            theta_star = [p.detach().clone() for p in LP]
        if use_replay and buffer:
            rng = random.Random(args.seed + k)
            if args.replay_policy == "miss":
                misses = [b for b in buffer if last_status.get(b["fid"], 1) == 0]
                rest = [b for b in buffer if last_status.get(b["fid"], 1) == 1]
                picks = (rng.sample(misses, args.replay_m) if len(misses) >= args.replay_m
                         else misses + rng.sample(rest, min(args.replay_m - len(misses), len(rest))))
            else:
                picks = rng.sample(buffer, min(args.replay_m, len(buffer)))
            for rf in picks:
                grad_step(opt, model, tok, rf)
        buffer.append(f)
        if (k + 1) % args.probe_every == 0 or k == len(facts) - 1:
            hits, margins = recall(model, tok, facts[: k + 1], facts)
            pt = {"k": k + 1, "hits": hits, "margins": margins}
            if args.selftest_form == "question":
                st = selftest_question(model, tok, facts[: k + 1])
                for i, h in enumerate(st):
                    last_status[i] = h
                pt["selftest_hits"] = st
            else:
                for i, h in enumerate(hits):
                    last_status[i] = h
            cr = sum(hits) / len(hits)
            nrec = sum(1 for m in margins if m > 0)
            pt.update({"cumrecall": round(cr, 3), "n_recalled": round(cr * (k + 1), 1),
                       "n_recognized": nrec})
            curve.append(pt)
            if k == len(facts) - 1 and args.facts == "counterfact":
                model.eval(); ph = []
                with torch.no_grad():
                    for f in facts:
                        if not f.get("para"):
                            ph.append(None); continue
                        ids = tok(f["para"], return_tensors="pt").to(DEV)
                        g = model.generate(**ids, max_new_tokens=12, do_sample=False,
                                           pad_token_id=tok.pad_token_id)
                        comp = tok.decode(g[0][ids["input_ids"].shape[1]:],
                                          skip_special_tokens=True).split("\n")[0]
                        ph.append(int(L.contains_match_ci(f["answer"], comp)))
                curve[-1]["para_hits"] = ph
                print(f"  [{args.mech}] paraphrase recall: "
                      f"{sum(h for h in ph if h)}/{sum(1 for h in ph if h is not None)}")
            print(f"  [{args.mech}] after {k+1:2d} facts: cumrecall(1..{k+1})={cr:.3f} "
                  f"(~{cr*(k+1):.0f} of {k+1} held, {nrec} recognized)")

    final = curve[-1]
    print(f"[accum {args.mech}] FINAL: {final['n_recalled']:.0f}/{args.n_stream} facts recallable, "
          f"{final['n_recognized']}/{args.n_stream} recognized "
          f"(cumrecall={final['cumrecall']:.3f})  curve={[c['n_recalled'] for c in curve]}")
    fw = {}
    if args.firewall_n > 0:
        with model.disable_adapter():
            fw = {"gsm8k_base": gsm_base, "gsm8k_off": L.eval_gsm8k(model, tok, gs_items, device=DEV),
                  "n": len(gs_items)}
        print(f"[accum {args.mech}] FIREWALL base={fw['gsm8k_base']:.2f} -> off={fw['gsm8k_off']:.2f} "
              f"(damage {fw['gsm8k_base']-fw['gsm8k_off']:+.2f})")
    out = {"mech": args.mech, "model": args.model, "n_stream": args.n_stream, "ws": args.ws, "rank": args.rank,
           "levels": N, "dt": args.dt, "ewc_lambda": args.ewc_lambda, "local_frac": args.local_frac,
           "replay_m": args.replay_m, "probe_every": args.probe_every, "seed": args.seed,
           "layers": args.layers, "lora_layers": lora_layers, "replay_policy": args.replay_policy,
           "selftest_form": args.selftest_form, "facts_src": args.facts,
           "final_n_recognized": final["n_recognized"],
           "curve": curve, "firewall": fw,
           "final_cumrecall": final["cumrecall"], "final_n_recalled": final["n_recalled"]}
    os.makedirs(args.out, exist_ok=True)
    tag = ""
    if args.mech == "bf":
        tag = f"_N{N}_dt{args.dt:g}"
    if args.mech in ("ewc", "ewcreplay"):
        tag += f"_l{args.ewc_lambda:g}"
    if args.layers != "all":
        tag += f"_L{args.layers}"
    if args.replay_policy != "uniform":
        tag += f"_P{args.replay_policy}"
    if args.selftest_form != "same":
        tag += f"_ST{args.selftest_form}"
    if args.model != "Qwen/Qwen3.5-2B":
        tag += "_M" + args.model.split("/")[-1].replace("-", "").replace(".", "")[:14]
    if args.facts != "synthetic":
        tag += f"_F{args.facts}"
    fn = f"accum_{args.mech}{tag}_n{args.n_stream}_pe{args.probe_every}_s{args.seed}.json"
    json.dump(out, open(os.path.join(args.out, fn), "w"), indent=2)
    print(f"[accum] saved {fn}")
    if args.save_adapter:
        ad_dir = os.path.join(args.out, "adapters")
        os.makedirs(ad_dir, exist_ok=True)
        sd = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k.lower()}
        ap_fn = fn.replace(".json", ".pt")
        torch.save(sd, os.path.join(ad_dir, ap_fn))
        print(f"[accum] saved adapter {ap_fn} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
