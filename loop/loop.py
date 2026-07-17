"""loop.py -- the end-to-end day/night in-weight memory loop.

DAY  : stream fpd facts into a fresh r64 day-core with the paper-1 winner
       (8 value-masked write steps + summed-Fisher EWC + 4 miss-gated replay steps,
       self-test every 2 facts over today's cohort, core+day active at probe time).
NIGHT: consolidate into the persistent r32 CORE, then reset the day-core. Arms:
  full    : train core on ALL past facts' true statements (ceiling; budget grows with days)
  markov  : train core on TODAY's true statements only (fixed budget — the scalable default)
  misslog : error-gated consolidation — probe CORE-ONLY recall over all past facts, train
            core from the log ONLY on failures, same gradient budget as markov
            (the extra core-only probe pass is the mechanism's self-test cost)
  recite  : self-distillation — core-only cloze-generate each past fact, train on the
            reconstructions (errors included), same gradient budget as markov
Probes after night (core only, day reset): per-fact recall + 2-AFC recognition margins over
ALL facts so far -> retention-by-age curves. Firewall (adapters-off GSM8K) at start+end.

Pre-registered predictions (2026-07-06): (a) markov shows old-day decay; misslog flattens the
age curve at matched budget; (b) recite's loss concentrates exactly on the recognition surplus
(facts recognized but not recalled by the core cannot survive recitation).

Run:  python loop.py --night misslog --days 6 --firewall-n 10 --seed 1234
"""
import argparse, json, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--night", choices=["full", "markov", "misslog", "recite", "hybrid"], default="markov")
ap.add_argument("--days", type=int, default=6)
ap.add_argument("--fpd", type=int, default=24)             # facts per day
ap.add_argument("--core-rank", type=int, default=32)
ap.add_argument("--day-rank", type=int, default=64)
ap.add_argument("--ws", type=int, default=8)               # write steps per fact (day)
ap.add_argument("--replay-m", type=int, default=4)         # miss-gated replay steps per fact (day)
ap.add_argument("--ewc-lambda", type=float, default=300.0)
ap.add_argument("--core-epochs", type=int, default=3)      # night budget = fpd * core_epochs steps
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--core-lr", type=float, default=3e-5)
ap.add_argument("--selftest-every", type=int, default=2)
ap.add_argument("--probe-stride", type=int, default=1)  # full-history post-night probe every k days (mechanism self-tests unaffected)
ap.add_argument("--firewall-n", type=int, default=10)
ap.add_argument("--oncap", type=int, default=0)   # >0: ON-state capability probes (GSM8K items) with core+day at pre-night and core-only at post-night
ap.add_argument("--save-core", action="store_true")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
args = ap.parse_args()
DEV = args.dev


# ---------- facts (same generator as the companion day-scale instrument) ----------
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
            v = "".join(x.capitalize() for x in r.sample(syl, 3))
            if v not in seen:
                seen.add(v); break
        facts.append({"fid": i, "statement": f"The user's {attrs[i]} is {v}.", "answer": v})
    assert len({f["answer"] for f in facts}) == n
    return facts


def cloze(f):
    return f["statement"][: f["statement"].find(f["answer"])].rstrip()


def pfx(tok, text_prefix):
    return tok(text_prefix, return_tensors="pt")["input_ids"].shape[1]


# ---------- model / adapters ----------
def cfg(r):
    return LoraConfig(r=r, lora_alpha=r * 2, target_modules="all-linear",
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")


def load():
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                attn_implementation="eager").to(DEV)
    model = get_peft_model(base, cfg(args.core_rank), adapter_name="core")
    model.add_adapter("day", cfg(args.day_rank))
    return tok, model


def params_of(model, name):
    return [p for n, p in model.named_parameters() if name in n and p.requires_grad]


def serve_adapters(model, names):
    model.eval(); model.base_model.set_adapter(names)


FLUENCY_TEXTS = [
    "The committee reviewed the proposal carefully and decided to postpone the vote until the following meeting, citing unresolved questions about the budget.",
    "Rivers that begin in the mountains tend to carry sediment downstream, gradually building fertile plains where agriculture can flourish for centuries.",
    "She packed her bag the night before the trip: a change of clothes, a paperback novel, and the camera her grandfather had given her.",
    "In distributed systems, a common source of subtle bugs is the assumption that two events observed in one order by one node were produced in that order.",
    "The recipe calls for browning the onions slowly over low heat, which draws out their sweetness and forms the base of the stew.",
]


@torch.no_grad()
def fluency_ppl(model, tok):
    """Mean per-token NLL over fixed neutral texts (cheap aphasia meter; lower = more fluent)."""
    model.eval(); tot, cnt = 0.0, 0
    for t in FLUENCY_TEXTS:
        b = tok(t, return_tensors="pt").to(DEV)
        out = model(**b, labels=b["input_ids"])
        n = b["input_ids"].shape[1] - 1
        tot += float(out.loss) * n; cnt += n
    return round(tot / cnt, 4)


# ---------- training primitives ----------
def masked_step(model, tok, opt, plist, statement, prefix, ewc=None):
    """One value-masked LM step on the active adapter (+ optional summed-Fisher EWC penalty)."""
    model.train()
    b = tok(statement, return_tensors="pt").to(DEV)
    labels = b["input_ids"].clone(); labels[:, :pfx(tok, prefix)] = -100
    out = model(**b, labels=labels); opt.zero_grad(); out.loss.backward()
    if ewc is not None and ewc["n"] > 0:
        tot = sum(fi.sum() for fi in ewc["fisher"]); cnt = sum(fi.numel() for fi in ewc["fisher"])
        mean = (tot / cnt).clamp_min(1e-12)
        for p, fi, ts in zip(plist, ewc["fisher"], ewc["theta"]):
            if p.grad is not None:
                p.grad.add_(args.ewc_lambda * (fi / mean) * (p.detach() - ts))
    torch.nn.utils.clip_grad_norm_(plist, 1.0); opt.step()
    return float(out.loss)


def day_fisher_update(model, tok, ewc, plist, f):
    model.train(); model.zero_grad()
    b = tok(f["statement"], return_tensors="pt").to(DEV)
    labels = b["input_ids"].clone(); labels[:, :pfx(tok, cloze(f))] = -100
    model(**b, labels=labels).loss.backward()
    fk = [(p.grad.detach() ** 2 if p.grad is not None else torch.zeros_like(p)) for p in plist]
    tot = sum(x.sum() for x in fk); cnt = sum(x.numel() for x in fk)
    mean = (tot / cnt).clamp_min(1e-12)
    for i in range(len(plist)):
        ewc["fisher"][i] += fk[i] / mean
    model.zero_grad()
    ewc["theta"] = [p.detach().clone() for p in plist]
    ewc["n"] += 1


# ---------- probes ----------
@torch.no_grad()
def gen_value(model, tok, f, max_new=12):
    ids = tok(cloze(f), return_tensors="pt").to(DEV)
    g = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0]


@torch.no_grad()
def answer_lp(model, tok, f, ans):
    full = tok(cloze(f) + " " + ans + ".", return_tensors="pt").to(DEV)
    npre = pfx(tok, cloze(f))
    logits = model(**full).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ids = full["input_ids"][0]
    return float(lp[torch.arange(npre - 1, len(ids) - 1), ids[npre:]].sum())


@torch.no_grad()
def probe(model, tok, facts, all_facts, margins=True):
    model.eval(); hits, margs = [], []
    for f in facts:
        hits.append(int(L.contains_match_ci(f["answer"], gen_value(model, tok, f))))
        if margins:
            others = [x["answer"] for x in all_facts if x["fid"] != f["fid"]]
            dis = random.Random(31 * f["fid"] + args.seed).choice(others)
            margs.append(round(answer_lp(model, tok, f, f["answer"]) - answer_lp(model, tok, f, dis), 2))
    return hits, margs


# ---------- main ----------
def main():
    L.check_env(); torch.manual_seed(args.seed)
    facts = make_facts(args.days * args.fpd, args.seed)
    tok, model = load()
    gs_items = None
    fw = {}
    if args.firewall_n > 0:
        gs_items = L.load_gsm8k_subset(os.path.join(os.path.dirname(__file__),
                                       "gsm8k_pilot_ids.json"))[: args.firewall_n]
        with model.disable_adapter():
            fw["gsm8k_base"] = L.eval_gsm8k(model, tok, gs_items, device=DEV)
        print(f"[loop] firewall base GSM8K = {fw['gsm8k_base']:.2f}")
    if args.oncap > 0:
        with model.disable_adapter():
            fw["ppl_base"] = fluency_ppl(model, tok)
        print(f"[loop] base fluency NLL/token = {fw['ppl_base']}")
    print(f"[loop] night={args.night} model={args.model} D={args.days}x{args.fpd} "
          f"core r{args.core_rank} day r{args.day_rank} seed={args.seed}")

    day_log = []           # ground-truth statements of ALL days (the file)
    days_out = []
    night_budget = args.fpd * args.core_epochs

    for d in range(args.days):
        today = facts[d * args.fpd:(d + 1) * args.fpd]
        # -- fresh day adapter state
        model.set_adapter("day")
        dp = params_of(model, "day")
        dopt = torch.optim.AdamW(dp, lr=args.lr)
        dewc = {"fisher": [torch.zeros_like(p) for p in dp],
                "theta": [p.detach().clone() for p in dp], "n": 0}
        last_status = {}
        # ---------------- DAY ----------------
        for k, f in enumerate(today):
            model.set_adapter("day")
            for _ in range(args.ws):
                masked_step(model, tok, dopt, dp, f["statement"], cloze(f), ewc=dewc)
            day_fisher_update(model, tok, dewc, dp, f)
            written = today[: k + 1]
            if k > 0:
                rng = random.Random(args.seed + 1000 * d + k)
                misses = [b for b in written[:-1] if last_status.get(b["fid"], 1) == 0]
                rest = [b for b in written[:-1] if last_status.get(b["fid"], 1) == 1]
                picks = (rng.sample(misses, args.replay_m) if len(misses) >= args.replay_m
                         else misses + rng.sample(rest, min(args.replay_m - len(misses), len(rest))))
                for rf in picks:
                    masked_step(model, tok, dopt, dp, rf["statement"], cloze(rf))
            if (k + 1) % args.selftest_every == 0 or k == len(today) - 1:
                serve_adapters(model, ["core", "day"])
                hits, _ = probe(model, tok, written, facts, margins=False)
                for i, h in zip([x["fid"] for x in written], hits):
                    last_status[i] = h
        day_log += [dict(f) for f in today]
        serve_adapters(model, ["core", "day"])
        pre_hits, _ = probe(model, tok, facts[: (d + 1) * args.fpd], facts, margins=False)
        oncap_day = {}
        if args.oncap > 0 and ((d + 1) % args.probe_stride == 0 or d == args.days - 1):
            oncap_day = {"gsm8k_core_day": L.eval_gsm8k(model, tok, gs_items[: args.oncap], device=DEV),
                         "ppl_core_day": fluency_ppl(model, tok)}
            print(f"  [oncap] core+day (hot): GSM8K {oncap_day['gsm8k_core_day']:.2f} | NLL {oncap_day['ppl_core_day']}")
        # ---------------- NIGHT ----------------
        model.set_adapter("core")
        cp = params_of(model, "core")
        copt = torch.optim.AdamW(cp, lr=args.core_lr)
        allpast = facts[: (d + 1) * args.fpd]
        if args.night == "full":
            pool = [(f["statement"], cloze(f)) for f in allpast]
            steps = len(pool) * args.core_epochs
        elif args.night == "markov":
            pool = [(f["statement"], cloze(f)) for f in today]
            steps = night_budget
        elif args.night == "misslog":
            serve_adapters(model, ["core"])                     # self-test: core alone
            chits, _ = probe(model, tok, allpast, facts, margins=False)
            fails = [f for f, h in zip(allpast, chits) if h == 0]
            pool = [(f["statement"], cloze(f)) for f in (fails if fails else today)]
            steps = night_budget
        elif args.night == "recite":  # self-distill core's own (lossy) readout, errors included
            serve_adapters(model, ["core"])
            pool = []
            for f in allpast:
                v = gen_value(model, tok, f, max_new=8).strip().rstrip(".") or f["answer"][:1]
                pool.append((f"{cloze(f)} {v}.", cloze(f)))
            steps = night_budget
        else:  # hybrid: today's TRUE statements (from the log) + self-distilled OLD (recite) = file-off analog
            serve_adapters(model, ["core"])
            pool = [(f["statement"], cloze(f)) for f in today]
            for f in facts[: d * args.fpd]:                     # everything before today: recite
                v = gen_value(model, tok, f, max_new=8).strip().rstrip(".") or f["answer"][:1]
                pool.append((f"{cloze(f)} {v}.", cloze(f)))
            steps = night_budget
        model.set_adapter("core")
        rng = random.Random(args.seed * 7 + d)
        for s in range(steps):
            st, pre = pool[s % len(pool)] if args.night in ("full",) else rng.choice(pool)
            masked_step(model, tok, copt, cp, st, pre)
        # reset day adapter
        model.delete_adapter("day"); model.add_adapter("day", cfg(args.day_rank))
        # ---------------- POST-NIGHT PROBE (core only) ----------------
        serve_adapters(model, ["core"])
        if (d + 1) % args.probe_stride == 0 or d == args.days - 1:
            post_hits, margins = probe(model, tok, allpast, facts, margins=True)
            by_day = [sum(post_hits[i * args.fpd:(i + 1) * args.fpd]) for i in range(d + 1)]
            entry = {"day": d, "pre_night_hits": pre_hits, "post_hits": post_hits,
                     "margins": margins, "recall_by_day": by_day,
                     "n_recognized": sum(1 for m in margins if m > 0)}
            if args.oncap > 0:
                entry["oncap"] = dict(oncap_day)
                entry["oncap"]["gsm8k_core"] = L.eval_gsm8k(model, tok, gs_items[: args.oncap], device=DEV)
                entry["oncap"]["ppl_core"] = fluency_ppl(model, tok)
                print(f"  [oncap] core-only (morning): GSM8K {entry['oncap']['gsm8k_core']:.2f} | NLL {entry['oncap']['ppl_core']}")
            days_out.append(entry)
            print(f"[loop {args.night}] day {d}: pre-night {sum(pre_hits)}/{len(pre_hits)} | "
                  f"post-night core-only {sum(post_hits)}/{len(post_hits)} by-day={by_day} "
                  f"recog={days_out[-1]['n_recognized']}/{len(margins)}")
        else:  # off-stride: cheap pulse on today's cohort only
            th, _ = probe(model, tok, today, facts, margins=False)
            days_out.append({"day": d, "pre_night_hits": pre_hits, "today_post": th})
            print(f"[loop {args.night}] day {d}: pre-night {sum(pre_hits)}/{len(pre_hits)} | "
                  f"today-cohort post-night {sum(th)}/{len(th)} (stride-skipped full probe)")

    if args.firewall_n > 0:
        with model.disable_adapter():
            fw["gsm8k_off"] = L.eval_gsm8k(model, tok, gs_items, device=DEV)
        print(f"[loop] FIREWALL base={fw['gsm8k_base']:.2f} -> off={fw['gsm8k_off']:.2f}")
    out = {"night": args.night, "model": args.model, "days": args.days, "fpd": args.fpd,
           "core_rank": args.core_rank, "day_rank": args.day_rank, "ws": args.ws,
           "replay_m": args.replay_m, "ewc_lambda": args.ewc_lambda,
           "core_epochs": args.core_epochs, "lr": args.lr, "core_lr": args.core_lr,
           "selftest_every": args.selftest_every, "seed": args.seed,
           "trajectory": days_out, "firewall": fw}
    if args.save_core:
        cdir = os.path.join(args.out, "cores"); os.makedirs(cdir, exist_ok=True)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if "core" in n},
                   os.path.join(cdir, f"core_{args.night}_D{args.days}_f{args.fpd}_s{args.seed}.pt"))
    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.model == "Qwen/Qwen3.5-2B" else \
        "_M" + args.model.split("/")[-1].replace("-", "").replace(".", "")[:14]
    if args.oncap > 0:
        tag += "_oncap"
    fn = f"loop_{args.night}{tag}_D{args.days}_f{args.fpd}_s{args.seed}.json"
    json.dump(out, open(os.path.join(args.out, fn), "w"), indent=2)
    print(f"[loop] saved {fn}")


if __name__ == "__main__":
    main()
