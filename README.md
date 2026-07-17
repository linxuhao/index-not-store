# An Index, Not a Store

Code and raw results for:

> **An Index, Not a Store: What Online In-Weight Memory Keeps, Loses, and Is For**
> (preprint, DOI to be added)

When a frozen language model writes facts into a LoRA adapter online — one per turn, single pass,
day after day — the adapter becomes a **recognition index, not a fact store**, and the system
design follows from that identity. The argument runs in four acts, each backed by the raw per-fact
timelines in this repository:

1. **Act 1 — the store fails.** Accumulation is re-instatement, not persistence: median
   time-to-first-miss is 3–8 writes under every mechanism we test while end-of-stream retention
   varies by 12×; naked sequential SGD sustains only a ~1.5-batch FIFO; the same adapters that
   recall ~90% in the write form answer only 12–31% of paraphrases; and a per-fact savings probe
   finds **no** accelerated re-acquisition of forgotten facts. (`day/`, `act4/recovery.py`)
2. **Act 2 — recognition survives, cross-mechanism.** Behind the failing recall readout, protected
   mechanisms leave most facts 2-AFC-discriminable: EWC on the second substrate recalls *nothing*
   while recognizing well above chance, and a GRACE-style editor at matched budget recalls 0–1/48
   while recognizing 45–47/48. Recognition is the last readout to die. (`day/`, `act4/grace.py`)
3. **Act 3 — truth must live in the log.** At identical nightly budget, error-gated consolidation
   from a ground-truth log retains **≈2.1–2.2×** a recency baseline and rescues exactly the
   recognized-but-not-recalled class (next-day survival 0% → 46%); nights without a log do not
   merely fail, they *poison* (pure self-recitation 0/144; a hybrid lands *below* doing nothing,
   4.3 vs 25.0); over 30 days no knee is found within range; a capability firewall plus a 100-item
   GSM8K probe scope what serving costs. (`loop/`, `act4/fw100.py`)
4. **Act 4 — the surviving recognition is the key to using the log.** A candidate-free
   adapter-vs-base log-probability signal on *presented* content separates written from
   never-written facts at **AUC 0.913–0.932** in a question form whose recall is 1–6/48 (cloze
   **0.977–0.999**); the base-only signal (S0) sits at chance (**0.489–0.541**) by construction and
   by measurement; the recognition meter's zero-point is at chance on the untrained base (2-AFC
   20–25/48 Qwen, 18–27/48 SmolLM2); and a minimal gate demo with predictions specified before
   execution both works (an answer-independent **slot-KL** gate cuts fabrication from ≈100% to
   **10–33%** at a frozen threshold) and fails informatively (candidate-scored gates collapse to
   self-argmax bias). The readout gap — recognized when shown at AUC 0.93–0.99, hard to read out
   unprompted (Youden **J ≤ 0.27**) — is the paper's closing open problem. (`act4/`)

The design that survives all four acts: generation upstream, truth in the log, and the adapter
serving as a familiarity organ over external content.

## Repository layout

```
day/                  # Act 1-2 day scale (48-fact single-pass streams)
  accum.py            #   the streaming instrument (all mechanisms; --selftest-form, --save-adapter)
  lib.py              #   shared machinery (probes, GSM8K firewall eval, templates)
  analyze.py          #   re-derives the day-scale tables from day/results/ (pinned definitions)
  reproduce.sh        #   full day-scale run matrix (~94 runs x ~5-8 min on one 24GB GPU)
  gsm8k_pilot_ids.json#   frozen 10-item GSM8K firewall subset
  results/            #   raw per-fact timelines for every day-scale run (+ PROVENANCE.md)
loop/                 # Act 3 day/night loop (up to 30 days / 720 facts)
  loop.py             #   the end-to-end loop driver (all night arms + dual-state probes)
  lib.py analyze.py make_figure.py reproduce.sh gsm8k_pilot_ids.json
  results/            #   raw per-night, per-fact timelines (incl. D30 and _oncap re-runs)
act4/                 # Act 2 baseline + Act 4 probes (inference-only over saved day-scale adapters)
  grace.py            #   GRACE-style key-value editing baseline (same stream/probes as accum.py)
  familiarity.py      #   candidate-free dlp familiarity probe (written vs never-written, AUC)
  gate_demo.py        #   minimal familiarity gate (cand_dlp / slot_kl signals, frozen threshold)
  recog_zero.py       #   recognition-meter zero-point control on the untrained base
  fw100.py            #   100-item GSM8K capability probe (freezes gsm8k_fw100_ids.json at repo root)
  recovery.py         #   per-fact savings (re-acquisition) probe over saved adapters
  accum.py lib.py     #   local copies of day/accum.py + day/lib.py (see note below)
  gsm8k_pilot_ids.json
  results/            #   grace timelines, familiarity/ (fam_*, gate_*), recovery_rerun/
LICENSE CITATION.cff .zenodo.json requirements.txt
```

Note: `act4/accum.py` and `act4/lib.py` are verbatim copies of `day/accum.py` and `day/lib.py` —
the Act-4 probes import `make_facts`/`cloze` from `accum.py` (with a `sys.argv` shim, since the
driver parses arguments at import) and the shared probes from `lib.py`. Copies were chosen over
`sys.path` tricks so each directory runs standalone.

## Reproduce

```bash
pip install -r requirements.txt   # torch transformers peft datasets matplotlib
```

**Day scale (Acts 1–2):**

```bash
cd day
python analyze.py        # re-derive all day-scale tables from shipped raw results (CPU-only)
bash reproduce.sh        # re-run everything (downloads Qwen3.5-2B; fp32, ~10GB VRAM)
# single arms:
python accum.py --mech ewcreplay --replay-policy miss --firewall-n 10 --seed 1234   # the winner arm
python accum.py --mech ewcreplay --replay-policy miss --selftest-form question --seed 1234  # circularity control
python accum.py --mech ewcreplay --replay-policy miss --save-adapter --seed 1234   # archive adapter for act4/
```

**The loop (Act 3):**

```bash
cd loop
python analyze.py        # re-derive every loop table (CPU-only)
python make_figure.py    # regenerate the horizon figure
bash reproduce.sh        # full run matrix (D6 ~1h, D12 ~3h, D30 ~10h each on one 24GB GPU)
```

**Act 4 probes** (inference-only; they mount saved end-of-stream adapters produced by
`day/accum.py --save-adapter` — adapter `.pt` files are not shipped, only the resulting JSONs):

```bash
cd act4
python grace.py --steps 8 --lr 1 --seed 1234                       # matched-budget editing baseline
python grace.py --steps 100 --lr 1 --facts counterfact --seed 1234 # upper-bound + CounterFact arm
python familiarity.py --adapter <adapter.pt> --model Qwen/Qwen3.5-2B --seed 1234
python gate_demo.py --adapter <adapter.pt> --seed 1234 --signal slot_kl --probe-form cloze  # calibrate
python gate_demo.py --adapter <adapter.pt> --seed 2025 --signal slot_kl --probe-form cloze --threshold 11.188  # frozen
python recog_zero.py --model Qwen/Qwen3.5-2B --seeds 1234 2025 777
python fw100.py --model Qwen/Qwen3.5-2B                            # 100-item capability reference
python recovery.py --adapter <adapter.pt> --seed 1234 --steps 1 2  # savings probe
```

### `grace.py` flags

| flag | values (default first) | meaning |
|---|---|---|
| `--model` | Qwen/Qwen3.5-2B · any HF causal LM | substrate (non-default encoded in the filename) |
| `--steps` | 8 | value-optimization steps per fact (8 = budget matched to the stream's ws=8) |
| `--lr` | 0.5 | Adam lr for the direct value-vector optimization |
| `--eps` | auto | deferral radius; default = half the median clean pairwise key distance |
| `--layer-frac` | 0.75 | depth of the wrapped Linear (late third, per the expression gradient) |
| `--facts` | synthetic · counterfact | fact source (as in `accum.py`) |
| `--probe-every` | 6 | probe cadence in writes |
| `--firewall-n` | 0 | >0: GSM8K items for the editor-off firewall check |
| `--n-stream --seed --dev --out` | 48 · 1234 · cuda:0 · ./results | stream / run housekeeping |

### `familiarity.py` flags

| flag | values (default first) | meaning |
|---|---|---|
| `--adapter` | (required) | saved end-of-stream LoRA state (`.pt`) to mount |
| `--model` | HuggingFaceTB/SmolLM2-1.7B-Instruct · any HF causal LM | substrate the adapter was trained on |
| `--rank` | 64 | LoRA rank of the saved adapter |
| `--seed` | (required) | stream seed of the adapter (regenerates the 48 written facts) |
| `--novel-seed-offset` | 50000 | disjoint seed offset for the never-written control facts |
| `--n --dev --out` | 48 · cuda:0 · ./results/familiarity | housekeeping |

Signals logged per form (cloze + question): `S0_base_lp` (base-only chance control), `S1_raw_answer`
(candidate-free dlp — the headline), `S2_contrastive`, `S3_prompt_attr`. Files written by the
earlier probe version lack the S0/contrastive rows (see Provenance).

### `gate_demo.py` flags

| flag | values (default first) | meaning |
|---|---|---|
| `--adapter` / `--seed` | (required) | saved adapter + its stream seed |
| `--model` | Qwen/Qwen3.5-2B | substrate |
| `--signal` | cand_dlp · slot_kl | gate signal: candidate-scored dlp, or answer-independent KL(adapter‖base) at the answer slot |
| `--probe-form` | question · cloze | form the gate queries in |
| `--threshold` | None | frozen threshold to replay; None = calibrate here (max Youden J) and print it |
| `--contrast` | off | contrastive candidate scoring (subtract reference-pseudoword dlp) |
| `--rank --n --novel-seed-offset --dev --out` | 64 · 48 · 50000 · cuda:0 · ./results/familiarity | housekeeping |

### `recog_zero.py` flags

| flag | values (default first) | meaning |
|---|---|---|
| `--model` | Qwen/Qwen3.5-2B | base model (no adapter — that is the point) |
| `--seeds` | 1234 2025 777 | stream seeds whose exact 2-AFC draws are replayed |
| `--n --dev` | 48 · cuda:0 | housekeeping |

### `fw100.py` flags

| flag | values (default first) | meaning |
|---|---|---|
| `--model` | Qwen/Qwen3.5-2B | substrate for the 100-item GSM8K reference |
| `--cores` | (none) | core `.pt` checkpoints to mount (rank 32) for ON-state probes |
| `--n --seed --dev --out` | 100 · 1234 · cuda:0 · ./results | housekeeping |

On first run it freezes the 100-item subset to `gsm8k_fw100_ids.json` at the repo root (seeded
sample of the openai/gsm8k test split, excluding the 10 pilot indices).

### `recovery.py` flags

| flag | values (default first) | meaning |
|---|---|---|
| `--adapter` / `--seed` | (required) | saved adapter + its stream seed |
| `--model` | HuggingFaceTB/SmolLM2-1.7B-Instruct | substrate |
| `--steps` | 1 2 | relearning step counts k (per-fact, isolated, matched to stream writes) |
| `--lr` | 3e-5 | relearning lr (same as stream writes; no EWC penalty) |
| `--rank --n --novel-seed-offset --dev --out` | 64 · 48 · 50000 · cuda:0 · ./results/recovery_rerun | housekeeping |

**Hardware note**: the code is standard PyTorch + transformers + peft — it runs on NVIDIA/CUDA
as-is (`--dev cuda:0` is the same device string under ROCm, which is simply what *our* GPUs were).
The ROCm remarks in this repo describe our measurement hardware, not a requirement; expect exact
numbers to differ on any hardware (they differ across our own seeds too).

## Provenance

1. **`day/results/`** (also in `day/results/PROVENANCE.md`): all files whose names contain a
   `_peN_` tag are the canonical runs used in the paper; `analyze.py` reads only these. Files
   **without** a `_peN_` tag (20 files) predate the recognition probe and the probe-cadence flag:
   they carry `recog: null` and an implicit probe cadence, and were superseded on 2026-07-04 when
   the recognition meter landed. They are kept for provenance (per our protocol, superseded runs
   are archived, not deleted) and are not cited by the paper or the analysis script.
   `pre_recog_2026-07-04/` holds a second provenance layer: the original `_pe2_` baseline runs
   backed up immediately before the recognition-instrumented re-runs replaced them at the same
   filenames (also uncited; the shipped top-level `_pe2_` files are the canonical ones).
2. **`loop/results/`**: `loop_misslog_D6_f24_s{1234,2025,777}.json` are the canonical error-gated
   D6 runs (finals **52/60/54**, git-preserved originals). `loop_misslog_oncap_D6_f24_s{1234,2025}.json`
   (finals **48/58**) are an independent re-run from a filename-collision incident — the `--oncap`
   re-runs briefly overwrote the originals before the originals were restored from git — retained
   as an unplanned replicate. The paper reports both (the error-gated vs recency ratio is
   ≈2.1–2.2× under either set).
3. **`act4/results/familiarity/`**: the `fam_*_s1234.json` files were written by probe v2 and lack
   the S0 base-only and contrastive signal rows; the `s2025`/`s777` files are probe v3 (with S0).
   The signals shared by both versions (S1 raw-answer dlp, S3 prompt-attr) are computed
   identically. Gate files record whether their threshold was calibrated in-run
   (`calibrated_here`) or replayed frozen from the s1234 calibration run.

## Citation

```bibtex
@article{lin2026index,
  title={An Index, Not a Store: What Online In-Weight Memory Keeps, Loses, and Is For},
  author={Lin, Xuhao},
  year={2026},
  note={Preprint}
}
```

Companion released artifacts: day-scale toolkit (software doi:10.5281/zenodo.21199026, preprint
doi:10.5281/zenodo.21232648) and loop toolkit (software doi:10.5281/zenodo.21309811, preprint
doi:10.5281/zenodo.21310038).

MIT license (see LICENSE).
