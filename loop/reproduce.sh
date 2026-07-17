#!/bin/bash
# Full run matrix (~40 runs; D6 ~1h, D12 ~3h, D30 ~10h each on one 24GB GPU).
# AMD ROCm note: export HSA_OVERRIDE_GFX_VERSION=11.0.0 on RDNA3.
set -e
DEV=${1:-cuda:0}
run() { echo "=== $* $(date +%H:%M:%S) ==="; python loop.py --fpd 24 --firewall-n 10 --dev $DEV "$@"; }
for s in 1234 2025 777; do
  for night in full markov misslog recite hybrid; do run --night $night --days 6 --seed $s; done
  for night in full markov misslog; do run --night $night --days 12 --seed $s; done
done
run --night misslog --days 30 --probe-stride 3 --seed 1234
run --night misslog --days 30 --probe-stride 3 --seed 2025
run --night full    --days 30 --probe-stride 3 --seed 1234
run --night markov  --days 30 --probe-stride 3 --seed 1234
# substrate transfer: all five night arms on SmolLM2 (hyperparameters unchanged)
M=HuggingFaceTB/SmolLM2-1.7B-Instruct
for s in 1234 2025; do
  for night in full markov misslog recite hybrid; do
    run --model $M --night $night --days 6 --seed $s
  done
done
# dual-state capability probes (separate filenames via _oncap)
run --night misslog --days 6 --oncap 10 --save-core --seed 1234
run --night misslog --days 6 --oncap 10 --save-core --seed 2025
python analyze.py
