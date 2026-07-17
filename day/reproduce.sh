#!/bin/bash
# Full run matrix for the paper (~94 runs, ~5-8 min each on one 24GB GPU).
# AMD ROCm note: export HSA_OVERRIDE_GFX_VERSION=11.0.0 on RDNA3.
set -e
DEV=${1:-cuda:0}
run() { echo "=== $* $(date +%H:%M:%S) ==="; python accum.py --n-stream 48 --probe-every 2 --dev $DEV "$@"; }
for s in 1234 2025 777; do
  # Table 1: mechanism ladder + policies (recognition margins logged in every run)
  run --mech naked --seed $s
  run --mech ewc --seed $s
  run --mech replay --seed $s
  run --mech replay --replay-policy miss --seed $s
  run --mech ewcreplay --firewall-n 10 --seed $s
  run --mech ewcreplay --replay-policy miss --firewall-n 10 --seed $s
  run --mech bf --seed $s
  # Table 2: depth bands (matched 8-layer capacity)
  for band in early mid late; do
    run --mech naked --layers $band --seed $s
    run --mech ewc --layers $band --seed $s
    run --mech replay --layers $band --seed $s
  done
  run --mech ewcreplay --layers late --firewall-n 10 --seed $s
done
# n=5 compositions (extra seeds)
for s in 7 42; do
  run --mech ewcreplay --firewall-n 10 --seed $s
  run --mech ewcreplay --replay-policy miss --firewall-n 10 --seed $s
done
# self-test cadence sweep (probe-every doubles as self-test cadence for miss policy)
for pe in 6 12; do
  for s in 1234 2025 777; do
    python accum.py --n-stream 48 --probe-every $pe --dev $DEV --mech ewcreplay --replay-policy miss --firewall-n 10 --seed $s
  done
done
# substrate transfer (hyperparameters unchanged)
M=HuggingFaceTB/SmolLM2-1.7B-Instruct
for s in 1234 2025 777; do
  run --model $M --mech naked --seed $s
  run --model $M --mech ewc --seed $s
  run --model $M --mech replay --seed $s
  run --model $M --mech replay --replay-policy miss --seed $s
  run --model $M --mech ewcreplay --firewall-n 10 --seed $s
  run --model $M --mech ewcreplay --replay-policy miss --firewall-n 10 --seed $s
done
# real-fact stream (CounterFact) + paraphrase probe
for s in 1234 2025 777; do
  run --facts counterfact --mech naked --seed $s
  run --facts counterfact --mech ewc --seed $s
  run --facts counterfact --mech replay --seed $s
  run --facts counterfact --mech ewcreplay --firewall-n 10 --seed $s
  run --facts counterfact --mech ewcreplay --replay-policy miss --firewall-n 10 --seed $s
done
python analyze.py
