#!/usr/bin/env bash
# Pre-freeze smoke (2026-10-07): every arm, 2 scenarios x 2 replicates on the smoke seed stream (777),
# laid out exactly as the main run lays out its files (RUN_MAIN.md), then scoring and the GOF layer.
#   S1_pi03_n200   rarest and smallest cell: degenerate fits, the LightGBM Platt rule, ridge path
#   T2_pi10_n500   p = 30: DeepGOF-1 axes reading; TabPFN-3.5 on a CPU worker (tag tabpfn_cpu)
#
#   bash code/smoke_freeze.sh        (from rebuild/; resumable)

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
export TFMCAL_TABPFN_VERSION=V3_5
export TFMCAL_TABPFN_SHA256=ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3

PYCPU="$REB/../.venv/Scripts/python.exe"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
OUT="results/smoke_freeze"
PRED="$OUT/preds"
LOG="$REB/logs/smoke_freeze.log"
SC="S1_pi03_n200,T2_pi10_n500"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "1 classical arms (project venv)"
"$PYCPU" code/run.py --smoke --out "$PRED" --scenarios "$SC" --reps 1-2 --models classical \
    --n-workers 4 --threads 1 >> "$LOG" 2>&1
say "1 exit $?"

say "2 GPU: TabICL, Mitra, legacy V2, both temperature arms; TabPFN-3.5 on S1"
"$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$SC" --reps 1-2 \
    --models tabicl,mitra,tabpfn_legacy,tabpfn_t1,tabicl_t1 --device cuda --threads 6 >> "$LOG" 2>&1
say "2a exit $?"
"$PYGPU" code/run.py --smoke --out "$PRED" --scenarios S1_pi03_n200 --reps 1-2 \
    --models tabpfn --device cuda --threads 6 >> "$LOG" 2>&1
say "2b exit $?"

say "3 CPU worker: TabPFN-3.5 on T2 (tag tabpfn_cpu)"
"$PYGPU" code/run.py --smoke --out "$PRED" --scenarios T2_pi10_n500 --reps 1-2 \
    --models tabpfn --device cpu --threads 6 --tag tabpfn_cpu >> "$LOG" 2>&1
say "3 exit $?"

say "4 scoring"
"$PYCPU" code/score.py --preds "$PRED" --scenarios "$SC" --force >> "$LOG" 2>&1
say "4 exit $?"

say "5 GOF layer: oracle only first (the reps 201-500 mode), then every arm (resume)"
Rscript R/gof_layer.R run --preds "$PRED" --out "$OUT/gof" --workers 6 --B 99 --arms oracle >> "$LOG" 2>&1
say "5a exit $?"
Rscript R/gof_layer.R run --preds "$PRED" --out "$OUT/gof" --workers 6 --B 99 >> "$LOG" 2>&1
say "5b exit $?"
Rscript R/gof_layer.R check --preds "$PRED" --out "$OUT/gof" >> "$LOG" 2>&1
say "5c exit $?"
