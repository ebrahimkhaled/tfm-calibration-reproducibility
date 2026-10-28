#!/usr/bin/env bash
# Second pre-freeze smoke (2026-10-07 b), after the package pin moved to ebrahim.gof 2.10.0@19480aa and
# the random-design tier was added. Smoke seed stream (777) only; laid out as the main run lays out its
# files (RUN_MAIN.md), then scoring and the GOF layer.
#   core:   S1_pi03_n200 and T2_pi10_n500, 2 replicates, every arm (as in smoke_freeze.sh)
#   random: R0001-R0005, 1 replicate, the tier's arms; TabPFN-3.5 on the GPU or a CPU worker by the
#           design's size, Mitra-v2 "not run" where it does not fit
#
#   bash code/smoke_freeze_b.sh [core|random|gof|all]     (from rebuild/; every step resumes)

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
export TFMCAL_TABPFN_VERSION=V3_5
export TFMCAL_TABPFN_SHA256=ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3

PYCPU="$REB/../.venv/Scripts/python.exe"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
OUT="results/smoke_freeze_b"
PRED="$OUT/preds"
LOG="$REB/logs/smoke_freeze_b.log"
SC="S1_pi03_n200,T2_pi10_n500"
RD="R0001-R0005"
WHAT="${1:-all}"
grp() { "$PYCPU" code/main_groups.py "$@"; }
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [ "$WHAT" = core ] || [ "$WHAT" = all ]; then
    say "core 1 classical arms"
    "$PYCPU" code/run.py --smoke --out "$PRED" --scenarios "$SC" --reps 1-2 --models classical \
        --n-workers 4 --threads 1 >> "$LOG" 2>&1; say "exit $?"
    say "core 2 GPU: TabICL, Mitra, legacy V2, both temperature arms; TabPFN-3.5 on S1"
    "$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$SC" --reps 1-2 \
        --models tabicl,mitra,tabpfn_legacy,tabpfn_t1,tabicl_t1 --device cuda --threads 6 >> "$LOG" 2>&1; say "exit $?"
    "$PYGPU" code/run.py --smoke --out "$PRED" --scenarios S1_pi03_n200 --reps 1-2 \
        --models tabpfn --device cuda --threads 6 >> "$LOG" 2>&1; say "exit $?"
    say "core 3 CPU worker: TabPFN-3.5 on T2 (tag tabpfn_cpu)"
    "$PYGPU" code/run.py --smoke --out "$PRED" --scenarios T2_pi10_n500 --reps 1-2 \
        --models tabpfn --device cpu --threads 6 --tag tabpfn_cpu >> "$LOG" 2>&1; say "exit $?"
fi

if [ "$WHAT" = random ] || [ "$WHAT" = all ]; then
    GPU_SET="$(grp random_tabpfn_gpu | tr ',' '\n' | grep -E '^R000[1-5]$' | paste -sd, -)"
    CPU_SET="$(grp random_tabpfn_cpu | tr ',' '\n' | grep -E '^R000[1-5]$' | paste -sd, -)"
    say "random 1 classical arms ($RD)"
    "$PYCPU" code/run.py --smoke --out "$PRED" --scenarios "$RD" --reps 1 --models classical \
        --n-workers 4 --threads 1 >> "$LOG" 2>&1; say "exit $?"
    say "random 2 GPU: TabICL and Mitra on $RD; TabPFN-3.5 on $GPU_SET"
    "$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$RD" --reps 1 --models tabicl,mitra \
        --device cuda --threads 6 >> "$LOG" 2>&1; say "exit $?"
    [ -n "$GPU_SET" ] && { "$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$GPU_SET" --reps 1 \
        --models tabpfn --device cuda --threads 6 >> "$LOG" 2>&1; say "exit $?"; }
    say "random 3 CPU worker: TabPFN-3.5 on $CPU_SET (tag tabpfn_cpu)"
    [ -n "$CPU_SET" ] && { "$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$CPU_SET" --reps 1 \
        --models tabpfn --device cpu --threads 6 --tag tabpfn_cpu >> "$LOG" 2>&1; say "exit $?"; }
fi

if [ "$WHAT" = gof ] || [ "$WHAT" = all ]; then
    say "scoring"
    "$PYCPU" code/score.py --preds "$PRED" --scenarios "$SC,$RD" --force >> "$LOG" 2>&1; say "exit $?"
    say "GOF layer: oracle only first (the reps 201-500 mode), then every arm (resume)"
    Rscript R/gof_layer.R run --preds "$PRED" --out "$OUT/gof" --workers 6 --B 99 --arms oracle >> "$LOG" 2>&1
    say "exit $?"
    Rscript R/gof_layer.R run --preds "$PRED" --out "$OUT/gof" --workers 6 --B 99 >> "$LOG" 2>&1; say "exit $?"
    Rscript R/gof_layer.R check --preds "$PRED" --out "$OUT/gof" >> "$LOG" 2>&1; say "exit $?"
fi
