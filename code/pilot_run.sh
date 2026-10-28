#!/usr/bin/env bash
# Phase-1 pilot (plan section 9, P1): every arm on 6 scenarios x 20 replicates, then the GOF layer.
# Smoke seed stream (777), so no pilot replicate is ever part of the main run.
# Every step resumes: rerunning this script only does what is missing.
#
#   bash code/pilot_run.sh            (from rebuild/, or any folder)
#
# Steps
#   1  classical arms (CPU, project venv)
#   2  foundation models, solo on the GPU: tabpfn, tabicl, mitra
#   3  side arms tabpfn_legacy and tabpfn_t1 on the four cheap scenarios
#   4  two GPU processes at once (throughput and cross-process reproducibility)
#   5  GOF layer on the pilot files (background) while the GPU runs the timing grid
#   6  timing grid: TFM fit times in the n x event-rate cells the pilot does not cover
#   7  scoring (oracle and empirical estimands)

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"

PYCPU="$REB/../.venv/Scripts/python.exe"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
LOG="$REB/logs"
SC="S1_pi03_n200,S2_pi10_n500,S3_pi30_n1000,S4_pi03_n3000,T2_pi10_n3000,T3_pi10_n500"
SC_SIDE="S1_pi03_n200,S2_pi10_n500,S3_pi30_n1000,T3_pi10_n500"
PRED="results/pilot/preds"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(stamp)] $*" | tee -a "$LOG/pilot_steps.log"; }

# GPU utilisation and CPU load every 15 s, for the whole pilot
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 15 >> "$LOG/pilot_gpu_util.csv" 2>/dev/null &
NVPID=$!
typeperf "\Processor(_Total)\% Processor Time" -si 15 >> "$LOG/pilot_cpu_load.csv" 2>/dev/null &
TPPID=$!
trap 'kill $NVPID $TPPID 2>/dev/null' EXIT

say "step 1: classical arms"
"$PYCPU" code/run.py --smoke --out "$PRED" --scenarios "$SC" --reps 1-20 --models classical \
    --n-workers 8 --threads 1 >> "$LOG/pilot_step1_classical.log" 2>&1
say "step 1 exit $?"

say "step 2: tabpfn, tabicl, mitra on the GPU (solo)"
"$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$SC" --reps 1-20 --models tabpfn,tabicl,mitra \
    --device cuda --threads 6 >> "$LOG/pilot_step2_gpu.log" 2>&1
say "step 2 exit $?"

say "step 3: tabpfn_legacy and tabpfn_t1 on $SC_SIDE"
"$PYGPU" code/run.py --smoke --out "$PRED" --scenarios "$SC_SIDE" --reps 1-20 \
    --models tabpfn_legacy,tabpfn_t1 --device cuda --threads 6 >> "$LOG/pilot_step3_side.log" 2>&1
say "step 3 exit $?"

say "step 4: two GPU processes at once"
CPRED="results/pilot_concurrency/preds"
"$PYGPU" code/run.py --smoke --out "$CPRED" --scenarios S1_pi03_n200,S2_pi10_n500 --reps 1-4 \
    --models tabpfn,tabicl,mitra --device cuda --threads 6 >> "$LOG/pilot_step4_procA.log" 2>&1 &
PA=$!
"$PYGPU" code/run.py --smoke --out "$CPRED" --scenarios S1_pi03_n200,S2_pi10_n500 --reps 5-8 \
    --models tabpfn,tabicl,mitra --device cuda --threads 6 >> "$LOG/pilot_step4_procB.log" 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
say "step 4 exit $EA $EB"

say "step 5: GOF layer (background, 12 workers)"
Rscript R/gof_layer.R run --preds "$PRED" --out results/pilot/gof --workers 12 \
    >> "$LOG/pilot_step5_gof.log" 2>&1 &
GOFPID=$!

say "step 6: timing grid (GPU, under the GOF load)"
"$PYGPU" code/run.py --smoke --out results/pilot_timing/preds \
    --scenarios S1_pi10_n200,S1_pi03_n500,S1_pi03_n1000,S1_pi10_n1000,S1_pi10_n3000,S1_pi30_n3000,S2_pi10_n500,S4_pi03_n3000 \
    --reps 1-3 --models tabpfn,tabicl,mitra --device cuda --threads 6 >> "$LOG/pilot_step6_timing.log" 2>&1
say "step 6 exit $?"

wait $GOFPID
say "step 5 exit $?"

say "step 7: scoring"
"$PYCPU" code/score.py --preds "$PRED" --scenarios "$SC" --n-workers 6 --force >> "$LOG/pilot_step7_score.log" 2>&1
say "step 7 exit $?"
Rscript R/gof_layer.R check --preds "$PRED" --out results/pilot/gof >> "$LOG/pilot_step7_score.log" 2>&1
say "pilot finished"
