#!/usr/bin/env bash
# Pilot extras, run after pilot_run.sh step 6 while the GOF layer is still busy:
#   a) timing for the two stress cells the grid missed (T2 at n = 500, T3 at n = 3000)
#   b) TabPFN speed options (pilot_speed.py)
#   c) Mitra on the CPU for T2 at n = 3000 (the cell that fails on the GPU), 100 predicted rows
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
LOG="$REB/logs"
say() { echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG/pilot_steps.log"; }

say "extra a: timing T2_pi10_n500, T3_pi10_n3000"
"$PYGPU" code/run.py --smoke --out results/pilot_timing/preds --scenarios T2_pi10_n500,T3_pi10_n3000 \
    --reps 1-3 --models tabpfn,tabicl,mitra --device cuda --threads 6 >> "$LOG/pilot_extra_timing.log" 2>&1
say "extra a exit $?"

say "extra b: TabPFN speed options"
"$PYGPU" code/pilot_speed.py >> "$LOG/pilot_extra_speed.log" 2>&1
say "extra b exit $?"

say "extra c: Mitra on CPU, T2 n = 3000"
"$PYGPU" code/pilot_mitra_cpu.py >> "$LOG/pilot_extra_mitra_cpu.log" 2>&1
say "extra c exit $?"
