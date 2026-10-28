#!/usr/bin/env bash
# Second phase of the real-data run (2026-10-24): starts when the GPU plasmode pass is done
# (logs/realdata/gpu_pl.done), so that its memory is not taken while the commit limit is low.
#   1. TabPFN-3.5 on the CPU at PhysioNet n = 3000, plasmode replicates 1-50 (D-03), refitting
#      records lost to host memory;
#   2. the classical arms of the IST temporal sensitivity split (D-05).
# A done marker is written only after a clean exit of its fit command; the gof stream waits for
# gpu.done, tfmcpu.done and cpu.done before its last pass.
#
#   started detached (WMI) by the author's coordinator session; resumable: run it again.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
export TFMCAL_TABPFN_VERSION=V3_5
export TFMCAL_TABPFN_SHA256=ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for _r in "/c/Program Files/R/R-4.4.1/bin" "/c/Program Files/R/R-4.4.1/bin/x64"; do
    [ -d "$_r" ] && PATH="$_r:$PATH"
done
export PATH
PYCPU="$REB/../.venv/Scripts/python.exe"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
LOGD="$REB/logs/realdata"
LOG="$LOGD/phase2.log"
RD="code/run_realdata.py"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] phase2: $*" >> "$LOG"; }

run_until_clean() {   # retry a fit command after any abnormal end (infrastructure, host memory)
    local tries=0 rc
    while :; do
        "$@" >> "$LOG" 2>&1
        rc=$?
        [ "$rc" -eq 0 ] && return 0
        tries=$((tries + 1))
        [ "$tries" -ge 300 ] && { say "giving up after $tries restarts (last exit $rc); no done marker"; return "$rc"; }
        say "exit $rc: restart $tries in 120 s"
        sleep 120
    done
}

say "waiting for the GPU plasmode pass (gpu_pl.done)"
until [ -f "$LOGD/gpu_pl.done" ]; do sleep 600; done

say "TabPFN-3.5 on the CPU, PhysioNet n = 3000, plasmode replicates 1-50"
run_until_clean "$PYGPU" $RD fit --part pl --keys physionet_ihd --ns 3000 --reps 1-50 --arms tabpfn \
    --group tfmcpu --device cpu --threads 6 --n-workers 1 --rerun-failed && touch "$LOGD/tfmcpu.done"

say "IST temporal split: classical arms (B = 100)"
run_until_clean "$PYCPU" $RD fit --part lct --keys all --ns all --reps 1-100 --arms classical \
    --group cpu --device cpu --threads 1 --n-workers 2 && touch "$LOGD/cpu.done"
say "done"
