#!/usr/bin/env bash
# Finish of the real-data run under the author's light-load rules (2026-10-27; the GTX 1660 Ti also
# drives the display and the PC hung under full GPU load):
#   1. the one replicate left on the GPU (ist_dd6, temporal split, n = 3000, rep 100) with a light
#      load: chunked prediction from the start (1,000 rows), PyTorch memory fraction 0.8 (0.7 did not hold TabPFN-3.5 at n = 3000), 2 threads;
#   2. the R battery with at most 2 R workers (about 10 GB at IST's 14,500 evaluation rows); the
#      Sepsis lane has no work left and only makes its last pass; then the gof stream writes the
#      final tables, the COMPLETE marker and fig8.
# Started through WMI at below-normal priority (children inherit it); resumable: run it again.

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
LOGD="$REB/logs/realdata"
LOG="$LOGD/finish.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] finish: $*" >> "$LOG"; }

if [ ! -f "$LOGD/gpu.done" ]; then
    say "last GPU replicate (light load)"
    for i in $(seq 1 20); do
        REALDATA_TFM_CHUNK=1000 TFMCAL_GPU_FRACTION=0.8 "$REB/.venv-gpu/Scripts/python.exe" code/run_realdata.py fit \
            --part lct --keys ist_dd6 --ns 3000 --reps 100 --arms tfm --group gpu --device cuda --threads 2 \
            --rerun-failed >> "$LOG" 2>&1 && { touch "$LOGD/gpu.done"; break; }
        say "GPU fit exit $?: retry $i in 120 s"; sleep 120
    done
fi
[ -f "$LOGD/gpu.done" ] || { say "GPU replicate not finished; stopping"; exit 1; }

say "R battery: gof (IST 2 workers, small 2) and the Sepsis lane's last pass"
RD_NO_COMMIT_GATE=1 RD_GOF_W_SEPSIS=1 bash code/run_realdata.sh gof_sepsis >> "$LOGD/gof_sepsis.out" 2>&1 &
RD_NO_COMMIT_GATE=1 RD_GOF_W_IST=2 RD_GOF_W_SMALL=2 bash code/run_realdata.sh gof >> "$LOGD/gof.out" 2>&1
wait
say "done (COMPLETE: $([ -f analysis/tables/realdata/COMPLETE ] && echo yes || echo no))"
