#!/usr/bin/env bash
# Rebuild the real-data results digest (analysis/tables/realdata/REALDATA_DIGEST.md) every two hours
# while the real-data streams run, and once more when the run is complete (the gof stream writes
# analysis/tables/realdata/COMPLETE after its last pass, scoring and summary tables).
#
#   bash code/realdata_digest_loop.sh        (started detached by code/launch_realdata.ps1 digest)

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYCPU="$REB/../.venv/Scripts/python.exe"
LOG="$REB/logs/realdata/digest.log"
FLAG="$REB/analysis/tables/realdata/COMPLETE"

until [ -f "$FLAG" ]; do
    "$PYCPU" analysis/realdata_digest.py >> "$LOG" 2>&1
    for _ in $(seq 1 24); do [ -f "$FLAG" ] && break; sleep 300; done
done
"$PYCPU" analysis/realdata_digest.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] digest: final version written" >> "$LOG"
# final figure (no watermark once COMPLETE exists) and the font / stroke / width check of every figure
"$PYCPU" analysis/figures/fig8_realdata.py >> "$LOG" 2>&1
"$PYCPU" analysis/figures/measure_fonts.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] digest: fig8 rendered, measure_fonts exit $?" >> "$LOG"
