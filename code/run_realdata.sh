#!/usr/bin/env bash
# Real-data stream (pre-registration Section 10). One call per stream; every stream resumes where it
# stopped (one file per replicate, locks, fingerprints), so a killed stream is restarted with the
# same command.
#
#   bash code/run_realdata.sh gpu      foundation models on the GPU: the plasmode tier (TabPFN-3.5,
#                                      TabICLv2, Mitra-v2; R = 50), then the IST temporal split
#                                      (the learning curves of the main arms are finished; the side
#                                      arms are dropped on real data, DEVIATIONS D-05)
#   bash code/run_realdata.sh tfmcpu   TabPFN-3.5 at PhysioNet n = 3000 on a CPU worker (does not fit
#                                      the 6 GB card; D-03): refits records lost to host memory
#   bash code/run_realdata.sh cpu      classical arms of the IST temporal split
#   bash code/run_realdata.sh gof      R battery (IST, CRASH-2, PhysioNet) + scoring + summary tables,
#                                      a pass every 30 minutes (learning curves first, then plasmode)
#                                      until the fit streams are done, then a last pass
#   bash code/run_realdata.sh gof_sepsis   the same R battery for Sepsis alone (one memory-heavy worker)
#   bash code/run_realdata.sh status   progress
#
# Logs: logs/realdata/<stream>.log (progress), logs/realdata/fit_*.jsonl (one line per replicate),
# logs/realdata/gof.log (R layer). Done markers: logs/realdata/<stream>.done.
# Infrastructure errors (CUDA runtime errors, crashed Rscript) leave the arm unrecorded and the fit
# command exits with 75; the stream then restarts it after a minute.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
export TFMCAL_TABPFN_VERSION=V3_5
export TFMCAL_TABPFN_SHA256=ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # per worker (INCIDENTS 2026-10-14)
export TFM_REBUILD_ROOT="$REBUILD_W"

# R on the PATH even when the stream is started outside a login shell (WMI process creation)
for _r in "/c/Program Files/R/R-4.4.1/bin" "/c/Program Files/R/R-4.4.1/bin/x64"; do
    [ -d "$_r" ] && PATH="$_r:$PATH"
done
export PATH
PYCPU="$REB/../.venv/Scripts/python.exe"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
LOGD="$REB/logs/realdata"
mkdir -p "$LOGD"
STREAM="${1:-}"
LOG="$LOGD/$STREAM.log"
RD="code/run_realdata.py"
CPU_WORKERS="${RD_CPU_WORKERS:-2}"         # classical arms (about 1 GB each with the Rscript call)
TFMCPU_WORKERS="${RD_TFMCPU_WORKERS:-1}"   # CPU TabPFN workers, 6 threads, about 5 GB each

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $STREAM: $*" | tee -a "$LOG"; }

# Free commit (GB) of the machine: the commit limit minus the committed bytes. About 33 GB of the
# 71 GB limit is held outside any process (INCIDENTS 2026-10-16), so heavy steps wait for room.
commit_free_gb() {
    local v
    v="$(powershell -NoProfile -Command "\$c=(Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit').CounterSamples; [int](((\$c[1].CookedValue)-(\$c[0].CookedValue))/1GB)" 2>/dev/null | tr -cd '0-9')"
    echo "${v:-0}"
}
wait_free() {   # $1 = GB of free commit needed before the next heavy step starts
    local f
    # RD_NO_COMMIT_GATE=1: skip the gate (a system-managed page file grows on demand, so the current
    # limit understates the room; host-memory errors are retried, never stored)
    [ "${RD_NO_COMMIT_GATE:-0}" = 1 ] && return 0
    while :; do
        f="$(commit_free_gb)"; f="${f:-0}"
        [ "$f" -ge "$1" ] 2>/dev/null && return 0
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $STREAM: waiting for $1 GB of free commit (now $f GB)" >> "$LOG"
        sleep 300
    done
}

# run one fit command until it ends cleanly: restart after an infrastructure error (exit 75) or
# any abnormal end (host memory, a crash), at most 200 times; $1 = GB of free commit it needs
fit() {
    local need="$1"; shift
    local tries=0 rc
    while :; do
        wait_free "$need"
        "$@" >> "$LOG" 2>&1
        rc=$?
        [ "$rc" -eq 0 ] && return 0
        tries=$((tries + 1))
        [ "$tries" -ge 200 ] && { say "giving up after $tries restarts (last exit $rc)"; return "$rc"; }
        say "exit $rc: restart $tries in 120 s"
        sleep 120
    done
}

GPU="--group gpu --device cuda --threads 6"
# Plan after the author's decisions of 2026-10-23 (DEVIATIONS D-05): the side arms are dropped on all
# real data; the plasmode tier runs the three main TFMs at R = 50 (reps 1-50; the classical arms
# were already fitted at R = 200); the IST temporal sensitivity split (part lct) runs last.
case "$STREAM" in
gpu)
    say "plasmode: TabPFN-3.5, TabICLv2, Mitra-v2 (R = 50)"
    fit 9 "$PYGPU" $RD fit --part pl --keys all --ns all --reps 1-50 --arms tfm $GPU --rerun-failed
    touch "$LOGD/gpu_pl.done"
    say "IST temporal sensitivity split: TabPFN-3.5, TabICLv2, Mitra-v2 (B = 100)"
    fit 9 "$PYGPU" $RD fit --part lct --keys all --ns all --reps 1-100 --arms tfm $GPU --rerun-failed
    touch "$LOGD/gpu.done"; say "done"
    ;;

tfmcpu)
    C="--group tfmcpu --device cpu --threads 6 --n-workers $TFMCPU_WORKERS --rerun-failed"
    say "TabPFN-3.5 at PhysioNet n = 3000 on $TFMCPU_WORKERS CPU worker (refits memory-failed records)"
    fit 7 "$PYGPU" $RD fit --part lc --keys physionet_ihd --ns 3000 --reps 1-100 --arms tabpfn $C
    touch "$LOGD/tfmcpu_lc.done"
    fit 7 "$PYGPU" $RD fit --part pl --keys physionet_ihd --ns 3000 --reps 1-50 --arms tabpfn $C
    touch "$LOGD/tfmcpu.done"; say "done"
    ;;

cpu)
    C="--group cpu --device cpu --threads 1 --n-workers $CPU_WORKERS"
    say "IST temporal sensitivity split: classical arms (B = 100)"
    fit 4 "$PYCPU" $RD fit --part lct --keys all --ns all --reps 1-100 --arms classical $C
    touch "$LOGD/cpu.done"; say "done"
    ;;

gof|gof_sepsis)
    # Two R lanes run side by side: "gof" (IST, CRASH-2, PhysioNet; it also scores and writes the
    # tables) and "gof_sepsis" (Sepsis alone). R workers per data set are bounded by memory: le Cessie
    # builds n_eval x n_eval matrices (about 5.6 GB per call at Sepsis' 18,551 rows, 2.4 GB at IST's
    # 12,000, under 0.5 GB elsewhere). A test that cannot get memory stops its replicate, which is
    # then computed again in the next pass (R/realdata_gof.R).
    w_of() { case "$1" in sepsis_death) echo "${RD_GOF_W_SEPSIS:-1}" ;; ist_*) echo "${RD_GOF_W_IST:-3}" ;;
             *) echo "${RD_GOF_W_SMALL:-4}" ;; esac; }
    if [ "$STREAM" = gof_sepsis ]; then KEYS="sepsis_death"; else KEYS="ist_death14 ist_dd6 crash2_death28 physionet_ihd"; fi
    rpass() {   # $1 = parts (lc or pl)
        for k in $KEYS; do
            [ "$1" = lct ] && [ "${k#ist_}" = "$k" ] && continue      # the temporal split is IST only
            case "$k" in sepsis_death) wait_free 8 ;; ist_*) wait_free 7 ;; *) wait_free 3 ;; esac
            GOF_FORCE_WORKERS="$(w_of $k)" Rscript R/realdata_gof.R run --keys "$k" --parts "$1" --B 99 >> "$LOG" 2>&1
        done
    }
    tables() {
        [ "$STREAM" = gof ] || return 0
        "$PYCPU" $RD score --keys all --n-workers 4 >> "$LOG" 2>&1
        "$PYCPU" $RD summarise >> "$LOG" 2>&1
        "$PYCPU" analysis/figures/fig8_realdata.py >> "$LOG" 2>&1
    }
    fits_done() { [ -f "$LOGD/gpu.done" ] && [ -f "$LOGD/tfmcpu.done" ] && [ -f "$LOGD/cpu.done" ]; }
    while ! fits_done; do
        say "GOF pass ($KEYS): learning curves"; rpass lc
        [ "$STREAM" = gof ] && { say "GOF pass ($KEYS): plasmode"; rpass pl; say "GOF pass: IST temporal split"; rpass lct; }
        tables
        sleep 1800
    done
    say "last GOF pass ($KEYS)"; rpass lc; [ "$STREAM" = gof ] && { rpass pl; rpass lct; }
    touch "$LOGD/$STREAM.done"
    if [ "$STREAM" = gof ]; then
        until [ -f "$LOGD/gof_sepsis.done" ]; do sleep 600; done
        tables
        Rscript R/realdata_gof.R check >> "$LOG" 2>&1
        touch "$REB/analysis/tables/realdata/COMPLETE"
        "$PYCPU" analysis/figures/fig8_realdata.py >> "$LOG" 2>&1
    fi
    say "done"
    ;;

status)
    "$PYCPU" $RD status
    ls "$LOGD"/*.done 2>/dev/null
    ;;

*)
    echo "usage: bash code/run_realdata.sh gpu|tfmcpu|cpu|gof|gof_sepsis|status" >&2
    exit 64
    ;;
esac
