#!/usr/bin/env bash
# Main run, schedule P4 (pre-registration 9.3). One call per stream; every stream resumes where it
# stopped, so a crashed or killed stream is restarted with the same command.
#
#   bash code/run_main.sh preflight     environment.json + checks (run once, first)
#   bash code/run_main.sh gpu           GPU process: TabICL, Mitra, TabPFN-3.5 (pi = 3 %, T2 n = 500),
#                                       legacy V2, then the two temperature arms
#   bash code/run_main.sh cpu_a         CPU worker A: TabPFN-3.5 on T2 n = 3000, core pi = 30 %, T1, T3
#   bash code/run_main.sh cpu_b         CPU worker B: TabPFN-3.5 on core pi = 10 %, T4a, T4b
#   bash code/run_main.sh classical     classical arms (R = 500), then the GOF layer and scoring,
#                                       streaming until the three model streams are done
#   bash code/run_main.sh random        the random-design tier (family R, one replicate per design);
#                                       each part waits for the main-run slot it takes over
#
# preflight may run at any time. Every other stream refuses to start until prereg/DEPOSITED.txt exists (the OSF registration link and its
# timestamp): the pre-registration must be deposited before the first main-run replicate is written.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB" || exit 1
source "$REB/env.sh"
export TFMCAL_TABPFN_VERSION=V3_5
export TFMCAL_TABPFN_SHA256=ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3

PYCPU="$REB/../.venv/Scripts/python.exe"
PYGPU="$REB/.venv-gpu/Scripts/python.exe"
LOGD="$REB/logs/main"
mkdir -p "$LOGD"
STREAM="${1:-}"
LOG="$LOGD/$STREAM.log"
RUN="code/run.py"                 # writes results/preds (main seed stream, never --smoke)

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $STREAM: $*" | tee -a "$LOG"; }
grp() { "$PYCPU" code/main_groups.py "$@"; }

if [ "$STREAM" != preflight ] && [ ! -s "$REB/prereg/DEPOSITED.txt" ]; then
    echo "Refusing to start: prereg/DEPOSITED.txt is missing. Deposit the pre-registration on OSF first" >&2
    echo "(prereg/HOW_TO_DEPOSIT.md), then write the registration link and its timestamp into that file." >&2
    exit 2
fi

case "$STREAM" in
preflight)
    "$PYGPU" code/write_environment.py >> "$LOG" 2>&1 || { say "environment check FAILED (see results/environment.json)"; exit 1; }
    Rscript R/verify_gof_build.R >> "$LOG" 2>&1 || { say "ebrahim.gof in rlib is not the build of commit 19480aa"; exit 1; }
    "$PYCPU" code/dgp_random.py check --workers 8 >> "$LOG" 2>&1 || { say "random designs do not re-solve to code/random_designs.csv"; exit 1; }
    Rscript R/tests/test_gof_layer.R >> "$LOG" 2>&1 || { say "GOF layer tests FAILED"; exit 1; }
    say "preflight ok"
    ;;

gpu)
    G="--device cuda --threads 6"
    say "TabICLv2 + Mitra-v2, every scenario (Mitra at T2 n = 3000 is recorded as not run)"
    "$PYGPU" $RUN --scenarios "$(grp all --n small)" --reps 1-200 --models tabicl,mitra $G >> "$LOG" 2>&1
    "$PYGPU" $RUN --scenarios "$(grp all --n 3000)"  --reps 1-150 --models tabicl,mitra $G >> "$LOG" 2>&1
    say "TabPFN-3.5 on core pi = 3 % and T2 n = 500"
    "$PYGPU" $RUN --scenarios "$(grp tabpfn_gpu --n small)" --reps 1-200 --models tabpfn $G >> "$LOG" 2>&1
    "$PYGPU" $RUN --scenarios "$(grp tabpfn_gpu --n 3000)"  --reps 1-150 --models tabpfn $G >> "$LOG" 2>&1
    say "legacy Real-TabPFN-2 (V2), core pi = 10 %, R = 100"
    "$PYGPU" $RUN --scenarios "$(grp legacy)" --reps 1-100 --models tabpfn_legacy $G >> "$LOG" 2>&1
    say "temperature arms (TabPFN-3.5 at T 0.9, TabICLv2 at T 1.0), 8 scenarios, R = 100"
    "$PYGPU" $RUN --scenarios "$(grp temperature)" --reps 1-100 --models tabpfn_t1,tabicl_t1 $G >> "$LOG" 2>&1
    touch "$LOGD/gpu.done"; say "done"
    ;;

cpu_a|cpu_b)
    C="--device cpu --threads 6 --tag tabpfn_cpu"
    g="tabpfn_$STREAM"
    say "TabPFN-3.5 on the CPU: $(grp $g)"
    # in group order (T2 n = 3000 first for worker A), each scenario at its own R
    for s in $(grp "$g" | tr ',' ' '); do
        case "$s" in *_n3000) R=1-150 ;; *) R=1-200 ;; esac
        "$PYGPU" $RUN --scenarios "$s" --reps "$R" --models tabpfn $C >> "$LOG" 2>&1
    done
    touch "$LOGD/$STREAM.done"; say "done"
    ;;

classical)
    say "classical arms, R = 500 (covariates stored on reps 1-200 only)"
    "$PYCPU" $RUN --scenarios all --reps 1-200   --models classical --n-workers 6 --threads 1 >> "$LOG" 2>&1
    "$PYCPU" $RUN --scenarios all --reps 201-500 --models classical --n-workers 6 --threads 1 --store-x none >> "$LOG" 2>&1
    touch "$LOGD/classical.done"
    SMALL="$(grp all --n small)"; BIG="$(grp all --n 3000)"
    gof_pass() {
        say "GOF pass"
        Rscript R/gof_layer.R run --scenario "$SMALL" --reps 1:200 --B 99 --workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
        Rscript R/gof_layer.R run --scenario "$BIG"   --reps 1:150 --B 99 --workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
        Rscript R/gof_layer.R run --scenario "$SMALL" --reps 201:500 --arms oracle --workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
        Rscript R/gof_layer.R run --scenario "$BIG"   --reps 151:500 --arms oracle --workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
        "$PYCPU" code/score.py --scenarios all --n-workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
    }
    while [ ! -f "$LOGD/gpu.done" ] || [ ! -f "$LOGD/cpu_a.done" ] || [ ! -f "$LOGD/cpu_b.done" ]; do
        gof_pass
        sleep 1800
    done
    gof_pass                      # the last pass sees every arm
    say "validation-size subsamples (battery only, real Y and Y*; e200 only at pi = 30 %)"
    BASE="oracle,lr,flic,ridge,gam,lgbm,lgbm_raw,tabpfn,tabicl,mitra,tabpfn_legacy,tabpfn_t1,tabicl_t1"
    for sub in e50 e200; do
        S1="$SMALL"; S3="$BIG"
        if [ "$sub" = e200 ]; then S1="$(grp pi30 --n small)"; S3="$(grp pi30 --n 3000)"; fi
        Rscript R/gof_layer.R run --subset $sub --scenario "$S1" --reps 1:200 --arms "$BASE" \
            --deepgof none --localize none --workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
        Rscript R/gof_layer.R run --subset $sub --scenario "$S3" --reps 1:150 --arms "$BASE" \
            --deepgof none --localize none --workers "${GOF_WORKERS:-6}" >> "$LOG" 2>&1
    done
    Rscript R/gof_layer.R check >> "$LOG" 2>&1
    touch "$LOGD/gof.done"; say "done"
    ;;

random)
    # The random-design tier (pre-registration 3.9): every design once, its own seed stream. Each part
    # waits for the main-run slot it takes over, so the stream can be started with the others:
    #   when cpu_b is done   the classical arms (6 workers), then TabPFN-3.5 on half of the CPU designs
    #   when cpu_a is done   TabPFN-3.5 on the other half of the CPU designs
    #   when gpu is done     TabICLv2 and Mitra-v2 on every design, TabPFN-3.5 on the GPU designs
    #   when gof is done     the test layer and scoring, a pass every 30 minutes until the fits are
    #                        done, then a last pass on 12 workers
    waitfor() { until [ -f "$LOGD/$1.done" ]; do sleep 600; done; }
    RAND="$(grp random)"
    say "random-design tier: $(echo "$RAND" | tr ',' '\n' | wc -l) designs; parts wait for their slots"
    (
        waitfor cpu_b
        say "random: classical arms"
        "$PYCPU" $RUN --scenarios random --reps 1 --models classical --n-workers 6 --threads 1 >> "$LOG" 2>&1
        touch "$LOGD/random_classical.done"
        say "random: TabPFN-3.5 on the CPU, part b"
        "$PYGPU" $RUN --scenarios "$(grp random_tabpfn_cpu_b)" --reps 1 --models tabpfn \
            --device cpu --threads 6 --tag tabpfn_cpu >> "$LOG" 2>&1
        touch "$LOGD/random_cpu_b.done"; say "random: CPU part b done"
    ) &
    (
        waitfor cpu_a
        say "random: TabPFN-3.5 on the CPU, part a"
        "$PYGPU" $RUN --scenarios "$(grp random_tabpfn_cpu_a)" --reps 1 --models tabpfn \
            --device cpu --threads 6 --tag tabpfn_cpu >> "$LOG" 2>&1
        touch "$LOGD/random_cpu_a.done"; say "random: CPU part a done"
    ) &
    (
        waitfor gpu
        say "random: TabICLv2 and Mitra-v2 (Mitra recorded as not run where it does not fit)"
        "$PYGPU" $RUN --scenarios random --reps 1 --models tabicl,mitra --device cuda --threads 6 >> "$LOG" 2>&1
        say "random: TabPFN-3.5 on the GPU designs"
        "$PYGPU" $RUN --scenarios "$(grp random_tabpfn_gpu)" --reps 1 --models tabpfn \
            --device cuda --threads 6 >> "$LOG" 2>&1
        touch "$LOGD/random_gpu.done"; say "random: GPU part done"
    ) &
    waitfor gof
    rpass() {
        say "random: test layer and scoring pass ($1 workers)"
        Rscript R/gof_layer.R run --scenario "$RAND" --reps 1:1 --B 99 --workers "$1" >> "$LOG" 2>&1
        "$PYCPU" code/score.py --scenarios random --n-workers 6 >> "$LOG" 2>&1
    }
    while [ ! -f "$LOGD/random_classical.done" ] || [ ! -f "$LOGD/random_cpu_a.done" ] || \
          [ ! -f "$LOGD/random_cpu_b.done" ] || [ ! -f "$LOGD/random_gpu.done" ]; do
        rpass 6
        sleep 1800
    done
    wait
    rpass 12                      # the last pass sees every arm
    Rscript R/gof_layer.R check --scenario "$RAND" >> "$LOG" 2>&1
    touch "$LOGD/random.done"; say "done"
    ;;

*)
    echo "usage: bash code/run_main.sh preflight|gpu|cpu_a|cpu_b|classical|random" >&2
    exit 64
    ;;
esac
