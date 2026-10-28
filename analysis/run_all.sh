#!/usr/bin/env bash
# Rebuild every table and figure of the analysis from the run's outputs, in one command:
#
#   bash analysis/run_all.sh                 # the full rebuild (incremental: only new files are read)
#   WORKERS=2 B=500 bash analysis/run_all.sh
#
# Steps: unit tests -> collect -> summarise (with the detection curves in R) -> meta-model of the
# random tier (if it has run; the synthetic test otherwise) -> every figure -> font/line check.
# Every Python and R process lowers itself to below-normal priority and one BLAS thread, and at most
# WORKERS (default 2) worker processes run at a time. Nothing is written outside analysis/.
# While the main run is going, keep WORKERS at 2. After it has finished, ANALYSIS_MAX_WORKERS=6
# WORKERS=6 lets the in-process scoring and the bootstrap use more cores.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REB="$(cd "$HERE/.." && pwd)"
cd "$REB"
PY="$REB/../.venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python
RS="Rscript"
WORKERS="${WORKERS:-2}"
B="${B:-500}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export R_LIBS="$REB/rlib" R_LIBS_USER="$REB/rlib"
mkdir -p "$HERE/logs"
LOG="$HERE/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
T0=$(date +%s)
step() { echo "[$(date '+%H:%M:%S') +$(( $(date +%s) - T0 )) s] $*" | tee -a "$LOG"; }

step "unit tests"
"$PY" analysis/tests/test_rules.py >> "$LOG" 2>&1 || { step "unit tests FAILED (see $LOG)"; exit 1; }

step "collect (workers $WORKERS)"
"$PY" analysis/collect.py --workers "$WORKERS" >> "$LOG" 2>&1

step "summarise (detection curves: B = $B)"
"$PY" analysis/summarise.py --detection-B "$B" --workers "$WORKERS" >> "$LOG" 2>&1

if [ -f analysis/tables/random_tier.parquet ]; then
    step "meta-model of the random-design tier"
    "$RS" analysis/meta_model.R >> "$LOG" 2>&1
    step "summarise again (Q6 needs the meta-model's transfer predictions)"
    "$PY" analysis/summarise.py --skip-detection >> "$LOG" 2>&1
else
    step "random tier not run yet: meta-model on the synthetic test data only"
    "$RS" analysis/make_synthetic_random.R >> "$LOG" 2>&1
    "$RS" analysis/meta_model.R --synthetic --max-designs 300 >> "$LOG" 2>&1
    "$PY" analysis/tests/check_meta_synthetic.py >> "$LOG" 2>&1 || step "  synthetic meta-model check FAILED"
fi

step "figures"
for f in analysis/figures/fig[0-9]*.py; do
    "$PY" "$f" >> "$LOG" 2>&1 || step "  $f FAILED (see $LOG)"
done
"$PY" analysis/figures/fig1_true_ici_vs_n.py --all-arms >> "$LOG" 2>&1 || true
"$PY" analysis/figures/fig3_min_n_table.py --lower-limit >> "$LOG" 2>&1 || true
for v in half double ll; do "$PY" analysis/figures/fig3_min_n_table.py --variant "$v" >> "$LOG" 2>&1 || true; done
for v in half double ll; do "$PY" analysis/figures/fig3b_share_reaching_class.py --variant "$v" >> "$LOG" 2>&1 || true; done
for n in 200 1000 3000; do "$PY" analysis/figures/fig6_recalibration_ladder.py --n "$n" >> "$LOG" 2>&1 || true; done
"$PY" analysis/figures/fig7_robustness_map.py --response D_logb >> "$LOG" 2>&1 || true
for r in D_ICI D_logb; do "$PY" analysis/figures/figS_meta_partial_effects.py --response "$r" >> "$LOG" 2>&1 || true; done

step "font and line-width check"
"$PY" analysis/figures/measure_fonts.py | tee -a "$LOG" || step "  some figure failed the check (see figures/out/font_report.csv)"
step "done in $(( $(date +%s) - T0 )) s; log: $LOG"
