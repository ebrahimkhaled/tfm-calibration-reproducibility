#!/usr/bin/env bash
# Second half of the timing benchmark, run after the GPU grid:
#   1) GPU memory probe at n = 3000 and 5000 (p = 20, 3,834 predicted rows)
#   2) LightGBM (CPU, 8 threads)
#   3) the three TFMs on CPU (8 torch threads, 240 s budget per fit)
# Every step resumes from its CSV, so the script can simply be restarted.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"
cd "$HERE"
PY=.venv-gpu/Scripts/python.exe

if [ ! -f results/bench/gpu_memprobe.csv ]; then
  $PY code/bench_01_python_arms.py --memprobe --models tabpfn_v2 tabicl_v2 mitra_v2
fi
$PY code/bench_01_python_arms.py --devices cpu --models lgbm
$PY code/bench_01_python_arms.py --devices cpu --models tabicl_v2 tabpfn_v2 mitra_v2 --threads 8 --cpu-budget 240
echo "bench_run_rest done"
