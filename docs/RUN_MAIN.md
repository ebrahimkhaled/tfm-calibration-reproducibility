# Starting the main run (schedule P4, with the random-design tier)

**Do not start before the pre-registration is registered on OSF and approved** (`prereg/HOW_TO_DEPOSIT.md`). Every stream refuses to start until `prereg/DEPOSITED.txt` exists. The design is in pre-registration Sections 3.9, 6.3 and 9.3; the code is `code/run_main.sh`.

## 0. Before you start (5 minutes)

1. Check that the machine is free:
   - the GPU should show only the desktop (`nvidia-smi`, about 0.8-1.0 GB used);
   - CPU load should be under about 20 % (Task Manager).

   The run takes the GPU and about 19 of the 24 threads for about 3.6 days in all: about 3.4 days (82 h) for the 58 scenarios, then about 4.3 hours more for the random-design tier (`results/random_budget.md`).
2. Never add, copy or delete files in `rebuild/models/` while the run is going. The TabPFN checkpoint is also pinned by SHA-256, so a changed file stops the run.
3. The environment comes from `rebuild/env.sh`. `run_main.sh` sources it and then sets:
   - `TFMCAL_TABPFN_VERSION=V3_5`;
   - `TFMCAL_TABPFN_SHA256=ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.

## 1. The start command

Open Git Bash. Then run, as one paste:

```bash
cd path/to/this/archive
bash code/run_main.sh preflight && {
  nohup bash code/run_main.sh gpu       > logs/main/gpu.out       2>&1 &
  nohup bash code/run_main.sh cpu_a     > logs/main/cpu_a.out     2>&1 &
  nohup bash code/run_main.sh cpu_b     > logs/main/cpu_b.out     2>&1 &
  nohup bash code/run_main.sh classical > logs/main/classical.out 2>&1 &
  nohup bash code/run_main.sh random    > logs/main/random.out    2>&1 &
}
```

- **preflight** writes `results/environment.json`. It checks every package version and checkpoint SHA-256 against pre-registration Section 13, checks that `ebrahim.gof` in `rlib` is the build of commit 19480aa (`R/verify_gof_build.R`), re-solves the random designs and compares them with `code/random_designs.csv` (about 10 minutes on 8 cores), and runs the test-layer unit tests. It stops on any mismatch, and then nothing else starts.
- **The five streams** run in the background (`nohup`). On Windows, Git Bash can still end its child processes when its window closes, so leave that window open (minimized) for the whole run. Each writes a progress log to `logs/main/<stream>.log`, and the per-replicate run logs go to `logs/run_*.jsonl`. Failures are appended to `logs/failures.jsonl`.

| Stream | What it does | Threads | Expected |
|---|---|---|---|
| `gpu` | TabICLv2 + Mitra-v2 on all 58 scenarios (Mitra at T2_pi10_n3000 is recorded as "not run"); TabPFN-3.5 on core pi = 3 % and T2_pi10_n500; legacy V2 (R = 100); `tabpfn_t1` + `tabicl_t1` (R = 100) | GPU + 6 | about 79 h |
| `cpu_a` | TabPFN-3.5 on the CPU: T2_pi10_n3000 first, then core pi = 30 %, T1, T3 (tag `tabpfn_cpu`) | 6 | 63-77 h |
| `cpu_b` | TabPFN-3.5 on the CPU: core pi = 10 %, T4a, T4b (tag `tabpfn_cpu`) | 6 | 57-71 h |
| `classical` | the classical arms, R = 500 (about 9 h); then a pass of the test layer (B = 99) and the scoring every time the previous pass ends (at least 30 minutes apart), until the three model streams are done; then a final pass and the e50/e200 subsamples | 6 workers | ends a few hours after the last model stream (about 82 h) |
| `random` | the random-design tier, 1,000 designs x 1 replicate (pre-registration 3.9). It starts at once and waits: when `cpu_b` is done, the classical arms (6 workers, about 0.8 h) and then TabPFN-3.5 on half of the CPU designs; when `cpu_a` is done, the other half; when `gpu` is done, TabICLv2 + Mitra-v2 on every design (Mitra "not run" where it does not fit) and TabPFN-3.5 on the GPU designs (about 6.8 GPU-h); when `classical` is done (`gof.done`), the test layer and the scoring every 30 minutes, then a last pass on 12 workers | the slots it takes over | ends about 4.3 h after `classical` |

The scenario groups, and the random designs' device groups, are printed by `python code/main_groups.py --table`.

## 2. Resuming

Every stream resumes. If the machine restarts, or a stream dies, run the same line again, for example:

```bash
cd path/to/this/archive
nohup bash code/run_main.sh cpu_a > logs/main/cpu_a.out 2>&1 &
```

What happens on a restart:
- **Finished replicates are skipped.** A replicate whose requested arms are all recorded (ok, degenerate, failed or not run) is not refitted.
- **Lost work is small.** A replicate that was mid-fit is refitted from its seed, with the same data, which the fingerprint check confirms.
- **Stale locks clear themselves.** A lock left by a dead process is broken at once.
- **The test layer** adds only the missing arm x outcome pairs.

## 3. Checking progress

```bash
cd path/to/this/archive
tail -n 3 logs/main/*.log                        # where each stream is
find results/preds -name "*.parquet" | wc -l    # prediction files so far
tail -n 5 logs/failures.jsonl                    # recent failed fits (expected at n = 200, pi = 3 %)
ls logs/main/*.done                              # finished streams (and parts of the random stream)
find results/preds -path "*/R[0-9]*" -name "*.parquet" | wc -l   # random-design files so far
```

## 4. The 24-hour balance check (pre-registration 9.3)

After about 24 hours, compare the streams' progress.

**If a CPU worker is clearly behind the GPU:**
1. Stop that worker: `kill` its bash process, which you can find with `ps`.
2. Hand its **unstarted** scenarios, whole, to the GPU. For example:

   ```bash
   .venv-gpu/Scripts/python.exe code/run.py --scenarios S4_pi30_n1000,S4_pi30_n3000 --reps 1-200 --models tabpfn --device cuda --threads 6
   ```

   Use `--reps 1-150` for n = 3000 scenarios.
3. Restart the worker without them. You can edit the group in `code/main_groups.py`; the assertion there must still pass.

**Never let two processes fit `tabpfn` for the same scenario at once.** The GPU file and the CPU tag file use different locks. The test layer stops with "appear in two files of one replicate" if it happens.

**Record it.** Write down which scenarios moved. The device of every scenario is reported in the paper.

**The random stream follows by itself.** Its CPU parts start when `cpu_a.done` and `cpu_b.done` appear, so a worker that was stopped and restarted hands its slot over only when it has really finished. Its device groups are fixed by the design's size (pre-registration 3.9) and are never moved.

## 5. When it is done

The `logs/main/*.done` files mark the end of the streams; `gof.done` ends the 58 scenarios, and `random.done` comes last. Then:

```bash
Rscript R/gof_layer.R check                      # fingerprints and open errors per scenario and design
../.venv/Scripts/python.exe code/score.py --scenarios all --n-workers 6
../.venv/Scripts/python.exe code/score.py --scenarios random --n-workers 6
```

After that, analysis phase P6 begins.
