# Simulation engine (Python)

This folder implements REBUILD_PLAN section 4 (ADEMP) and pre-registration sections 3-5. It covers the data-generating mechanisms, the model arms, the per-replicate prediction files, the recalibration ladder and the oracle estimands. The goodness-of-fit layer is in `../R/gof_layer.R` (another stream), and it reads the files written here.

| File | What it does |
|---|---|
| `dgp.py` | The 58 scenarios. It solves kappa and alpha, draws the replicates and fingerprints them. `python dgp.py build` writes `scenarios.csv`. |
| `scenarios.csv` | One row per `scenario_id` (1-48 core, 49-58 stress). Columns: kappa, alpha, realised prevalence, oracle AUC, EPV, Riley minimum n (Python, exact likelihood) and `riley_n_min_pmsampsize` (R `pmsampsize`, from the c-statistic, as pre-registered). This is the Appendix A table of the pre-registration. |
| `models.py` | One wrapper per arm. Failures come back as `status = "failed"` with the error text. |
| `fit_r_arms.R` | LR-MLE (`glm`), FLIC (`logistf`, `flic = TRUE`), ridge-CV (`glmnet`, 10 stratified folds) and GAM (`mgcv`, REML, `s(x, k = 5)`). There is one `Rscript` call per replicate, with a file hand-off and no rpy2. |
| `riley_pmsampsize.R` | Adds the `pmsampsize` column to `scenarios.csv` (called by `dgp.py build`). |
| `recal.py` | Recalibration ladder R1-R3 on the first m in {100, 500} rows of the recalibration set. |
| `metrics.py` | Oracle and empirical estimands, the hierarchy class and the MCSE helpers. |
| `run.py` | The loop over scenario x replicate. It is resumable and lock-safe, and writes `results/preds/<scenario>/<rep>.parquet`. |
| `score.py` | Applies `metrics.py` to every stored column (all arms and ladder levels) and writes `results/metrics/<scenario>.parquet`. |
| `tests/run_tests.py` | The unit tests (no pytest in the venvs); `tests/test_random_designs.py` covers the random-design tier. `tests/check_resume.py` checks the locks, resume after a hard kill, and regeneration. |
| `main_groups.py` | Scenario groups of the main run (schedule P4): which scenarios TabPFN-3.5 fits on the GPU and on each CPU worker, the legacy and temperature groups. |
| `run_main.sh` | The main run, one stream per call (`preflight`, `gpu`, `cpu_a`, `cpu_b`, `classical`, and `random` for the random-design tier). It refuses to start before `prereg/DEPOSITED.txt` exists. See `../RUN_MAIN.md`. |
| `write_environment.py` | Writes `results/environment.json` and checks the versions and checkpoint SHA-256 against pre-registration Section 13. |
| `smoke_freeze.sh`, `smoke_freeze_summary.py` | The pre-freeze smoke run of 2026-10-07 (every arm, 2 scenarios x 2 replicates, smoke stream) and its summary in `results/smoke_freeze/SUMMARY.md`. |
| `time_tabpfn35_fast.py` | Times TabPFN-3.5 standard against Fast: `results/tabpfn35_fast_timing.csv`. |
| `dgp_random.py`, `random_designs.csv` | The random-design tier, scenario family R (pre-registration 3.9): D designs `R0001`..., one replicate each, every design with its own event rate, n, p, correlation, AUC target, shape severities and discrete fraction. `python dgp_random.py build --D 1000` solves kappa and alpha per design; `check` re-solves and compares. Also the device rules (`tabpfn_on_gpu`, `mitra_fits`). |
| `probe_random_memory.py` | GPU memory and time of the three TFMs at (n, p, predicted rows) cells: `results/random_memory_probe.csv`, the basis of the device rules. |
| `random_budget.py` | The tier's compute budget from measured per-fit costs, for a choice of D: `results/random_budget.md`. |
| `smoke_freeze_b.sh` | The second pre-freeze smoke (2026-10-07 b): 2 core scenarios x 2 replicates and R0001-R0005, every arm, scoring and the GOF layer at ebrahim.gof 19480aa; `results/smoke_freeze_b/`. |
| `audit_calibration.py` | Calibration audit of any model's (y, p[, X]) in one call: runs `R/audit_calibration.R` (`run.all.external()` + `localize.external()`); a command-line tool and a function `audit()` (deliverable b of pre-registration 14). |
| `benchmark_loader.py` | Loader of the released benchmark: a replicate's predictions (wide or long), iteration over a family, the study's GOF rows (deliverable a). |
| `get_tabpfn_weights.py` | Downloads the TabPFN checkpoints after the user has accepted the licence, and checks their SHA-256; prints the licence notice. TabPFN weights are never archived. |

The `bench_*.py` and `realdata.py` files belong to other streams.

## Data-generating mechanisms

These follow pre-registration section 3 to the letter. Section 3.2 is still marked "AUTHOR TO CONFIRM"; if the author changes a coefficient, edit `dgp.py`, rerun `python dgp.py build`, and discard any results already produced.

- Covariates: x1..x6 are N(0, 1) with corr(x1, x2) = 0.5; x7 ~ Bern(0.5); x8 ~ Bern(0.3).
- Linear term: L = 1.0 x1 + 0.8 x2 + 0.6 x3 + 0.4 x4 + 0.2 x5 + 0 x6 + 0.5 x7 + 0.3 x8.
- Shapes:
  - S1: linear logit.
  - S2: L + gamma((x1^2 - 1)/sqrt 2 + (x3^3 - 3 x3)/sqrt 6), with gamma = 0.5.
  - S3: L + 0.5 x1 x7 + 0.5 x3 x4. The true subgroup is x7 == 1.
  - S4: cloglog link.
- Stress families:
  - T1: t3 margins through a Gaussian copula, scaled to unit variance.
  - T2: 22 extra N(0, 1) noise columns (p = 30).
  - T3: x8 becomes a 6-level categorical (codes 0-5, equal probabilities, effects 0, .3, -.3, .6, -.6, .9) on the S3 shape.
  - T4a and T4b: S2 with gamma = 0.25 and 1.0.
- kappa is solved per family so that the oracle AUC is 0.80 at pi = 10 %. Then alpha is solved per (family, pi) so that E[p*] = pi. Both are solved on 10^6 fixed calibration draws.
  - kappa is solved separately for every stress family. T2 shares S1's values exactly, because the noise columns do not enter p*.
  - The realised prevalence equals pi to 1e-9.
- Seeds:
  - Main run: `SeedSequence([scenario_id, n_index, rep])`, spawned into development, evaluation and recalibration streams.
  - Smoke: `[..., 777]`.
  - Probe: `[scenario_id, 99999]`.
  - Model seeds and null-outcome seeds come from the same entropy with spawn keys (4, arm_code) and (3, arm_code). The arm codes are fixed in `models.ARM_CODES`.
- Fingerprints: SHA-256 of the bytes of (X, p*, Y) of each set, plus a combined one.
  - When arms are added to an existing file, the regenerated data must match the stored fingerprint.
  - `run.py` refuses two replicates with identical data in one run (the RNG-reset guard).
  - `score.py` re-checks the stored outcomes against the regenerated replicate.

## Random-design tier (family R; added 2026-10-07 b)

- **Designs.** `random_designs.csv` holds D solved designs (`scenario_id` = 1000 + d, names `R0001`...). Design d draws its setting from `SeedSequence([4242, 0, d])` and its calibration draws from `[4242, 1, d]`, so adding designs never changes earlier ones. The table is never part of `scenarios.csv`, of `--scenarios all` or of the core groups.
- **Data.** `SeedSequence([4242, 2, d, rep])` (smoke adds 777), spawned into development, evaluation and (empty) recalibration streams as in the core design. The evaluation set has max(1000, ceil(100/pi)) rows; there is no recalibration set, no probe set and no ladder (`run.py` skips the ladder for family R).
- **Shape.** L (coefficients drawn per design, rescaled to the core SD 1.755) + w_nl x S2's non-linear term + w_int x S3's interaction term, both from `dgp.shape_score` on the design's predictors in the core roles, and a mixture of the logit and cloglog inverse links with weight w_link.
- **Hooks in the engine.** `dgp.get_scenario`, `dgp.replicate_entropy`, `dgp._draw_set` and `dgp.make_probe` hand family R to `dgp_random`; the core branches are untouched (a unit test checks a stored core fingerprint bit for bit). `run.parse_scenarios` accepts `random`, `R0001-R0005` and `R0007`. `run.not_run_reason` records Mitra-v2 as "does not fit in 6 GB GPU memory" where n^2 x p > 8.0e7 (the T2 rule).
- **Device of TabPFN-3.5.** On the GPU when (n + N_eval) x p <= 80,000, else on a CPU worker (`main_groups.py random_tabpfn_gpu`, `random_tabpfn_cpu_a`, `random_tabpfn_cpu_b`). Both rules come from `results/random_memory_probe.csv` and the pilot.
- **Running.** `bash code/run_main.sh random` (see `../RUN_MAIN.md`); by hand, for example `run.py --scenarios R0001-R0010 --reps 1 --models classical`.

## Prediction file contract (`results/preds/<scenario>/<rep:04d>.parquet`)

Each file holds one row group. Rows:

| `set` | rows | columns |
|---|---|---|
| `eval` | N_eval = max(1000, ceil(100/pi)) | `row`, `y`, `p_star` (float64), `in_sub_e50`, `in_sub_e200`, `x1..xp` (float32), `p_<arm>` (float32), `ystar_<arm>` (Int8), `p_<arm>__R{1,2,3}_m{100,500}` (float32) |
| `recal` | 500 | `row`, `y`, `p_star`, `p_<arm>` |
| `probe` | 500 (fixed per scenario) | `row`, `p_star`, `p_<arm>`; the probe X is in `results/probe/<scenario>.parquet` |

- The oracle has no `p_` column: it *is* `p_star`. It does have `ystar_oracle`. The R reader adds the oracle from `p_star`.
- `ystar_<arm>` ~ Bern(stored p-hat) is drawn on the evaluation rows only, from the arm's own seed.
- `in_sub_e50` and `in_sub_e200` mark the first ceil(50/pi) and ceil(200/pi) evaluation rows (nested validation subsamples).
  - When the subsample would be larger than the evaluation set, it is all False, and `sub_sizes` in the metadata is null.
  - This is the case for 200 events at pi = 3 % and 10 %.
- In T3, `x8` holds the integer code of the categorical (0-5). The metadata `col_types` says which columns are continuous, binary and categorical. The R layer must turn x8 into a factor in T3 before localization or DeepGOF.
- The metadata key `tfmcal` (JSON) holds:
  - scenario, rep, entropy, fingerprints, event counts and sizes;
  - per output: status, error, seconds, flags (separation, convergence, lambda, grid, Platt fit), checkpoint file, SHA-256 and revision, `n_estimators`, `softmax_temperature`, peak GPU memory, and `n_boundary` (predictions within 1e-6 of 0 or 1 before the float32 cast);
  - the ladder fits;
  - a history of every write (time, host, pid, arms, code hash).
- A failed arm has no `p_` column. Its failure is in the metadata and in `logs/failures.jsonl`. It is never averaged over.

The R layer's reader (`R/gof_layer.R::read_rep`) parses these files as they are. This was checked on the smoke files. It lower-cases the column names, so the ladder arms appear as `flic__r2_m500` there.

## Arms

| Arm | Implementation | State on 2026-10-06 |
|---|---|---|
| `oracle` | p* | ok |
| `lr`, `flic`, `ridge`, `gam` | R, `fit_r_arms.R`, packages from `rebuild/rlib` (logistf 1.26.1, glmnet 4.1-10, mgcv 1.9-4) | ok. FLIC agrees with the Python `flic_py` to 1e-5 and LR with `lr_py` to 1e-6 (unit test). |
| `lgbm` (+ `lgbm_raw`) | Python LightGBM 4.7.0: 5-fold stratified CV over num_leaves {7, 15, 31} x min_data_in_leaf {10, 30}, learning rate 0.05, at most 2,000 rounds with early stopping after 50; refit on all rows; out-of-fold Platt | ok |
| `tabpfn` | The checkpoint pinned by `TFMCAL_TABPFN_VERSION`; for the main run, `V3_5` = `tabpfn-v3.5-20260909.safetensors` (SHA-256 ece4d67e...0be3, also checked through `TFMCAL_TABPFN_SHA256`). Unpinned, the newest file present wins (order: 3.5, 3.5-Fast, 3, 2.6, 2.5, 2). | ok; shipped softmax temperature 1.0 |
| `tabpfn_legacy` | `ModelVersion.V2`, the same default checkpoint | ok (CPU-tested) |
| `tabpfn_t1` | Temperature arm: the `tabpfn` checkpoint (TabPFN-3.5) at `softmax_temperature = 0.9`, with the `tabpfn` seed. The name is kept from the plan. | ok; logit slope against `tabpfn` = 1/0.9 (unit test) |
| `tabicl_t1` | Temperature arm: TabICLv2 at `softmax_temperature = 1.0` (it ships 0.9), with the `tabicl` seed. | ok; logit slope 1/0.9 (unit test) |
| `tabicl` | `tabicl` 2.2.0, `tabicl-classifier-v2-20260212.ckpt` from the project HF cache | ok in `.venv-gpu` (CPU-tested, 40 rows) |
| `mitra` | `MitraClassifier`, `autogluon/mitra-classifier-2` (rev edada0d), `fine_tune=False`, the whole development set as context (a 16-row validation set is passed only so that AutoGluon does not hold out 20 %) | ok in `.venv-gpu` (CPU-tested, 40 rows); about 1 s per predicted row on CPU, so GPU only |

Changes of 2026-10-07 (author decisions before the pre-registration freeze; `prereg/preregistration.md` Section 0.2):
- **LightGBM + Platt.** A Platt slope <= 0 keeps the raw LightGBM scores and sets `platt_fallback` (`models.platt_or_raw`). A Platt fit that does not converge is still a failure.
- **Ridge-CV.** The lambda path is 200 log-spaced values from glmnet's lambda_max down to 1e-6 of it, with glmnet's early path stop off (`glmnet.control(fdev = 0)`, reset after the fit). The ridge info records `lambda_min_index`, `lambda_at_edge` and `lambda_at_lower_edge`. On 50 smoke fits the optimum was interior in 49; the one exception sat at the upper end (the null model).
- **Degenerate fits.** A fit that ran is stored with `status = "degenerate"` when more than 100 evaluation predictions lie within 1e-6 of 0 or 1 (`n_boundary_eval`), or when LR-MLE shows separation. It is scored, gets Y\* and its ladder, and is used like "ok" (`models.USABLE_STATUSES`). `score.py` writes `degenerate`, `degenerate_reason`, `platt_fallback` and `lambda_at_edge` columns, and counts degenerate and failed fits per scenario.
- **Mitra-v2 at T2_pi10_n3000** is recorded as `status = "not_run"` ("does not fit in 6 GB GPU memory"), with no fit (`run.NOT_RUN`). It is never retried.
- **SHA-256 pin.** `run.py` checks the TabPFN checkpoint's SHA-256 against `TFMCAL_TABPFN_SHA256` before the first replicate.
- **Relative thresholds.** At pi = 3 % the probability-scale class thresholds (mean, moderate, strong) are multiplied by 0.3 (`metrics.probability_scale`). `score.py` also writes the classes at half and double the thresholds (`cls_*_half`, `cls_*_double`).

Notes on the arms:
- Binary and categorical columns are declared categorical to TabPFN (`categorical_features_indices`) and to TabICL (pandas `category`), as pre-registration 5.1 requires.
- Mitra's sklearn interface has no categorical argument. It gets the integer codes, which is how AutoGluon itself passes categoricals to Mitra (label encoding).
- Added after the 2026-10-07 review (see `../REVIEW_REPORT.md`):
  - every GPU arm caps its process at 80 % of the card (`TFMCAL_GPU_FRACTION`), so an overflow is a recorded CUDA out-of-memory failure instead of a silent spill into system RAM;
  - Mitra counts as failed when AutoGluon's out-of-memory fallback cut its context below the full development set (it does at p = 30, n = 3000);
  - Mitra's random feature-sign flips use the global numpy generator, which Mitra's own seed does not set; the wrapper now seeds it from the replicate (and restores it), so a rerun gives the same predictions;
  - `tabpfn_t1` uses the `tabpfn` seed, so the two differ only in the temperature;
  - a development set with no events (or no non-events) is recorded as the same failure for every arm;
  - `run.py` refuses to add `tabpfn` replicates fitted with a different checkpoint version from those already stored in the same folder.

### Steps the author takes himself (licences are never accepted from code)

1. **TabPFN-3.5.** Accept the licence at Prior Labs / Hugging Face (`Prior-Labs/tabpfn_3_5`) and log in. Then, in a terminal with `rebuild/env.sh` sourced:
   `python -c "from tabpfn.model_loading import download_model; from tabpfn.constants import ModelVersion; from pathlib import Path; import os; print(download_model(Path(os.environ['TABPFN_MODEL_CACHE_DIR'])/'tabpfn-v3.5-20260909.safetensors', version=ModelVersion.V3_5, which='classifier'))"`
   From then on, `tabpfn` resolves to V3_5 by itself, and the file's SHA-256 is recorded per replicate. Pre-registration 5.2: a 3.5 run after results under V2 have been seen goes into `DEVIATIONS.md` as an *added* arm.
2. Nothing else is gated. Mitra-v2 and TabICL are already in `rebuild/models/hf/hub` (the setup stream fetched them).

## Running

Use the project venv for the CPU arms, and `rebuild/.venv-gpu` (CUDA torch 2.13+cu126, tabpfn 9.1.0, tabicl 2.2.0) for the foundation models. Source `rebuild/env.sh` first, or rely on `models.configure_environment()`, which sets the same folders.

```
# classical arms, R = 500 (plan 4.7); X stored only where localization/DeepGOF need it (reps 1-200)
python run.py --scenarios all --reps 1-200   --models classical --n-workers 8 --threads 1
python run.py --scenarios all --reps 201-500 --models classical --n-workers 8 --threads 1 --store-x none
# foundation models on the GPU, R = 200, paired with the first 200 classical datasets
.venv-gpu/Scripts/python.exe code/run.py --scenarios all --reps 1-200 --models mitra,tabicl --device cuda
.venv-gpu/Scripts/python.exe code/run.py --scenarios S1_pi10_n200,S1_pi10_n500,... --reps 1-200 --models tabpfn_legacy --device cuda
# TabPFN on CPU workers in parallel with the GPU queue (plan 4.10), in its own sub-folder so that it
# never waits for the GPU process's locks; the R reader merges the files of one replicate
.venv-gpu/Scripts/python.exe code/run.py --scenarios all --reps 1-200 --models tabpfn --device cpu --n-workers 2 --threads 8 --tag tabpfn_cpu
python score.py --scenarios all --n-workers 8
```

Options:
- `--scenarios` accepts names, ids, id ranges (`1-12`), families (`S3`, `T3`), or `core`, `stress` and `all`.
- `--models` accepts arm names or `all`, `cpu-safe`, `classical` and `tfm`.
- `--rerun-failed` refits arms that are recorded as failed.
- `--smoke` switches to the 777 seed stream and writes to `results/smoke/`.
- `score.py --preds <dir> [--out <dir>]` scores another prediction folder (added for the 2026-10-07 pilot; the metrics go next to it in `metrics/` by default).
- Since 2026-10-07, `metrics.loess` fits tied points once (the isotonic ladder columns have a handful of distinct values); the numbers are bit-identical and scoring at N_eval = 3,334 is 3.4 times faster. See `../PILOT_REPORT.md` for the pilot itself (`pilot_*.py`, `pilot_run.sh`).

Resume and safety:
- A replicate whose requested arms are all recorded (ok or failed) is skipped.
- Writes are atomic (temporary file, then rename).
- A `.lock` file per replicate lets two processes share a scenario. A lock whose process is dead (checked with the Windows API) is broken at once; any other lock is broken after 6 hours.
- `--tag` writes to `<scenario>/<tag>/<rep>.parquet`, and the arms already stored in any file of that replicate are not refitted.
- Every run writes `logs/run_<time>.jsonl` (one line per replicate, with the timing per arm), appends failures to `logs/failures.jsonl`, and writes `results/environment_engine.json`.

## Smoke test (2026-10-06, CPU only)

The GPU was busy with the setup stream's timing benchmark, so no GPU work was started.

Runs:
- `run.py --smoke --scenarios S1_pi10_n200,S3_pi03_n200 --reps 1-2 --models cpu-safe` wrote 4 files. A second identical call skipped all 4.
- `--models tfm --reps 1` merged TabPFN (V2) and the legacy arm into the existing files after the fingerprint assertion. In the project venv, where tabicl and the Mitra weights are absent, Mitra and TabICL were recorded as failed, as designed.
- `tests/check_resume.py`:
  - a live lock was respected and a dead owner's lock was broken;
  - a run hard-killed after its first replicate resumed with that replicate skipped and the others written, with no lock or temp files left behind;
  - a deleted replicate was regenerated bit-identically (data, all predictions, Y*, ladder).
- T1, T2 and T3 (n = 500) ran with all CPU arms.
- `score.py --smoke` scored every column.

Timing on a loaded CPU (other R jobs running at 77-90 %):

| Arm | Seconds per replicate |
|---|---|
| R arms | about 3-5 per call, of which 2.5 is loading the logistf namespace (recorded separately). The fits take 0.03 s (FLIC), 0.3 s (ridge) and 0.3-8 s (GAM; 8 s at p = 30). |
| LightGBM | 0.3-2.8 |
| TabPFN V2 on CPU (`n_estimators` "auto" = 8, softmax temperature 0.9) | 30-65 |

Disk use:
- 150 KB per replicate at N_eval = 1,000 with the classical arms;
- 340 KB at 3,334;
- 420 KB with two TFMs and their ladders.

The full design comes to about 6-8 GB with X stored on reps 1-200 only (C: had 30 GB free on 2026-10-06).

## Points for the author or the pre-registration (decisions, not bugs)

1. **Loess floor in the pre-registered true ICI.**
   - A loess (span 0.75, degree 2) of p* on logit(p-hat) cannot follow the logistic curve. A *perfect* model therefore gets a true ICI of 0.0009-0.0043 and an E90 of up to 0.015 (S4 at 30 %), instead of 0. That is 6-30 % of the "moderate" threshold of 0.015.
   - `metrics.oracle_metrics` reports the pre-registered value (`true_ici`) and a version that smooths logit p* on logit p-hat (`true_ll_ici`). The second is exactly 0 for the oracle, and it equals the OCE whenever logit p* is linear in logit p-hat (unit tests).
   - Options: keep the pre-registered estimator and report the oracle row as its floor; or pre-register `true_ll_ici` as primary. Decide before the freeze.
2. **Absolute class thresholds at pi = 3 %.**
   - E90 <= 0.03 and ICI <= 0.015 are easy to meet when the mean risk is 0.03. In the smoke, arms with b* = 0.47 met "strong" while failing "weak".
   - `cls_highest` is therefore nested (Van Calster's hierarchy: a class counts only when every lower class holds). The four rules are also stored one by one.
   - Consider thresholds relative to pi, and state the nesting in pre-registration 4.5.
3. **R `loess` surface.**
   - R's default `surface = "interpolate"` differs from the exact (`"direct"`) fit by up to 0.02 at single points and about 0.001 in the ICI (checked on 777 points).
   - `metrics.loess` reproduces `"direct"` to 1e-14. Above 1,200 rows, it fits exactly at 400 anchors joined by a spline, which changes the ICI by less than 1e-6.
   - The R layer should use `surface = "direct"` for any ICI it computes, or the two layers will disagree by about 0.001.
4. **m = 100 at pi = 3 %.** The local sample holds about 3 events, and P(0 events) is about 5 %. The R1-R3 fits then fail and are recorded as failed (it happened in smoke S3_pi03 rep 1). Report the failure counts, or draw m = 100 with at least one event, decided before the run.
5. **LightGBM** runs in Python, not R. R `lightgbm` is not in `rlib`; it is the same library, with the pre-registered grid and the same early-stopping rule. Note this in `DEVIATIONS.md` or the pre-registration text.
6. **Prediction storage.**
   - p-hat is float32 (pre-registration 3.8) and p* is float64.
   - Predictions within 1e-6 of 0 or 1 are counted per arm (`n_boundary`) before the cast. All computations clip at 1e-6, so the cast changes no estimand.
7. **Oracle size check DGP.** `R/size_check.R` uses its own coefficients (a different BETA and shape terms). Its header says this, and under the oracle it does not matter. However, the pre-registration's Appendix A should list only `scenarios.csv`.
8. **TabPFN `n_estimators = "auto"`** resolved to 8, and `softmax_temperature` to 0.9 for V2 and 1.0 for 3.5. Both are recorded per replicate. The temperature arms run on the 8 scenarios of `main_groups.py temperature`.

Points 1, 2, 4 and 5 above were settled in the pre-registration (Sections 4.1, 4.5, 8.2 and 5.2).
