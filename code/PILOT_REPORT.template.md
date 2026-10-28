# Phase-1 pilot report (2026-10-07)

**Scope.** REBUILD_PLAN section 9, phase P1: every model arm on the GPU environment, 6 scenarios x 20 replicates, then the GOF layer, with times, GPU memory and failure rates measured. The full run was **not** started. The pilot used the smoke seed stream (`SeedSequence([..., 777])`), so no pilot replicate belongs to the main run. The old draft and old code were not touched.

**Verdict.** The engine passes the pilot. All 120 replicate files were written and scored, and the GOF layer finished on all of them with 0 open errors and 20 unique data fingerprints per scenario. Predictions are bit-identical across processes and under CPU load (TabICL differs by at most 8e-7). Before the main run, six decisions are needed (section 9). Three are new findings from the pilot:
- **Temperature.** TabPFN V2 and TabICLv2 ship with softmax temperature 0.9. This scales every logit by 1/0.9 (measured slope 1.1105-1.1109). As a result, the mean predicted risk is 0.012-0.017 below the true mean risk at pi = 10-30 % (traced in 6.1). TabPFN-3.5 ships with temperature 1.0.
- **TabPFN-3.5 memory.** TabPFN-3.5 runs out of GPU memory at T2, n = 3000 (as Mitra does). On the CPU it fits that cell in 134 s.
- **LightGBM + Platt.** At n = 200, pi = 3 %, the Platt step can fit a negative slope, which inverts the ranking (3 of 20 replicates).

**Compute.** With the recommended settings, the full design needs about **118 GPU-hours** and **490 CPU core-hours**. Moving about 45 GPU-hours of TabPFN-3.5 work to two CPU workers brings the wall time to about **3 days on 20 cores** (2.9-3.5 days; section 8).

**TabPFN-3.5 arrived during the pilot.** At 08:08-08:15 the checkpoints `tabpfn-v3.5-20260909.safetensors` (SHA-256 `ece4d67e...0be3`) and `tabpfn-v3.5-fast-20260909.safetensors` appeared in `rebuild/models/tabpfn`. This session did not download them. The main pilot (steps 1-7) had already finished with the V2 checkpoint as the `tabpfn` arm, so I ran TabPFN-3.5 as an extra arm: 5 scenarios x 20 replicates on the GPU, plus timing cells. It is reported beside the others as "TabPFN-3.5".

---

## 1. What ran

| Step | What | Where | Wall time |
|---|---|---|---|
| 1 | classical arms (oracle, LR-MLE, FLIC, ridge-CV, GAM, LightGBM raw + Platt) | project venv, 8 workers | 3.0 min |
| 2 | TabPFN (V2 checkpoint, `n_estimators` 8, T 0.9), TabICLv2, Mitra-v2 | `.venv-gpu`, one GPU process, 6 threads | 85 min |
| 3 | side arms `tabpfn_legacy` (V2, own seed) and `tabpfn_t1` (V2 at T 1.0) on the 4 cheaper scenarios | GPU | 24.5 min |
| 4 | the same fits by two GPU processes at once (throughput, cross-process reproducibility) | GPU | 5.1 min |
| 5 | GOF layer: `run.all.external` + `edge.gof` on real Y and Y*, localization, DeepGOF-1 (B = 199) on FLIC + TFMs | R, 7 workers | 81 min |
| 6 | timing grid: TFM fits in 8 more n x event-rate cells, 3 reps each, under the GOF load | GPU | 14 min |
| 7 | `score.py` (oracle and empirical estimands) and `gof_layer.R check` | 6 workers | 7.2 min |
| extra | T2 n = 500 and T3 n = 3000 timing; TabPFN speed options; Mitra on CPU | GPU / CPU | 35 min |
| extra | TabPFN-3.5: 5 scenarios x 20 reps (GPU), 2 timing cells, T2 n = 3000 (GPU fails; CPU works), score + GOF | GPU / CPU | 75 min |

The scenarios were chosen to cover:
- every shape;
- every event rate (3, 10 and 30 %);
- every n (200, 500, 1000 and 3000);
- the rarest-and-smallest cell;
- two stress families.

| Scenario | Why |
|---|---|
| S1_pi03_n200 | rarest event rate and smallest n (about 6 development events), N_eval = 3,334 |
| S2_pi10_n500 | misspecified smooth non-linear shape (GAM correct, LR wrong) |
| S3_pi30_n1000 | interactions (the localization target), common event |
| S4_pi03_n3000 | cloglog link at 3 %, the most expensive p = 8 cell (4,334 predicted rows) |
| T2_pi10_n3000 | p = 30, the GPU-memory and DeepGOF stress cell |
| T3_pi10_n500 | 6-level categorical |

Commands: `code/pilot_run.sh` (steps 1-7), `code/pilot_extra.sh`, and the TabPFN-3.5 runs listed in section 10. Logs are `logs/pilot_*.log`. GPU utilisation and CPU load were sampled every 15 s (`logs/pilot_gpu_util.csv`, `logs/pilot_cpu_load.csv`).

---

## 2. Fit times, GPU memory and failures

Fits with any failure or degenerate output (all other arm x scenario pairs had 20/20 ok fits and no degenerate output):

{{report_fits.md}}

Median seconds per fit (GPU, one process; predicted rows = N_eval + 500 recalibration + 500 probe):

| Cell (n, predicted rows) | TabPFN V2 | TabPFN-3.5 | TabICLv2 | Mitra-v2 | peak GPU MB (V2 / 3.5 / TabICL / Mitra) |
|---|---|---|---|---|---|
| n 200, 4,334 (S1_pi03) | 11.6 | 25.4 | 2.1 | 4.5 | 217 / 2,899 / 1,193 / 704 |
| n 200, 2,000 (S1_pi10, grid) | 6.0 | - | 1.6 | 3.4 | 124 / - / 648 / 704 |
| n 500, 2,000 (S2_pi10) | 7.4 | 15.5 | 1.5 | 3.4 | 135 / 2,139 / 724 / 743 |
| n 500, 4,334 (S1_pi03, grid) | 13.9 | - | 2.4 | 6.3 | 229 / - / 1,274 / 743 |
| n 1000, 2,000 (S3_pi30) | 10.5 | 20.4 | 1.8 | 4.8 | 156 / 2,404 / 851 / 952 |
| n 1000, 4,334 (S1_pi03, grid) | 18.2 | 34.7 | 3.0 | 9.4 | 249 / 2,917 / 1,405 / 952 |
| n 3000, 2,000 (S1_pi10/pi30, grid) | 27.9 | 44.0 | 10.1 | 13.0 | 235 / 2,799 / 870 / 3,687 |
| n 3000, 4,334 (S4_pi03) | 41.2 | 61.4 | 13.7 | 24.2 | 327 / 2,760 / 1,196 / 3,684 |
| T2 n 500 (p = 30) | 23.9 | 17.7 | 2.5 | 9.3 | - / 3,845 / - / - |
| T2 n 3000 (p = 30) | 94.0 | **OOM** (CPU: 134) | 13.6 | **fails** | 782 / - / 2,237 / - |
| T3 n 500 (categorical) | 7.3 | 15.5 | 1.5 | 3.4 | 135 / 2,139 / 724 / 743 |
| T3 n 3000 (grid) | 27.9 | - | 10.0 | 12.8 | |

Notes:
- **Repeatable to the tenth of a second.** TFM times are stable over replicates (p90 within 2 % of the median). The first fit of each process carries a warm-up of 5-9 s.
- **The GPU is the bottleneck.** GPU utilisation averaged 94 % in step 2 and 96 % in step 3, against 25 % at idle (the display).
- **Two GPU processes at once give no gain.** Each replicate took 2.2-2.4 times as long, so throughput was 8-15 % lower than one process. The predictions were bit-identical to the solo run.
- **The GOF load (7 R workers) did not slow the GPU fits.** S2_pi10_n500 TabPFN took 7.5 s against 7.4 s, and S4_pi03_n3000 took 41.5 s against 41.2 s. The predictions were identical.
- **No TabPFN option makes it faster.** I tried `memory_saving_mode=False`, `n_preprocessing_jobs=6` and both together; times stayed between 0.94 and 1.05 times the default, with no consistent gain, and the predictions were identical. `fit_mode="fit_with_cache"` (0.92-1.04 times) changed the predictions by up to 9.7e-4 (`results/pilot/tables/tabpfn_speed.csv`).
- **Classical arms** took 3.7-8.1 s per replicate, and 38 s at T2 (GAM 25 s). Most of it is Rscript start-up.

---

## 3. Oracle estimands (20 replicates; mean (MCSE); b* as median (bootstrap MCSE))

"weak" counts the replicates with |a*| <= 0.1 and |b* - 1| <= 0.1. "mean p - mean p*" is the calibration-in-the-large error on the probability scale.

{{report_oracle.md}}

How to read this table (pilot level only, 20 replicates on the smoke stream; these are not hypothesis tests):
- **S1_pi03_n200.**
  - Median b* runs from 0.14 (GAM) to 1.00 (TabPFN-3.5); every classical arm is over-confident (b* <= 0.88). FLIC's 0.49 (0.05) is below the TFMs' 0.71-1.00 (MCSE 0.08-0.13).
  - The paired true-ICI difference from FLIC is -0.0061 to -0.0094 for TabPFN V2, TabICL and Mitra (MCSE 0.0012-0.0020; `paired_vs_flic.md`).
  - The mean b* is not usable here: ridge-CV's mean b* is 19.7 (MCSE 18.1). It is driven by one replicate with an almost flat prediction (b* = 346).
- **S2_pi10_n500, S3_pi30_n1000 and T3_pi10_n500.** Every TFM's paired true-ICI difference from FLIC is within 2 MCSE of 0, except TabPFN V2 at S3 (+0.0068 (0.0017)), and that gap disappears at T = 1.0 (+0.0002 (0.0010); 6.1). Among the classical arms, GAM is worse than FLIC at T3 (+0.0062 (0.0008)).
- **S4_pi03_n3000 and T2_pi10_n3000.**
  - TabPFN V2 and TabICL have a* = 0.18-0.32 (MCSE 0.02-0.03) and true ICI 2-3 times FLIC's. EDGE, Cox and HL reject them in 6-16 of 20 replicates.
  - At S4, Mitra-v2 and TabPFN-3.5 have a* = 0.00-0.02 and true ICI 0.0036-0.0038 (FLIC 0.0033). Mitra fails at T2 n = 3000, and TabPFN-3.5 was not run there on the CPU beyond the timing replicate.
- **The pilot-informed H1 is not seen in these cells.** That hypothesis predicts median b* > 1.1 for every TFM at n <= 500, pi = 10 %. The two pilot cells that qualify (S2_pi10_n500 and T3_pi10_n500) give TFM medians of 0.88-1.11 (MCSE 0.05-0.09). See decision 9.4.
- **The oracle's true ICI is not 0.** It is 0.0008-0.0029, the known loess floor of the pre-registered estimator (review D6). `true_ll_ici` is exactly 0 for the oracle (`oracle.md`).

The full tables (true E90, `true_ll_ici`, empirical ICI, mean b*) are in `results/pilot/tables/oracle.md`. Paired differences from FLIC are in `paired_vs_flic.md`, and the replicates needed per MCSE target in `mcse_requirements.md`.

### Recalibration ladder (true ICI, mean (MCSE); [kF] = k failed fits)

{{ladder.md}}

- **m = 100 makes calibration worse.** It raises the true ICI above R0 for every arm and level, except FLIC's R1 and R2 at S1_pi03_n200, because the noise of a 100-row recalibration fit outweighs the bias it removes.
- **m = 500 helps only arms that start above its own noise floor.** R2 lowers the true ICI at S1, S2 and T3, and for TabPFN V2 and TabICL at T2. It raises it for every arm at S3 (to about 0.022) and for FLIC and Mitra at S4 and T2. A 500-row logistic recalibration has a noise floor of about 0.008 (S4) to 0.022 (S3).
- **The test-guided rule matters.** These results are what make it relevant to plan H5: recalibrating a well-calibrated model with 100-500 rows does harm.
- **The R1-R3 failures at m = 100 and pi = 3 %** are subsamples with no events (1-2 of 20). They are recorded, as designed.

---

## 4. GOF layer

### Size on each arm's own predictions (exact null Y* ~ Bern(p-hat); rejection rate at 0.05 (MCSE), pooled over the arms of each scenario)

{{gof_size_by_scenario.md}}

- **Every test is within 2 MCSE of 0.05 when pooled.** The pooled MCSE is 0.006. Le Cessie is the highest at 0.059 and HL the lowest at 0.043.
- **Per scenario, the MCSE is 0.007-0.027**, so a scenario-level rate of 0.094 (le Cessie, T2) is 2 MCSE from 0.05 and is not evidence of liberality on its own.
- **Stukel is 0.052 pooled here.** The review's size check (2,000 replicates) found 0.063 at N = 3,334. This pilot is too small to confirm or contradict that.
- **The oracle's localization row is low:** 1 of 120 on Y* and 3 of 120 on real Y. The main run's 1,000 oracle nulls per scenario will settle it.
- **DeepGOF-1 has no Y* rows by design** (`--deepgof-null` is off).

"NR" counts the tests that did not run. All of them are in `gof_not_run.md`:
- **EDGE (G = 10) and the localization step** did not run on 7 of the GAM fits at S1_pi03_n200, where 3,268-3,319 of 3,334 predictions sit at 0 or 1.
- **GiViTI failed on 4 of those GAM fits.**
- **EDGE on the isotonic R3_m100 ladder columns** reported "too few distinct group probabilities" (expected; review D8).

### Rejections on the real outcomes (count of 20)

{{report_gof_real.md}}

- **The oracle row is the main run's built-in size check.** Here it rejects in 0-3 of 20 replicates per test and scenario.
- **The detection pattern follows the oracle estimands of section 3.** At S1_pi03_n200, most tests reject every arm in 12-20 of 20 replicates. The temperature-driven a* > 0 of TabPFN V2 and TabICL at S4 and T2 is detected in 6-16 of 20. At S3_pi30_n1000, where the arms are close to calibrated, the rates are low.

### GOF cost (core-seconds per replicate, all arms, B = 199, 7 workers on a machine running the GPU process)

{{gof_timing_per_rep.md}}

Share of that time by call type (%):

{{gof_timing_share.md}}

- **Per call:**
  - battery: 0.37-0.45 s at N = 1,000, 1.2-1.4 s at N = 3,334;
  - localization: 0.17-1.1 s;
  - DeepGOF-1: 19-20 s at p = 8, 60 s on T3 (13 dummy-coded columns), 229 s median on T2 (435 column pairs).
- **With 12 workers** (the TabPFN-3.5 GOF run), DeepGOF-1 took 20-21 s at p = 8 and 34 s on T3, so 12 workers do not slow each call.
- **DeepGOF-1 is 52-97 % of the GOF cost.** It is the only part worth cutting (section 8).

---

## 5. Reproducibility checks

| Check | Result |
|---|---|
| two GPU processes vs one (16 TFM fits) | TabPFN and Mitra max \|diff\| 0; TabICL 8.3e-7 |
| GPU fits under the 7-worker GOF load vs idle (6 fits) | TabPFN and Mitra 0; TabICL 8.0e-7 |
| `gof_layer.R check` | 20 replicate files, 20 unique fingerprints, 0 open errors in each of the 6 + 5 scenarios |
| `score.py` fingerprint assertion (regenerates every replicate) | passed on all 220 files |
| TabPFN-3.5 CPU vs GPU (S2_pi10_n500 rep 1) | max \|diff\| 2.6e-4, mean 2.9e-5 |
| Mitra CPU (fp32) vs GPU (bf16) (S2_pi10_n500 rep 1, 100 rows) | max \|diff\| 0.028, mean 0.0022 |
| unit tests after the pilot's code changes | 30/30 pass |

---

## 6. Bugs and surprising values, traced to their cause

### 6.1 The softmax temperature of TabPFN V2 and TabICL shifts the mean risk down (a model default, not a code bug)

- **What we saw.** TabPFN V2 and TabICLv2 have a* = +0.07 to +0.32 (MCSE 0.02-0.05) at S2, S3, S4, T2 and T3, and their mean predicted risk is below the mean true risk. Mitra-v2, `tabpfn_t1` and TabPFN-3.5 do not show this.
- **Cause.** Both models default to `softmax_temperature = 0.9`, which multiplies each logit by 1/0.9. Regressing logit p-hat(T 0.9) on logit p-hat(T 1.0) over the same replicates gives a slope of 1.1105-1.1109 (SD <= 0.0033 over replicates) in all four scenarios. Most patients have a negative logit, so stretching it lowers the mean risk: mean p-hat - mean p* is -0.0118, -0.0156 and -0.0130 at T 0.9 against +0.0029, -0.0033 and +0.0021 at T 1.0 (S2, S3, T3).
- **Consequences.**
  - b* falls by about 10 % (V2 medians 0.91-0.97 against 1.04-1.07 at T 1.0).
  - The arm drops out of the "weak" class (S3: 9/20 at T 0.9 against 15/20 at T 1.0).
  - The tests detect it (T3 Cox: 11/20 against 7/20).
- **For the main run.**
  - TabPFN-3.5 defaults to T = 1.0 (recorded per replicate), so the main `tabpfn` arm is not affected. But **`tabpfn_t1` is now identical to `tabpfn`** (same checkpoint, seed and temperature): about 13 GPU-hours at R = 200 for nothing.
  - TabICLv2 keeps T = 0.9. A TabICL T = 1.0 arm (review D10) is not tested; that the same mechanism applies is an inference.
  - The legacy V2 arm runs at T = 0.9.
  - The paper must state each model's default temperature, because it changes the calibration findings.

### 6.2 LightGBM + Platt can invert the ranking at tiny n (code behaviour; fix before the freeze)

- **What we saw.** At S1_pi03_n200 the Platt arm's mean AUC (against p*) is 0.644, below the raw arm's 0.672. Platt is meant to be monotone.
- **Cause.** In replicates 4, 5 and 7 (5-7 development events), early stopping chose 1-2 boosting rounds. The out-of-fold scores were then almost noise, and the logistic Platt fit returned slopes of -18.3, -3.4 and -1.6. Those replicates have expected AUC 0.477, 0.328 and 0.415 against 0.523, 0.672 and 0.585 for the raw model.
- **Fix (author decision, pre-registration text).** If the Platt slope is <= 0, fall back to an intercept-only recalibration (or to the raw model), and record the event as a flag. `models.fit_lgbm` was **not** changed.

### 6.3 Ridge-CV picks the lower edge of glmnet's lambda path

- **What we saw.** The `lambda_at_edge` flag is set in 19/20 replicates at S3_pi30_n1000, 10/20 at S4_pi03_n3000 and 6/20 at T2. lambda.min is 0.016-0.019 at S3 and 0.0026-0.0032 at S4, which is the small-lambda end.
- **Cause.** glmnet stops the default path early, when the deviance stops changing, so cross-validation wants less shrinkage than the path offers. Ridge's b* of 1.04-1.08 at n >= 1000 (LR: 0.92-0.97) is partly this truncation.
- **Fix (author decision).** Pass an explicit lambda sequence that reaches lower, for example `lambda.min.ratio = 1e-6` or `glmnet.control(fdev = 0)`, and keep the flag. Not changed.

### 6.4 Degenerate classical fits are stored as "ok"

At S1_pi03_n200, GAM puts more than 100 of 4,334 predictions within 1e-6 of 0 or 1 in 12/20 replicates, and LR-MLE does so in 8/20 (separation flag set in 4). The engine keeps these as `status = ok`, which is right, since they are the arm's real output. The plan's degenerate-fit counts (M2), however, need them labelled. I recommend that `score.py` add a `degenerate` column (`n_boundary` > 1 % of rows) and that every table counts it. Not changed.

### 6.5 Mitra-v2 and TabPFN-3.5 cannot fit T2 at n = 3000 on 6 GB

**Mitra-v2.** Its context was cut to 1,024 rows in 20/20 replicates, and each fit is recorded as failed (review F2, confirmed). On the CPU it is not feasible:
- S2 n = 500 took 88 s for 100 predicted rows;
- T2 n = 3000 had not finished 100 rows after 15 min, and I stopped it;
- the CPU predictions (fp32) differ from the GPU ones (bf16) by up to 0.028.

**TabPFN-3.5.** It runs out of memory at the 0.8 cap and at 0.92 with `expandable_segments`.
- Batching the test rows does not help, because the training set is the limit: 500- and 250-row batches also fail.
- Batching is also 4.5 times slower on S4 and changes the predictions by up to 5.8e-4.
- **The CPU works:** 134 s per fit with 8 threads. CPU and GPU predictions agree to 2.6e-4.

**Recommendation.** Run TabPFN-3.5 at T2_pi10_n3000 on a CPU worker (device recorded per replicate; one declared deviation). Report Mitra there as "does not fit in 6 GB" (review D2, option 1).

### 6.6 A checkpoint that appears mid-run breaks the `tabpfn` arm

At 08:10 a CPU timing call picked up the half-downloaded `tabpfn-v3.5-20260909.safetensors` and failed with `SafetensorError` (3 failures in `logs/failures.jsonl`, files in `results/pilot_cpu/`; harmless). `resolve_tabpfn` takes the newest file present. The review's F6 guard stops version mixing inside one output folder, but not a broken file. For the main run:
- set `TFMCAL_TABPFN_VERSION=V3_5` explicitly;
- check the SHA-256 (`ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`) before starting;
- never put files into `models/tabpfn` while a run is going.

### 6.7 Scoring was slow on isotonic columns (fixed, outputs identical)

- **What we saw.** `score.py` took 44.5 s per file at N_eval = 3,334, against 12 s at 1,000.
- **Cause.** `metrics.loess` fits exactly at every point when a column has at most 400 distinct values. The isotonic R3 ladder columns have a handful of distinct values among 3,334 rows, so the same fit was computed thousands of times.
- **Fix.** Fit once per distinct value and map back.
  - Output is bit-identical: max |diff| 0 on every estimand of 4 files, and 30/30 unit tests pass.
  - The file now takes 13 s, a 3.4-times speed-up.
  - This is the only change to the engine's numerical code in this session.

### 6.8 Values that look odd but are expected

- **The `tabpfn` and `tabpfn_legacy` arms are the same model until TabPFN-3.5 is pinned** (both V2). They differ only in the seed, and the seed-to-seed spread of the true ICI is at most 0.0008.
- **The oracle's empirical ICI on Y is 0.004-0.019, while its true ICI is 0.001-0.003.** That gap is the Bernoulli noise the plan's oracle estimands are designed to remove.
- **The recalibration-ladder worsening at m = 100** is described in section 3.

---

## 7. Changes made to the engine in this session

- `code/metrics.py`: fit tied points once in `loess` (6.7).
- `code/score.py`: new `--preds` and `--out` options (the defaults are unchanged).
- New pilot scripts (no effect on the engine):
  - `code/pilot_run.sh` and `code/pilot_extra.sh`;
  - `code/pilot_speed.py` and `code/pilot_mitra_cpu.py`;
  - `code/pilot_analyse.py`, `code/pilot_budget.py` and `code/pilot_report_tables.py`;
  - `code/pilot_build_report.py` and `code/PILOT_REPORT.template.md`.
- `run.py`'s code hash includes `metrics.py`, so the pilot files written after 05:20 carry a different code hash from those of step 2. The predictions are unaffected, because `metrics.py` is used only in scoring and in the ladder's logistic fit, which did not change.

No arm, seed, scenario or GOF setting was changed.

---

## 8. Final compute budget and run schedule

### 8.1 Basis

- **GPU:** the medians of section 2. TabPFN-3.5 was measured in 9 cells. The other cells use the V2 time multiplied by the 3.5/V2 ratio measured at the same n: 2.18 at n = 200, 2.11 at 500, 1.92 at 1000 and 1.54 at 3000.
- **T2 n = 3000:** TabPFN-3.5 runs on the CPU (134 s with 8 threads); Mitra is dropped.
- **GOF:** the measured core-seconds per replicate, split into oracle, classical, FLIC ladder, TFM, TFM ladder, side arms and DeepGOF, then multiplied by 1.25 for contention. With 12 workers the per-call time did not grow, so this is a margin.
- **Classical fits:** 3.7-8.1 s per replicate (38 s at T2).
- **Scoring:** 12.5-13.5 s per full file and 5-5.5 s per classical-only file.
- **CPU TabPFN-3.5:** f = CPU time / GPU time with 8 threads = 2.4, measured on S2_pi10_n500 on a quiet machine. f = 3.0 is used as the loaded case.

Per-scenario inputs: `results/pilot/tables/budget_per_scenario.txt`. Everything is computed by `code/pilot_budget.py`.

### 8.2 Plans (GPU-hours and CPU core-hours before balancing)

{{budget.md}}

### 8.3 After moving whole TabPFN-3.5 scenarios to CPU workers until GPU and CPU finish together

{{budget_balanced.md}}

The plan as written (P1) needs 4.0-4.7 days even with balancing. Its largest avoidable costs are:
- the `tabpfn_t1` arm, which is identical to `tabpfn` under 3.5;
- DeepGOF-1 at B = 199;
- the GOF battery on 300 extra classical replicates, which no pre-registered estimand needs. Their oracle metrics come from `score.py`, not from the GOF layer.

### 8.4 Recommended schedule (P4, about 3 days on 20 cores)

**Settings.**
- TFM arms: R = 200 at n <= 1000 and R = 150 at n = 3000.
- Classical arms: R = 500.
- Legacy TabPFN V2: R = 100 on the 16 core pi = 10 % scenarios.
- No `tabpfn_t1`.
- DeepGOF-1 at B = 99 on FLIC and the TFMs (real Y, reps 1-R).
- GOF battery: every arm on reps 1-R; the oracle only (real Y and Y*) on reps R+1 to 500.

**Order of work.**

| Stream | Cores | Work | Hours |
|---|---|---|---|
| GPU process (`--device cuda --threads 6`) | GPU + 1 | TabICLv2 and Mitra-v2 on all scenarios; TabPFN-3.5 on core pi = 3 % (30.1 GPU-h) and T2 n = 500; then legacy V2 | about 70 |
| CPU TabPFN-3.5, worker A (`--device cpu --threads 6 --tag tabpfn_cpu`) | 6 | T2_pi10_n3000 first (about 6 h), then core pi = 30 % (18.2 GPU-h equivalent) | about 70 |
| CPU TabPFN-3.5, worker B (same) | 6 | core pi = 10 % (18.2 GPU-h equivalent), then T1/T3/T4 (10.7) if ahead of the GPU | about 70 |
| R / Python | 6 | classical arms first (52 core-h, about 9 h), then the GOF layer and `score.py` streaming every 2-3 h | about 70 |
| free | 5 | the author's own jobs | |

**Assigning scenarios.**
- Scenarios are assigned whole, so each scenario is fitted on one device. The CPU-fitted scenarios are listed in `DEVIATIONS.md` (CPU and GPU predictions agree to 2.6e-4).
- Check the balance after 24 h:
  - if a CPU worker is behind, hand its unstarted scenarios back to the GPU;
  - if it is ahead, give it T1, T3 and T4.

**What makes it longer.**
- Without the CPU workers, the GPU alone needs 118 h (4.9 days).
- With 6 threads per CPU worker, f is likely nearer 3.0 than 2.4 (only 8 threads were timed): 3.1 days.
- With 16 rather than 20 cores, it takes 3.3-3.5 days.
- Keeping R = 200 at n = 3000 (P3) gives 3.3-3.6 days.

**Disk.**
- The pilot files were 0.28-0.68 MB per full prediction file and 0.24 MB per GOF CSV.
- The design needs about 10 GB; C: has 25 GB free.
- gzip the GOF CSVs (`gof_layer.R` reads `.csv.gz`) if space gets tight.

### 8.5 Replicate counts against the plan's MCSE targets (section 4.7), from the pilot SDs

| Quantity | Pilot SD (worst cell) | R recommended | MCSE achieved | Plan target |
|---|---|---|---|---|
| mean true ICI, TFM arms | 0.0104 (TabPFN T 1.0, S2) | 200 (150 at n = 3000) | <= 0.0007 (0.0009) | <= 0.002 |
| mean true ICI, classical arms | 0.0236 (GAM, S1_pi03_n200); FLIC 0.0106 | 500 | GAM 0.0011, FLIC <= 0.0005 | <= 0.001 (GAM at n = 200, pi = 3 % misses by 0.0001) |
| paired true-ICI difference vs FLIC | 0.0116 (TabPFN V2, T3) | 200 / 150 | <= 0.0008 / 0.0009 | report it |
| rejection rate at 0.05 | binomial | 200 / 150 | 0.0154 / 0.0178 | 0.015 |
| rejection rate at 0.5 | binomial | 200 / 150 | 0.035 / 0.041 | 0.035 |
| proportion "weak" near 0.8 (minimum-n rule) | binomial | 200 / 150 | 0.028 / 0.033 | (none set) |
| median b*, TFMs at S1_pi03_n200 | bootstrap MCSE 0.08-0.13 at R = 20 | 200 | about 0.03 | resolves the 1.1 threshold of H1 |
| size on the main DGP (oracle, real Y + Y*) | binomial | 500 x 2 = 1,000 nulls per scenario | 0.0069 per scenario, 0.0009 pooled | 0.005 (dedicated study, already done with 2,000) |

- The true-ICI targets would be met with R = 100 for the TFMs (worst MCSE 0.0010). The binding targets are the rejection rates, which need R of about 200.
- At n = 3000, R = 150 misses the plan's rejection-rate targets by about 15 %. State this in the pre-registration, or keep R = 200 there and accept 3.3-3.6 days.

---

## 9. Decisions needed before the main run (not taken here)

1. **Plan option for the schedule:** P4 (recommended, about 3 days), P3 (R = 200 everywhere, 3.3-3.9 days) or P5 (lean, 2.5-3.0 days). Within P4, say whether the CPU TabPFN-3.5 workers (two scenario groups on the CPU, declared) are acceptable.
2. **T2_pi10_n3000.** TabPFN-3.5 on the CPU (recommended); Mitra "does not fit in 6 GB".
3. **Temperature arm.** Drop `tabpfn_t1`, or redefine it as TabPFN-3.5 at T = 0.9 (8 scenarios, R = 100, about 6 GPU-h), which would show the temperature effect of 6.1 on the current model. Optionally add a TabICL T = 1.0 arm (review D10; about 2 GPU-h at R = 100 on the same 8 scenarios).
4. **Hypotheses.** H1 as written (b* > 1.1 at n <= 500) is not seen in the two qualifying pilot cells (TFM medians 0.88-1.11). The plan says "pilot-informed". The new pilot used the smoke stream, which the main run never touches, so it may inform the pre-registration if it is declared there. Keep H1 and report it as it falls, or restate it before the freeze. The same applies to H3, whose sign at pi = 3 % depends on the model's temperature.
5. **LightGBM Platt slope <= 0** (6.2) and **the ridge lambda path** (6.3): fix both before the freeze, or declare them.
6. **Degenerate-fit labelling** (6.4), plus the still-open estimand questions of the review (D4 Stukel, D6 loess floor and thresholds at 3 %).

Then:
1. freeze and deposit the pre-registration;
2. pin `TFMCAL_TABPFN_VERSION=V3_5` and record the SHA-256;
3. start.

---

## 10. Files

- **Report and tables:**
  - `rebuild/PILOT_REPORT.md` (this file, built by `code/pilot_build_report.py` from `code/PILOT_REPORT.template.md`);
  - `results/pilot/tables/`: `oracle.md`, `paired_vs_flic.md`, `mcse_requirements.md`, `ladder.md`, `gof_*.md`, `fit_summary.md`, `fits.csv`, `metrics_all.csv`, `budget*.{md,json,csv,txt}`, `tabpfn_speed.csv`, `repro_*.md`, `util_by_step.md`.
- **Pilot data (smoke stream, all safe to delete after the paper):**
  - `results/pilot/{preds,probe,metrics,gof}`;
  - `results/pilot_timing`, `results/pilot_concurrency`;
  - `results/pilot_tabpfn35/{preds,metrics,gof}`, `results/pilot_tabpfn35_cpu`, `results/pilot_tabpfn35_cap92`, `results/pilot_cpu` (the failed partial-download run).
- **Logs:**
  - `logs/pilot_*.log`, `logs/pilot_gpu_util.csv`, `logs/pilot_cpu_load.csv`;
  - `logs/run_20261007_0440*`-`0832*_smoke.jsonl`;
  - `logs/gof_layer.log`;
  - pilot failures appended to `logs/failures.jsonl`.
- **TabPFN-3.5 commands** (from `rebuild/`, with `env.sh` sourced):
  - `run.py --smoke --out results/pilot_tabpfn35/preds --scenarios S1_pi03_n200,S2_pi10_n500,S3_pi30_n1000,S4_pi03_n3000,T3_pi10_n500 --reps 1-20 --models tabpfn --device cuda --threads 6`;
  - timing cells with `--reps 1-2`;
  - `--device cpu --threads 8` into `results/pilot_tabpfn35_cpu`;
  - `score.py --preds results/pilot_tabpfn35/preds`;
  - `gof_layer.R run --preds results/pilot_tabpfn35/preds --out results/pilot_tabpfn35/gof`.
