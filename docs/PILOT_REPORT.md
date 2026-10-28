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

| scenario | arm | fits | failed | degenerate (>100 preds at 0/1) | median s | max s | peak GPU MB |
|---|---|---|---|---|---|---|---|
| S1_pi03_n200 | LR-MLE | 20 | 0 | 8 | 0.0 | 0.0 | - |
| S1_pi03_n200 | GAM | 20 | 0 | 12 | 0.3 | 14.1 | - |
| T2_pi10_n3000 | Mitra-v2 | 20 | 20 | 0 | - | - | - |
| T2_pi10_n3000 | TabPFN-3.5 (T 1.0) | 2 | 2 | 0 | - | - | - |
| T3_pi10_n500 | LR-MLE | 20 | 0 | 1 | 0.0 | 0.0 | - |
| T3_pi10_n500 | GAM | 20 | 0 | 1 | 0.3 | 0.5 | - |

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

| scenario | arm | R ok/failed | true ICI | OCE | b* median | a* mean | mean p - mean p* | AUC (vs p*) | weak |
|---|---|---|---|---|---|---|---|---|---|
| S1_pi03_n200 | oracle | 20/0 | 0.0010 (0.0000) | 0.0000 (0.0000) | 1.00 (0.00) | 0.00 (0.00) | 0.0000 (0.0000) | 0.811 (0.001) | 20/20 |
| S1_pi03_n200 | LR-MLE | 20/0 | 0.0304 (0.0031) | 0.0357 (0.0027) | 0.31 (0.05) | -0.41 (0.17) | 0.0092 (0.0029) | 0.713 (0.018) | 0/20 |
| S1_pi03_n200 | FLIC | 20/0 | 0.0215 (0.0024) | 0.0287 (0.0019) | 0.49 (0.05) | -0.26 (0.10) | 0.0076 (0.0026) | 0.729 (0.014) | 0/20 |
| S1_pi03_n200 | ridge-CV | 20/0 | 0.0133 (0.0016) | 0.0221 (0.0012) | 0.84 (0.20) | -0.15 (0.08) | 0.0054 (0.0023) | 0.744 (0.015) | 0/20 |
| S1_pi03_n200 | GAM | 20/0 | 0.0453 (0.0053) | 0.0490 (0.0046) | 0.14 (0.06) | -3.74 (1.48) | 0.0135 (0.0035) | 0.643 (0.019) | 0/20 |
| S1_pi03_n200 | LightGBM + Platt | 20/0 | 0.0204 (0.0028) | 0.0312 (0.0023) | 0.55 (0.15) | -0.29 (0.11) | 0.0087 (0.0029) | 0.644 (0.027) | 0/20 |
| S1_pi03_n200 | LightGBM raw | 20/0 | 0.0155 (0.0022) | 0.0272 (0.0016) | 0.88 (0.19) | -0.13 (0.08) | 0.0049 (0.0024) | 0.672 (0.016) | 1/20 |
| S1_pi03_n200 | TabPFN V2 (T 0.9) | 20/0 | 0.0121 (0.0015) | 0.0218 (0.0012) | 0.84 (0.10) | -0.13 (0.06) | 0.0043 (0.0017) | 0.737 (0.015) | 0/20 |
| S1_pi03_n200 | TabPFN V2 legacy seed | 20/0 | 0.0127 (0.0015) | 0.0218 (0.0012) | 0.79 (0.12) | -0.13 (0.06) | 0.0043 (0.0017) | 0.738 (0.015) | 1/20 |
| S1_pi03_n200 | TabPFN V2, T 1.0 | 20/0 | 0.0167 (0.0016) | 0.0261 (0.0013) | 0.93 (0.11) | -0.41 (0.05) | 0.0140 (0.0019) | 0.737 (0.015) | 0/20 |
| S1_pi03_n200 | TabICLv2 (T 0.9) | 20/0 | 0.0141 (0.0018) | 0.0236 (0.0013) | 0.71 (0.08) | -0.23 (0.07) | 0.0071 (0.0021) | 0.737 (0.015) | 1/20 |
| S1_pi03_n200 | Mitra-v2 | 20/0 | 0.0153 (0.0016) | 0.0235 (0.0012) | 0.82 (0.13) | -0.29 (0.06) | 0.0097 (0.0021) | 0.755 (0.012) | 0/20 |
| S1_pi03_n200 | TabPFN-3.5 (T 1.0) | 20/0 | 0.0114 (0.0013) | 0.0217 (0.0010) | 1.00 (0.11) | -0.17 (0.06) | 0.0061 (0.0021) | 0.735 (0.013) | 3/20 |
| S2_pi10_n500 | oracle | 20/0 | 0.0020 (0.0001) | 0.0000 (0.0000) | 1.00 (0.00) | -0.00 (0.00) | 0.0000 (0.0000) | 0.800 (0.001) | 20/20 |
| S2_pi10_n500 | LR-MLE | 20/0 | 0.0206 (0.0020) | 0.0403 (0.0016) | 0.83 (0.04) | -0.00 (0.05) | 0.0014 (0.0039) | 0.772 (0.002) | 3/20 |
| S2_pi10_n500 | FLIC | 20/0 | 0.0197 (0.0020) | 0.0398 (0.0016) | 0.86 (0.04) | -0.00 (0.05) | 0.0015 (0.0039) | 0.772 (0.002) | 3/20 |
| S2_pi10_n500 | ridge-CV | 20/0 | 0.0207 (0.0026) | 0.0394 (0.0020) | 1.03 (0.07) | -0.00 (0.05) | 0.0013 (0.0039) | 0.774 (0.002) | 2/20 |
| S2_pi10_n500 | GAM | 20/0 | 0.0207 (0.0021) | 0.0374 (0.0021) | 0.81 (0.05) | -0.00 (0.05) | 0.0015 (0.0038) | 0.768 (0.006) | 4/20 |
| S2_pi10_n500 | LightGBM + Platt | 20/0 | 0.0211 (0.0022) | 0.0520 (0.0019) | 1.06 (0.08) | -0.03 (0.05) | 0.0034 (0.0040) | 0.740 (0.005) | 5/20 |
| S2_pi10_n500 | LightGBM raw | 20/0 | 0.0201 (0.0017) | 0.0509 (0.0016) | 1.04 (0.05) | 0.01 (0.05) | 0.0001 (0.0037) | 0.740 (0.005) | 1/20 |
| S2_pi10_n500 | TabPFN V2 (T 0.9) | 20/0 | 0.0205 (0.0021) | 0.0354 (0.0016) | 0.93 (0.06) | 0.17 (0.05) | -0.0118 (0.0034) | 0.776 (0.003) | 2/20 |
| S2_pi10_n500 | TabPFN V2 legacy seed | 20/0 | 0.0209 (0.0022) | 0.0357 (0.0017) | 0.93 (0.06) | 0.18 (0.05) | -0.0120 (0.0036) | 0.776 (0.003) | 2/20 |
| S2_pi10_n500 | TabPFN V2, T 1.0 | 20/0 | 0.0203 (0.0023) | 0.0375 (0.0018) | 1.04 (0.07) | -0.02 (0.05) | 0.0029 (0.0036) | 0.776 (0.003) | 3/20 |
| S2_pi10_n500 | TabICLv2 (T 0.9) | 20/0 | 0.0189 (0.0018) | 0.0340 (0.0014) | 0.90 (0.05) | 0.14 (0.05) | -0.0094 (0.0036) | 0.776 (0.002) | 3/20 |
| S2_pi10_n500 | Mitra-v2 | 20/0 | 0.0217 (0.0023) | 0.0382 (0.0017) | 1.02 (0.09) | -0.02 (0.05) | 0.0025 (0.0039) | 0.775 (0.002) | 2/20 |
| S2_pi10_n500 | TabPFN-3.5 (T 1.0) | 20/0 | 0.0206 (0.0023) | 0.0368 (0.0018) | 1.05 (0.07) | -0.00 (0.05) | 0.0012 (0.0038) | 0.777 (0.002) | 2/20 |
| S3_pi30_n1000 | oracle | 20/0 | 0.0029 (0.0000) | 0.0000 (0.0000) | 1.00 (0.00) | 0.00 (0.00) | 0.0000 (0.0000) | 0.783 (0.001) | 20/20 |
| S3_pi30_n1000 | LR-MLE | 20/0 | 0.0131 (0.0016) | 0.0481 (0.0011) | 0.97 (0.01) | 0.01 (0.02) | -0.0010 (0.0026) | 0.769 (0.001) | 15/20 |
| S3_pi30_n1000 | FLIC | 20/0 | 0.0124 (0.0016) | 0.0480 (0.0011) | 0.98 (0.01) | 0.01 (0.02) | -0.0009 (0.0026) | 0.769 (0.001) | 16/20 |
| S3_pi30_n1000 | ridge-CV | 20/0 | 0.0133 (0.0015) | 0.0487 (0.0012) | 1.08 (0.01) | 0.00 (0.02) | -0.0007 (0.0026) | 0.769 (0.001) | 14/20 |
| S3_pi30_n1000 | GAM | 20/0 | 0.0141 (0.0018) | 0.0505 (0.0012) | 0.96 (0.01) | 0.01 (0.02) | -0.0012 (0.0025) | 0.768 (0.001) | 14/20 |
| S3_pi30_n1000 | LightGBM + Platt | 20/0 | 0.0156 (0.0013) | 0.0716 (0.0015) | 1.05 (0.02) | 0.00 (0.02) | -0.0007 (0.0027) | 0.756 (0.001) | 13/20 |
| S3_pi30_n1000 | LightGBM raw | 20/0 | 0.0167 (0.0018) | 0.0717 (0.0015) | 1.01 (0.02) | 0.01 (0.02) | -0.0019 (0.0027) | 0.756 (0.001) | 13/20 |
| S3_pi30_n1000 | TabPFN V2 (T 0.9) | 20/0 | 0.0193 (0.0022) | 0.0468 (0.0016) | 0.93 (0.01) | 0.10 (0.02) | -0.0156 (0.0027) | 0.772 (0.001) | 9/20 |
| S3_pi30_n1000 | TabPFN V2 legacy seed | 20/0 | 0.0199 (0.0022) | 0.0472 (0.0016) | 0.93 (0.01) | 0.10 (0.02) | -0.0160 (0.0027) | 0.772 (0.001) | 8/20 |
| S3_pi30_n1000 | TabPFN V2, T 1.0 | 20/0 | 0.0127 (0.0015) | 0.0452 (0.0014) | 1.04 (0.02) | 0.02 (0.02) | -0.0033 (0.0026) | 0.772 (0.001) | 15/20 |
| S3_pi30_n1000 | TabICLv2 (T 0.9) | 20/0 | 0.0149 (0.0022) | 0.0458 (0.0016) | 0.96 (0.02) | 0.07 (0.02) | -0.0105 (0.0028) | 0.771 (0.001) | 11/20 |
| S3_pi30_n1000 | Mitra-v2 | 20/0 | 0.0133 (0.0014) | 0.0519 (0.0014) | 1.03 (0.02) | 0.01 (0.02) | -0.0014 (0.0026) | 0.769 (0.001) | 15/20 |
| S3_pi30_n1000 | TabPFN-3.5 (T 1.0) | 20/0 | 0.0131 (0.0013) | 0.0451 (0.0014) | 1.03 (0.02) | 0.00 (0.02) | -0.0005 (0.0025) | 0.772 (0.001) | 15/20 |
| S4_pi03_n3000 | oracle | 20/0 | 0.0008 (0.0000) | 0.0000 (0.0000) | 1.00 (0.00) | 0.00 (0.00) | 0.0000 (0.0000) | 0.795 (0.001) | 20/20 |
| S4_pi03_n3000 | LR-MLE | 20/0 | 0.0034 (0.0004) | 0.0067 (0.0005) | 0.93 (0.02) | 0.02 (0.03) | -0.0003 (0.0007) | 0.789 (0.001) | 8/20 |
| S4_pi03_n3000 | FLIC | 20/0 | 0.0033 (0.0004) | 0.0066 (0.0005) | 0.94 (0.02) | 0.02 (0.03) | -0.0003 (0.0007) | 0.789 (0.001) | 8/20 |
| S4_pi03_n3000 | ridge-CV | 20/0 | 0.0036 (0.0004) | 0.0066 (0.0004) | 1.04 (0.03) | 0.02 (0.03) | -0.0003 (0.0007) | 0.789 (0.001) | 7/20 |
| S4_pi03_n3000 | GAM | 20/0 | 0.0038 (0.0004) | 0.0080 (0.0005) | 0.88 (0.03) | 0.01 (0.02) | -0.0002 (0.0007) | 0.786 (0.001) | 6/20 |
| S4_pi03_n3000 | LightGBM + Platt | 20/0 | 0.0053 (0.0004) | 0.0144 (0.0003) | 1.03 (0.02) | 0.02 (0.03) | -0.0003 (0.0008) | 0.759 (0.001) | 7/20 |
| S4_pi03_n3000 | LightGBM raw | 20/0 | 0.0054 (0.0005) | 0.0143 (0.0003) | 0.98 (0.03) | 0.03 (0.03) | -0.0008 (0.0007) | 0.759 (0.001) | 9/20 |
| S4_pi03_n3000 | TabPFN V2 (T 0.9) | 20/0 | 0.0085 (0.0007) | 0.0099 (0.0004) | 0.97 (0.03) | 0.32 (0.03) | -0.0075 (0.0007) | 0.784 (0.002) | 2/20 |
| S4_pi03_n3000 | TabICLv2 (T 0.9) | 20/0 | 0.0078 (0.0006) | 0.0094 (0.0004) | 0.87 (0.02) | 0.29 (0.03) | -0.0068 (0.0006) | 0.786 (0.001) | 0/20 |
| S4_pi03_n3000 | Mitra-v2 | 20/0 | 0.0038 (0.0003) | 0.0087 (0.0003) | 1.03 (0.03) | 0.00 (0.03) | 0.0001 (0.0007) | 0.784 (0.002) | 7/20 |
| S4_pi03_n3000 | TabPFN-3.5 (T 1.0) | 20/0 | 0.0036 (0.0003) | 0.0077 (0.0003) | 1.00 (0.03) | 0.02 (0.03) | -0.0004 (0.0007) | 0.786 (0.001) | 8/20 |
| T2_pi10_n3000 | oracle | 20/0 | 0.0015 (0.0001) | 0.0000 (0.0000) | 1.00 (0.00) | 0.00 (0.00) | 0.0000 (0.0000) | 0.800 (0.001) | 20/20 |
| T2_pi10_n3000 | LR-MLE | 20/0 | 0.0074 (0.0006) | 0.0210 (0.0005) | 0.92 (0.01) | -0.00 (0.02) | 0.0002 (0.0012) | 0.791 (0.001) | 11/20 |
| T2_pi10_n3000 | FLIC | 20/0 | 0.0068 (0.0006) | 0.0208 (0.0006) | 0.94 (0.01) | -0.00 (0.02) | 0.0002 (0.0012) | 0.791 (0.001) | 13/20 |
| T2_pi10_n3000 | ridge-CV | 20/0 | 0.0075 (0.0009) | 0.0207 (0.0008) | 1.06 (0.02) | 0.00 (0.02) | -0.0000 (0.0012) | 0.792 (0.001) | 14/20 |
| T2_pi10_n3000 | GAM | 20/0 | 0.0100 (0.0007) | 0.0260 (0.0008) | 0.88 (0.01) | -0.01 (0.02) | 0.0007 (0.0012) | 0.787 (0.001) | 5/20 |
| T2_pi10_n3000 | LightGBM + Platt | 20/0 | 0.0098 (0.0008) | 0.0345 (0.0007) | 1.06 (0.02) | -0.01 (0.01) | 0.0006 (0.0012) | 0.776 (0.002) | 13/20 |
| T2_pi10_n3000 | LightGBM raw | 20/0 | 0.0105 (0.0008) | 0.0345 (0.0007) | 1.07 (0.02) | 0.02 (0.02) | -0.0012 (0.0013) | 0.776 (0.002) | 12/20 |
| T2_pi10_n3000 | TabPFN V2 (T 0.9) | 20/0 | 0.0181 (0.0011) | 0.0224 (0.0008) | 0.91 (0.02) | 0.24 (0.02) | -0.0171 (0.0012) | 0.794 (0.001) | 0/20 |
| T2_pi10_n3000 | TabICLv2 (T 0.9) | 20/0 | 0.0139 (0.0010) | 0.0203 (0.0008) | 0.92 (0.01) | 0.18 (0.02) | -0.0132 (0.0012) | 0.794 (0.001) | 2/20 |
| T2_pi10_n3000 | Mitra-v2 | 0/20 | - | - | - | - | - | - | - |
| T3_pi10_n500 | oracle | 20/0 | 0.0016 (0.0000) | 0.0000 (0.0000) | 1.00 (0.00) | -0.00 (0.00) | 0.0000 (0.0000) | 0.800 (0.002) | 20/20 |
| T3_pi10_n500 | LR-MLE | 20/0 | 0.0200 (0.0022) | 0.0413 (0.0024) | 0.80 (0.04) | -0.00 (0.03) | 0.0004 (0.0025) | 0.762 (0.005) | 2/20 |
| T3_pi10_n500 | FLIC | 20/0 | 0.0170 (0.0021) | 0.0400 (0.0023) | 0.84 (0.04) | -0.00 (0.03) | 0.0005 (0.0025) | 0.763 (0.005) | 4/20 |
| T3_pi10_n500 | ridge-CV | 20/0 | 0.0157 (0.0010) | 0.0385 (0.0018) | 1.14 (0.06) | 0.00 (0.03) | 0.0003 (0.0025) | 0.766 (0.004) | 1/20 |
| T3_pi10_n500 | GAM | 20/0 | 0.0232 (0.0022) | 0.0448 (0.0023) | 0.74 (0.05) | 0.01 (0.04) | -0.0001 (0.0026) | 0.756 (0.005) | 1/20 |
| T3_pi10_n500 | LightGBM + Platt | 20/0 | 0.0175 (0.0013) | 0.0552 (0.0011) | 1.04 (0.06) | -0.01 (0.03) | 0.0013 (0.0029) | 0.725 (0.005) | 3/20 |
| T3_pi10_n500 | LightGBM raw | 20/0 | 0.0191 (0.0017) | 0.0552 (0.0011) | 1.01 (0.09) | 0.02 (0.04) | -0.0011 (0.0031) | 0.725 (0.005) | 3/20 |
| T3_pi10_n500 | TabPFN V2 (T 0.9) | 20/0 | 0.0174 (0.0015) | 0.0383 (0.0014) | 0.97 (0.06) | 0.18 (0.03) | -0.0130 (0.0023) | 0.765 (0.004) | 2/20 |
| T3_pi10_n500 | TabPFN V2 legacy seed | 20/0 | 0.0178 (0.0015) | 0.0386 (0.0013) | 0.97 (0.06) | 0.18 (0.03) | -0.0132 (0.0024) | 0.765 (0.004) | 2/20 |
| T3_pi10_n500 | TabPFN V2, T 1.0 | 20/0 | 0.0145 (0.0010) | 0.0399 (0.0013) | 1.07 (0.07) | -0.02 (0.03) | 0.0021 (0.0023) | 0.765 (0.004) | 2/20 |
| T3_pi10_n500 | TabICLv2 (T 0.9) | 20/0 | 0.0165 (0.0015) | 0.0385 (0.0014) | 0.88 (0.05) | 0.15 (0.03) | -0.0107 (0.0024) | 0.764 (0.004) | 2/20 |
| T3_pi10_n500 | Mitra-v2 | 20/0 | 0.0149 (0.0014) | 0.0424 (0.0012) | 1.04 (0.07) | -0.01 (0.03) | 0.0009 (0.0026) | 0.762 (0.003) | 2/20 |
| T3_pi10_n500 | TabPFN-3.5 (T 1.0) | 20/0 | 0.0151 (0.0009) | 0.0391 (0.0013) | 1.11 (0.06) | 0.00 (0.03) | 0.0004 (0.0026) | 0.766 (0.004) | 1/20 |

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

| scenario | arm | R0 | R1_m100 | R2_m100 | R3_m100 | R1_m500 | R2_m500 | R3_m500 |
|---|---|---|---|---|---|---|---|---|
| S1_pi03_n200 | flic | 0.0215 (0.0024) | 0.0206 (0.0013) [1F] | 0.0179 (0.0016) [1F] | 0.0218 (0.0018) [1F] | 0.0186 (0.0018) | 0.0089 (0.0010) | 0.0131 (0.0012) |
| S1_pi03_n200 | tabpfn | 0.0121 (0.0015) | 0.0159 (0.0015) [1F] | 0.0197 (0.0020) [1F] | 0.0231 (0.0018) [1F] | 0.0130 (0.0012) | 0.0092 (0.0009) | 0.0129 (0.0009) |
| S1_pi03_n200 | tabicl | 0.0141 (0.0018) | 0.0166 (0.0016) [1F] | 0.0190 (0.0019) [1F] | 0.0236 (0.0013) [1F] | 0.0128 (0.0014) | 0.0096 (0.0009) | 0.0126 (0.0010) |
| S1_pi03_n200 | mitra | 0.0153 (0.0016) | 0.0161 (0.0016) [1F] | 0.0196 (0.0021) [1F] | 0.0234 (0.0016) [1F] | 0.0129 (0.0012) | 0.0097 (0.0008) | 0.0127 (0.0010) |
| S2_pi10_n500 | flic | 0.0197 (0.0020) | 0.0316 (0.0033) | 0.0353 (0.0036) | 0.0432 (0.0039) | 0.0161 (0.0016) | 0.0142 (0.0012) | 0.0198 (0.0017) |
| S2_pi10_n500 | tabpfn | 0.0205 (0.0021) | 0.0312 (0.0038) | 0.0334 (0.0039) | 0.0437 (0.0045) | 0.0167 (0.0019) | 0.0146 (0.0011) | 0.0197 (0.0015) |
| S2_pi10_n500 | tabicl | 0.0189 (0.0018) | 0.0302 (0.0037) | 0.0325 (0.0040) | 0.0440 (0.0042) | 0.0154 (0.0016) | 0.0140 (0.0013) | 0.0200 (0.0016) |
| S2_pi10_n500 | mitra | 0.0217 (0.0023) | 0.0312 (0.0038) | 0.0336 (0.0037) | 0.0437 (0.0046) | 0.0181 (0.0018) | 0.0166 (0.0011) | 0.0207 (0.0015) |
| S3_pi30_n1000 | flic | 0.0124 (0.0016) | 0.0319 (0.0037) | 0.0451 (0.0044) | 0.0711 (0.0060) | 0.0194 (0.0027) | 0.0216 (0.0025) | 0.0321 (0.0024) |
| S3_pi30_n1000 | tabpfn | 0.0193 (0.0022) | 0.0322 (0.0037) | 0.0440 (0.0042) | 0.0695 (0.0059) | 0.0218 (0.0025) | 0.0224 (0.0022) | 0.0314 (0.0020) |
| S3_pi30_n1000 | tabicl | 0.0149 (0.0022) | 0.0318 (0.0037) | 0.0438 (0.0040) | 0.0675 (0.0054) | 0.0203 (0.0025) | 0.0218 (0.0022) | 0.0320 (0.0020) |
| S3_pi30_n1000 | mitra | 0.0133 (0.0014) | 0.0313 (0.0035) | 0.0423 (0.0040) | 0.0676 (0.0046) | 0.0197 (0.0024) | 0.0216 (0.0022) | 0.0306 (0.0022) |
| S4_pi03_n3000 | flic | 0.0033 (0.0004) | 0.0167 (0.0025) [1F] | 0.0183 (0.0022) [2F] | 0.0255 (0.0030) [1F] | 0.0062 (0.0010) | 0.0082 (0.0011) | 0.0112 (0.0011) |
| S4_pi03_n3000 | tabpfn | 0.0085 (0.0007) | 0.0167 (0.0023) [1F] | 0.0201 (0.0021) [2F] | 0.0257 (0.0032) [1F] | 0.0067 (0.0010) | 0.0081 (0.0010) | 0.0112 (0.0012) |
| S4_pi03_n3000 | tabicl | 0.0078 (0.0006) | 0.0172 (0.0023) [1F] | 0.0190 (0.0023) [2F] | 0.0270 (0.0028) [1F] | 0.0068 (0.0010) | 0.0082 (0.0011) | 0.0110 (0.0012) |
| S4_pi03_n3000 | mitra | 0.0038 (0.0003) | 0.0161 (0.0023) [1F] | 0.0196 (0.0022) [2F] | 0.0250 (0.0021) [1F] | 0.0065 (0.0009) | 0.0084 (0.0011) | 0.0112 (0.0012) |
| T2_pi10_n3000 | flic | 0.0068 (0.0006) | 0.0263 (0.0037) | 0.0309 (0.0035) | 0.0435 (0.0035) | 0.0097 (0.0010) | 0.0120 (0.0011) | 0.0220 (0.0015) |
| T2_pi10_n3000 | tabpfn | 0.0181 (0.0011) | 0.0265 (0.0035) | 0.0297 (0.0034) | 0.0430 (0.0045) | 0.0120 (0.0008) | 0.0115 (0.0012) | 0.0208 (0.0015) |
| T2_pi10_n3000 | tabicl | 0.0139 (0.0010) | 0.0258 (0.0036) | 0.0304 (0.0035) | 0.0439 (0.0045) | 0.0106 (0.0009) | 0.0110 (0.0012) | 0.0216 (0.0016) |
| T2_pi10_n3000 | mitra | - (-) [20F] | - (-) | - (-) | - (-) | - (-) | - (-) | - (-) |
| T3_pi10_n500 | flic | 0.0170 (0.0021) | 0.0332 (0.0042) | 0.0359 (0.0036) | 0.0487 (0.0042) | 0.0180 (0.0024) | 0.0138 (0.0019) | 0.0214 (0.0019) |
| T3_pi10_n500 | tabpfn | 0.0174 (0.0015) | 0.0307 (0.0041) | 0.0371 (0.0032) | 0.0468 (0.0034) | 0.0159 (0.0017) | 0.0154 (0.0018) | 0.0211 (0.0019) |
| T3_pi10_n500 | tabicl | 0.0165 (0.0015) | 0.0305 (0.0042) | 0.0368 (0.0033) | 0.0485 (0.0038) | 0.0158 (0.0021) | 0.0153 (0.0018) | 0.0219 (0.0020) |
| T3_pi10_n500 | mitra | 0.0149 (0.0014) | 0.0310 (0.0043) | 0.0375 (0.0039) | 0.0499 (0.0040) | 0.0169 (0.0019) | 0.0151 (0.0019) | 0.0205 (0.0019) |

- **m = 100 makes calibration worse.** It raises the true ICI above R0 for every arm and level, except FLIC's R1 and R2 at S1_pi03_n200, because the noise of a 100-row recalibration fit outweighs the bias it removes.
- **m = 500 helps only arms that start above its own noise floor.** R2 lowers the true ICI at S1, S2 and T3, and for TabPFN V2 and TabICL at T2. It raises it for every arm at S3 (to about 0.022) and for FLIC and Mitra at S4 and T2. A 500-row logistic recalibration has a noise floor of about 0.008 (S4) to 0.022 (S3).
- **The test-guided rule matters.** These results are what make it relevant to plan H5: recalibrating a well-calibrated model with 100-500 rows does harm.
- **The R1-R3 failures at m = 100 and pi = 3 %** are subsamples with no events (1-2 of 20). They are recorded, as designed.

---

## 4. GOF layer

### Size on each arm's own predictions (exact null Y* ~ Bern(p-hat); rejection rate at 0.05 (MCSE), pooled over the arms of each scenario)

| scenario | EDGE10 | EDGEa | Cox | CITL | Spieg | GiViTI | HL | Stukel | leCes | Loc | DG1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_pi03_n200 | 0.060 (0.016) [8 NR] | 0.055 (0.015) [2 NR] | 0.050 (0.014) | 0.050 (0.014) | 0.037 (0.012) | 0.051 (0.014) [4 NR] | 0.058 (0.015) | 0.050 (0.014) | 0.075 (0.017) | 0.059 (0.019) [7 NR] | - |
| S2_pi10_n500 | 0.067 (0.016) | 0.050 (0.014) | 0.054 (0.015) | 0.029 (0.011) | 0.037 (0.012) | 0.054 (0.015) | 0.050 (0.014) | 0.058 (0.015) | 0.042 (0.013) | 0.050 (0.017) | - |
| S3_pi30_n1000 | 0.037 (0.012) | 0.042 (0.013) | 0.037 (0.012) | 0.050 (0.014) | 0.037 (0.012) | 0.037 (0.012) | 0.033 (0.012) | 0.046 (0.013) | 0.037 (0.012) | 0.025 (0.012) | - |
| S4_pi03_n3000 | 0.055 (0.016) | 0.050 (0.015) | 0.050 (0.015) | 0.050 (0.015) | 0.050 (0.015) | 0.050 (0.015) | 0.045 (0.015) | 0.010 (0.007) | 0.055 (0.016) | 0.033 (0.016) | - |
| T2_pi10_n3000 | 0.056 (0.017) | 0.078 (0.020) | 0.078 (0.020) | 0.067 (0.019) | 0.078 (0.020) | 0.078 (0.020) | 0.033 (0.013) | 0.072 (0.019) | 0.094 (0.022) | 0.080 (0.027) | - |
| T3_pi10_n500 | 0.033 (0.012) | 0.042 (0.013) | 0.046 (0.013) | 0.037 (0.012) | 0.054 (0.015) | 0.042 (0.013) | 0.033 (0.012) | 0.075 (0.017) | 0.058 (0.015) | 0.037 (0.015) | - |
| pooled | 0.051 (0.006) [8 NR] | 0.052 (0.006) [2 NR] | 0.051 (0.006) | 0.046 (0.006) | 0.048 (0.006) | 0.051 (0.006) [4 NR] | 0.043 (0.006) | 0.052 (0.006) | 0.059 (0.006) | 0.046 (0.007) [7 NR] | - |

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

| scenario | arm | EDGE10 | Cox | CITL | HL | Stukel | leCes | Loc | DG1 |
|---|---|---|---|---|---|---|---|---|---|
| S1_pi03_n200 | oracle | 1/20 | 0/20 | 0/20 | 1/20 | 0/20 | 2/20 | 0/20 | - |
| S1_pi03_n200 | LR-MLE | 20/20 | 20/20 | 14/20 | 20/20 | 19/20 | 20/20 | - | - |
| S1_pi03_n200 | FLIC | 20/20 | 20/20 | 14/20 | 18/20 | 17/20 | 20/20 | 20/20 | 20/20 |
| S1_pi03_n200 | ridge-CV | 16/19 (1 NR) | 18/20 | 13/20 | 17/20 | 15/20 | 20/20 | - | - |
| S1_pi03_n200 | GAM | 13/13 (7 NR) | 20/20 | 15/20 | 20/20 | 20/20 | 20/20 | 13/13 (7 NR) | - |
| S1_pi03_n200 | LightGBM + Platt | 18/20 | 19/20 | 18/20 | 19/20 | 15/20 | 20/20 | - | - |
| S1_pi03_n200 | LightGBM raw | 16/20 | 18/20 | 14/20 | 17/20 | 12/20 | 20/20 | - | - |
| S1_pi03_n200 | TabPFN V2 (T 0.9) | 16/20 | 16/20 | 12/20 | 14/20 | 12/20 | 19/20 | 20/20 | 18/20 |
| S1_pi03_n200 | TabPFN V2 legacy seed | 18/20 | 18/20 | 12/20 | 16/20 | 12/20 | 19/20 | 20/20 | - |
| S1_pi03_n200 | TabPFN V2, T 1.0 | 19/20 | 18/20 | 14/20 | 17/20 | 13/20 | 19/20 | 20/20 | - |
| S1_pi03_n200 | TabICLv2 (T 0.9) | 18/20 | 19/20 | 12/20 | 17/20 | 15/20 | 20/20 | 20/20 | 19/20 |
| S1_pi03_n200 | Mitra-v2 | 18/20 | 19/20 | 14/20 | 17/20 | 12/20 | 20/20 | 20/20 | 19/20 |
| S1_pi03_n200 | TabPFN-3.5 (T 1.0) | 16/20 | 16/20 | 14/20 | 14/20 | 14/20 | 19/20 | 20/20 | 18/20 |
| S2_pi10_n500 | oracle | 2/20 | 0/20 | 1/20 | 2/20 | 0/20 | 1/20 | 0/20 | - |
| S2_pi10_n500 | LR-MLE | 11/20 | 10/20 | 5/20 | 10/20 | 7/20 | 15/20 | - | - |
| S2_pi10_n500 | FLIC | 9/20 | 9/20 | 5/20 | 10/20 | 6/20 | 15/20 | 20/20 | 10/20 |
| S2_pi10_n500 | ridge-CV | 12/20 | 12/20 | 5/20 | 8/20 | 8/20 | 13/20 | - | - |
| S2_pi10_n500 | GAM | 10/20 | 10/20 | 6/20 | 9/20 | 7/20 | 14/20 | 14/20 | - |
| S2_pi10_n500 | LightGBM + Platt | 11/20 | 10/20 | 6/20 | 7/20 | 7/20 | 19/20 | - | - |
| S2_pi10_n500 | LightGBM raw | 9/20 | 9/20 | 5/20 | 7/20 | 6/20 | 17/20 | - | - |
| S2_pi10_n500 | TabPFN V2 (T 0.9) | 11/20 | 13/20 | 11/20 | 10/20 | 7/20 | 14/20 | 15/20 | 11/20 |
| S2_pi10_n500 | TabPFN V2 legacy seed | 13/20 | 13/20 | 11/20 | 11/20 | 8/20 | 14/20 | 14/20 | - |
| S2_pi10_n500 | TabPFN V2, T 1.0 | 11/20 | 10/20 | 6/20 | 9/20 | 10/20 | 14/20 | 13/20 | - |
| S2_pi10_n500 | TabICLv2 (T 0.9) | 13/20 | 12/20 | 9/20 | 12/20 | 8/20 | 14/20 | 18/20 | 10/20 |
| S2_pi10_n500 | Mitra-v2 | 10/20 | 11/20 | 5/20 | 11/20 | 10/20 | 15/20 | 18/20 | 11/20 |
| S2_pi10_n500 | TabPFN-3.5 (T 1.0) | 11/20 | 10/20 | 5/20 | 9/20 | 10/20 | 14/20 | 15/20 | 13/20 |
| S3_pi30_n1000 | oracle | 2/20 | 2/20 | 1/20 | 1/20 | 3/20 | 1/20 | 2/20 | - |
| S3_pi30_n1000 | LR-MLE | 1/20 | 4/20 | 3/20 | 0/20 | 1/20 | 6/20 | - | - |
| S3_pi30_n1000 | FLIC | 1/20 | 4/20 | 3/20 | 0/20 | 3/20 | 6/20 | 9/20 | 3/20 |
| S3_pi30_n1000 | ridge-CV | 1/20 | 6/20 | 3/20 | 0/20 | 4/20 | 5/20 | - | - |
| S3_pi30_n1000 | GAM | 2/20 | 4/20 | 3/20 | 1/20 | 1/20 | 7/20 | 11/20 | - |
| S3_pi30_n1000 | LightGBM + Platt | 1/20 | 5/20 | 2/20 | 0/20 | 6/20 | 11/20 | - | - |
| S3_pi30_n1000 | LightGBM raw | 5/20 | 6/20 | 2/20 | 1/20 | 6/20 | 11/20 | - | - |
| S3_pi30_n1000 | TabPFN V2 (T 0.9) | 2/20 | 4/20 | 5/20 | 4/20 | 5/20 | 9/20 | 7/20 | 5/20 |
| S3_pi30_n1000 | TabPFN V2 legacy seed | 3/20 | 5/20 | 6/20 | 5/20 | 5/20 | 10/20 | 7/20 | - |
| S3_pi30_n1000 | TabPFN V2, T 1.0 | 0/20 | 4/20 | 2/20 | 1/20 | 4/20 | 7/20 | 3/20 | - |
| S3_pi30_n1000 | TabICLv2 (T 0.9) | 2/20 | 3/20 | 4/20 | 2/20 | 4/20 | 9/20 | 6/20 | 3/20 |
| S3_pi30_n1000 | Mitra-v2 | 3/20 | 4/20 | 3/20 | 2/20 | 7/20 | 9/20 | 7/20 | 2/20 |
| S3_pi30_n1000 | TabPFN-3.5 (T 1.0) | 0/20 | 4/20 | 3/20 | 1/20 | 4/20 | 8/20 | 4/20 | 4/20 |
| S4_pi03_n3000 | oracle | 1/20 | 1/20 | 3/20 | 0/20 | 3/20 | 0/20 | 0/20 | - |
| S4_pi03_n3000 | LR-MLE | 5/20 | 6/20 | 4/20 | 5/20 | 5/20 | 5/20 | - | - |
| S4_pi03_n3000 | FLIC | 4/20 | 5/20 | 4/20 | 5/20 | 3/20 | 5/20 | 8/20 | 6/20 |
| S4_pi03_n3000 | ridge-CV | 3/20 | 3/20 | 4/20 | 3/20 | 1/20 | 4/20 | - | - |
| S4_pi03_n3000 | GAM | 7/20 | 6/20 | 4/20 | 8/20 | 5/20 | 7/20 | 10/20 | - |
| S4_pi03_n3000 | LightGBM + Platt | 7/20 | 5/20 | 4/20 | 5/20 | 1/20 | 12/20 | - | - |
| S4_pi03_n3000 | LightGBM raw | 7/20 | 6/20 | 4/20 | 6/20 | 2/20 | 11/20 | - | - |
| S4_pi03_n3000 | TabPFN V2 (T 0.9) | 15/20 | 10/20 | 14/20 | 15/20 | 9/20 | 15/20 | 13/20 | 16/20 |
| S4_pi03_n3000 | TabICLv2 (T 0.9) | 14/20 | 15/20 | 12/20 | 15/20 | 14/20 | 12/20 | 13/20 | 13/20 |
| S4_pi03_n3000 | Mitra-v2 | 7/20 | 5/20 | 4/20 | 5/20 | 1/20 | 6/20 | 4/20 | 7/20 |
| S4_pi03_n3000 | TabPFN-3.5 (T 1.0) | 5/20 | 5/20 | 4/20 | 2/20 | 1/20 | 7/20 | 6/20 | 6/20 |
| T2_pi10_n3000 | oracle | 0/20 | 0/20 | 1/20 | 0/20 | 2/20 | 2/20 | 1/20 | - |
| T2_pi10_n3000 | LR-MLE | 3/20 | 7/20 | 3/20 | 2/20 | 2/20 | 5/20 | - | - |
| T2_pi10_n3000 | FLIC | 1/20 | 6/20 | 3/20 | 1/20 | 1/20 | 5/20 | 2/20 | 3/20 |
| T2_pi10_n3000 | ridge-CV | 1/20 | 4/20 | 3/20 | 1/20 | 2/20 | 4/20 | - | - |
| T2_pi10_n3000 | GAM | 3/20 | 6/20 | 3/20 | 3/20 | 4/20 | 6/20 | 4/20 | - |
| T2_pi10_n3000 | LightGBM + Platt | 5/20 | 5/20 | 3/20 | 1/20 | 2/20 | 5/20 | - | - |
| T2_pi10_n3000 | LightGBM raw | 5/20 | 5/20 | 3/20 | 3/20 | 4/20 | 6/20 | - | - |
| T2_pi10_n3000 | TabPFN V2 (T 0.9) | 11/20 | 12/20 | 11/20 | 9/20 | 8/20 | 12/20 | 13/20 | 13/20 |
| T2_pi10_n3000 | TabICLv2 (T 0.9) | 6/20 | 10/20 | 10/20 | 6/20 | 7/20 | 11/20 | 11/20 | 10/20 |
| T3_pi10_n500 | oracle | 0/20 | 1/20 | 1/20 | 2/20 | 2/20 | 1/20 | 0/20 | - |
| T3_pi10_n500 | LR-MLE | 11/20 | 13/20 | 3/20 | 11/20 | 10/20 | 13/20 | - | - |
| T3_pi10_n500 | FLIC | 9/20 | 10/20 | 3/20 | 9/20 | 7/20 | 13/20 | 10/20 | 11/20 |
| T3_pi10_n500 | ridge-CV | 6/20 | 7/20 | 3/20 | 3/20 | 6/20 | 7/20 | - | - |
| T3_pi10_n500 | GAM | 15/20 | 15/20 | 4/20 | 13/20 | 10/20 | 15/20 | 15/20 | - |
| T3_pi10_n500 | LightGBM + Platt | 5/20 | 6/20 | 5/20 | 5/20 | 1/20 | 16/20 | - | - |
| T3_pi10_n500 | LightGBM raw | 8/20 | 11/20 | 4/20 | 8/20 | 5/20 | 16/20 | - | - |
| T3_pi10_n500 | TabPFN V2 (T 0.9) | 7/20 | 11/20 | 8/20 | 9/20 | 9/20 | 13/20 | 11/20 | 13/20 |
| T3_pi10_n500 | TabPFN V2 legacy seed | 9/20 | 11/20 | 8/20 | 8/20 | 9/20 | 14/20 | 12/20 | - |
| T3_pi10_n500 | TabPFN V2, T 1.0 | 5/20 | 7/20 | 2/20 | 3/20 | 5/20 | 7/20 | 12/20 | - |
| T3_pi10_n500 | TabICLv2 (T 0.9) | 9/20 | 10/20 | 8/20 | 6/20 | 10/20 | 12/20 | 12/20 | 13/20 |
| T3_pi10_n500 | Mitra-v2 | 4/20 | 6/20 | 2/20 | 4/20 | 4/20 | 13/20 | 10/20 | 9/20 |
| T3_pi10_n500 | TabPFN-3.5 (T 1.0) | 6/20 | 8/20 | 3/20 | 3/20 | 5/20 | 9/20 | 11/20 | 6/20 |

- **The oracle row is the main run's built-in size check.** Here it rejects in 0-3 of 20 replicates per test and scenario.
- **The detection pattern follows the oracle estimands of section 3.** At S1_pi03_n200, most tests reject every arm in 12-20 of 20 replicates. The temperature-driven a* > 0 of TabPFN V2 and TabICL at S4 and T2 is detected in 6-16 of 20. At S3_pi30_n1000, where the arms are close to calibrated, the rates are low.

### GOF cost (core-seconds per replicate, all arms, B = 199, 7 workers on a machine running the GPU process)

| scenario | count | mean | median | max |
|---|---|---|---|---|
| S1_pi03_n200 | 20 | 156.0 | 157.0 | 160.0 |
| S2_pi10_n500 | 20 | 105.0 | 105.0 | 110.0 |
| S3_pi30_n1000 | 20 | 99.0 | 101.0 | 106.0 |
| S4_pi03_n3000 | 20 | 129.0 | 128.0 | 145.0 |
| T2_pi10_n3000 | 20 | 811.0 | 752.0 | 1094.0 |
| T3_pi10_n500 | 20 | 171.0 | 157.0 | 284.0 |

Share of that time by call type (%):

| scenario | battery | deepgof | edge.gof | localize |
|---|---|---|---|---|
| S1_pi03_n200 | 42.4 | 51.6 | 0.2 | 5.8 |
| S2_pi10_n500 | 20.7 | 76.0 | 0.2 | 3.0 |
| S3_pi30_n1000 | 19.4 | 77.4 | 0.2 | 3.0 |
| S4_pi03_n3000 | 38.6 | 56.2 | 0.2 | 5.0 |
| T2_pi10_n3000 | 2.0 | 96.6 | 0.0 | 1.4 |
| T3_pi10_n500 | 11.0 | 87.1 | 0.1 | 1.9 |

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

| plan | GPU h: TabPFN-3.5 / TabICL / Mitra / legacy V2 / T-sens | GPU h total | CPU core-h: classical fits / TabPFN-3.5 T2 on CPU / GOF battery / DeepGOF / scoring | wall h: GPU / CPU (14 cores) | days |
|---|---|---|---|---|---|
| P1 plan as written: R 200, legacy 200, T-sensitivity 200 (8 sc), DeepGOF B 199, GOF on every classical rep | 89 / 15 / 25 / 12 / 13 | 153 | 52 / 60 / 228 / 421 / 66 | 153 / 55 | 6.4 |
| P2 R 200; legacy 100; no T-sensitivity; DeepGOF B 199 (T2: 99); reps 201-500 GOF on the oracle only | 89 / 15 / 25 / 6 / 0 | 135 | 52 / 60 / 146 / 367 / 66 | 135 / 45 | 5.6 |
| P3 as P2 with DeepGOF B 99 everywhere | 89 / 15 / 25 / 6 / 0 | 135 | 52 / 60 / 146 / 211 / 66 | 135 / 34 | 5.6 |
| P4 (recommended) as P3 with R 150 at n = 3000 | 78 / 12 / 21 / 6 / 0 | 118 | 52 / 45 / 137 / 193 / 64 | 118 / 32 | 4.9 |
| P5 lean: R 150 everywhere; legacy 100; DeepGOF B 99 | 67 / 11 / 19 / 6 / 0 | 102 | 52 / 45 / 115 / 158 / 60 | 102 / 27 | 4.3 |

### 8.3 After moving whole TabPFN-3.5 scenarios to CPU workers until GPU and CPU finish together

| plan | 16 cores, f 2.4 | 16 cores, f 3.0 | 20 cores, f 2.4 | 20 cores, f 3.0 |
|---|---|---|---|---|
| P1 plan as written: R 200, legacy 200, T-sensitivity 200 (8 sc), DeepGOF B 199, GOF on every classical rep | 4.5 d (move 46 GPU-h) | 4.7 d (move 41 GPU-h) | 4.0 d (move 57 GPU-h) | 4.3 d (move 51 GPU-h) |
| P2 R 200; legacy 100; no T-sensitivity; DeepGOF B 199 (T2: 99); reps 201-500 GOF on the oracle only | 3.9 d (move 42 GPU-h) | 4.1 d (move 37 GPU-h) | 3.5 d (move 51 GPU-h) | 3.7 d (move 45 GPU-h) |
| P3 as P2 with DeepGOF B 99 everywhere | 3.7 d (move 46 GPU-h) | 3.9 d (move 40 GPU-h) | 3.3 d (move 55 GPU-h) | 3.6 d (move 49 GPU-h) |
| P4 (recommended) as P3 with R 150 at n = 3000 | 3.3 d (move 40 GPU-h) | 3.5 d (move 35 GPU-h) | 2.9 d (move 48 GPU-h) | 3.1 d (move 42 GPU-h) |
| P5 lean: R 150 everywhere; legacy 100; DeepGOF B 99 | 2.8 d (move 34 GPU-h) | 3.0 d (move 30 GPU-h) | 2.5 d (move 41 GPU-h) | 2.7 d (move 37 GPU-h) |

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
