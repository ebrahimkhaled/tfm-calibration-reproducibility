# Analysis pipeline

This folder turns the main run's outputs into every table and figure that the pre-registration
(`../prereg/preregistration.md`, OSF doi 10.17605/OSF.IO/8HMTZ) plans. It adds no estimand and no
comparison: each output is tied to its section of the protocol, and the fixed constants (margins,
thresholds, R, the 80 % rule) live in one place, `_common.py`.

One command rebuilds everything once the run is done:

```sh
bash analysis/run_all.sh                    # WORKERS=2 B=500 by default
```

## While the run is going

- Every Python and R process lowers itself to below-normal priority at start (`_common.py`,
  `_common.R`), and children such as Rscript and the worker pools inherit it.
- BLAS/OpenMP use one thread, and at most 2 worker processes run at once. The cap is
  `ANALYSIS_MAX_WORKERS` (default 2). Raise it only after the run has finished.
- Nothing touches the GPU.
- The run's folders (`results/preds`, `results/gof*`, `logs/`, `results/metrics`) are only read.
  Every write goes through `assert_writable()` / `check_out()`, which refuse any path outside `analysis/`.
- Files changed in the last 2 minutes, or with a live `.lock` beside them, are left for the next pass.
- **The partial-data guard.** Until `tables/manifest.json` says `complete` (every planned arm x
  scenario x replicate is present, and the GOF layer has its 500 oracle nulls everywhere):
  - every summary table carries `data_state = "PARTIAL - NOT FOR INTERPRETATION"`;
  - every figure carries that watermark;
  - the console prints no TFM label.

  Figures drawn from the synthetic random tier carry "SYNTHETIC DATA - PIPELINE TEST ONLY".

## Files

| File | What it does | Protocol |
|---|---|---|
| `_common.py`, `_common.R` | Paths, process limits, arm lists, the fixed margins and thresholds | 4.5, 7.0 |
| `rules.py` | MCSE, medians with order-statistic intervals, Hodges-Lehmann with the signed-rank interval, every label rule, the minimum-n rule, the size rule, the pairing and 5 % rules, and the reporting rule of 6.3. Pure functions; no data. | 4.5, 6.1, 6.3, 7, 8.2 |
| `collect.py` | Tidy tables `tables/{fits,scores,gof,pairs}.parquet` and `manifest.json`. Scores come from the run's `results/metrics/<scenario>.parquet` for every prediction file older than it. Any newer file is scored in-process by `code/score.py::score_file` (same code, same fingerprint check). Incremental through `tables/cache/`. | 4, 8.2 |
| `summarise.py` | All summaries, labels and counts: `tables/summary/*.csv` (list in its docstring) | 6.1-6.3, 7.1-7.6, 11 |
| `detection_curves.R` | Logistic GAM detection curves, `s(true ICI, k = 10)`, REML, with a cluster bootstrap (B = 500), plus EDGE minus HL at 0.01-0.05 | 6.2, 7.4 |
| `meta_model.R` | The random-tier meta-model: scaled-t GAM, smooths by model, a design random effect, simultaneous bands from 10,000 draws, regions, shares, maps, transfer predictions, and the detection and size meta-models | 6.4, 7.6 |
| `make_synthetic_random.R` | A synthetic random tier (design settings from `code/random_designs.csv`, invented responses with a known truth). It is used only to test `meta_model.R`. | - |
| `figures/_style.py` | The house style. Arial 7 pt (panel labels bold 8 pt), lines >= 1 pt, Okabe-Ito with a fixed colour, marker and line style per model, vector PDF at 89 or 183 mm. | - |
| `figures/fig*.py` | One script per planned figure (below) | 11 |
| `figures/measure_fonts.py` | Reads every PDF's content streams and reports the font sizes, stroke widths and page width. It fails if any text is under 7 pt, any stroke is under 1 pt, or the width is not 89 or 183 mm. Writes `figures/out/font_report.csv`. | - |
| `tests/test_rules.py` | Unit tests (23). The Hodges-Lehmann interval and `qsignrank` are checked against R. | - |
| `tests/check_meta_synthetic.py` | Checks the meta-model on the synthetic tier against its known truth | 6.4 |
| `run_all.sh` | The one-command rebuild | - |

## Display items (proposal: 7 figures + 1 table in the main paper)

| # | Item | Script | Question |
|---|---|---|---|
| Fig. 1 | Mean true ICI against n by event rate and shape; the oracle row is the estimator's floor | `fig1_true_ici_vs_n.py` | A1, primary 1 |
| Fig. 2 | Distributions of the true slope b* in the 8 main cells | `fig2_slope_distributions.py` | Q1 |
| Fig. 3 | Minimum-n-for-calibration heat-map table (weak, moderate), with Riley's n | `fig3_min_n_table.py` | Q2, primary 4 |
| Fig. 4 | \|a\*\| against FLIC at pi = 3 %, with the shipped temperatures, and the temperature effect | `fig4_intercept_temperature.py` | Q3 |
| Fig. 5 | Detection curves of every test, with a size inset | `fig5_detection_curves.py` | Q4 |
| Fig. 6 | Recalibration-ladder slopegraph (n = 500) | `fig6_recalibration_ladder.py` | Q5 |
| Fig. 7 | Robustness map of the random tier | `fig7_robustness_map.py` | Q6, primary 5 |
| Table 1 | Summary of findings, one row per question with its label counts | `tables/summary/summary_of_findings.csv` | 7.0 |

The real-data panel (`fig8_realdata.py`) is planned as an eighth figure. Two options:

- move it to the main text in place of Fig. 4, which would then go to the Supplement;
- keep it in the Supplement if the real-data stream is late.

The script draws nothing until a real-data learning-curve table exists.

**Supplementary list:**

- S1: Fig. 1 with every arm (`fig1 --all-arms`).
- S2: the minimum-n table at the lower 95 % limit and at half or double the thresholds, and with `true_ll_ici` (`fig3 --lower-limit`, `--variant half|double|ll`).
- S3: the ladder at n = 200, 1000 and 3000 (`fig6 --n`).
- S4: the D_logb robustness map (`fig7 --response D_logb`).
- S5: the partial effects of the meta-model (`figS_meta_partial_effects.py`).
- S6: the design schematic.
- S7: the size table, every test on every arm (`q4a_size.csv`).
- S8: rejection against validation size, e50/e200 (`rejection_rates`).
- S9: empirical against true ICI (`ici_gap`).
- S10: localization, COV named on S3/T3.
- S11: failure, degenerate and flag counts (`fit_counts.csv`).
- S12: the legacy arm (exploratory).
- S13: prediction instability (exploratory).

## Time of a full rebuild (measured on 2026-10-10, 2 workers, below-normal priority, with the run going)

| Step | Measured | Full run (estimate) |
|---|---|---|
| Unit tests | 25 s | the same |
| `collect.py`, scores from the run's `results/metrics` | (not yet written by the run) | about 20-40 min (reads metadata and the GOF files; later passes are incremental) |
| `collect.py`, in-process scoring when `results/metrics` is absent or stale | 2,665 s for 580 files (4.6 s per file of wall time with 2 workers; about 20 s per file at N_eval = 3,334 with every ladder column) | about 37 h for all ~29,000 files with 2 workers. Avoid it: the run's own `score.py` pass writes `results/metrics` with 6 workers. |
| `collect.py`, GOF files only | 1,064 s for 596,832 rows (4 scenarios done by the GOF layer) | about 1-1.5 h the first time, then incremental |
| `summarise.py` without the detection curves | 99-112 s on 16,440 score rows (R = 10) | about 30-60 min (the bootstrap MCSEs grow with R = 500) |
| Detection curves (R, B = 200, 2 workers) | 75 s on 5,119 rows | about 2-4 h at B = 500 on about 150,000 rows per test |
| Meta-model, exact `gam` path | 848 s on 300 synthetic designs (both responses) | about 2-3 h per response on 1,000 designs |
| Meta-model, `bam` approximation | 205 s on 1,000 synthetic designs | - |
| Figures + font check | about 1 min | the same |

**Total once the run is done and `results/metrics` exists:** about 5-9 h at 2 workers. Most of it
is the detection bootstrap and the exact meta-model fit. With `ANALYSIS_MAX_WORKERS=6 WORKERS=6`
after the run, the detection bootstrap shrinks about threefold.

## Choices the protocol leaves open (implemented as below; for the author to confirm)

These are implementation details, not new analyses. Each is fixed here before any TFM result was looked at.

1. **Rejection.** A test rejects when p <= 0.05. A test row with no p-value ("Not run", "Failed")
   counts neither way (8.2).
2. **Class shares and minimum n.** The denominator is every replicate the arm was attempted on.
   A failed fit counts as not meeting the class (failures are kept). Q2 compares the TFM with FLIC
   on the replicates where both are usable (7.0).
3. **The median b\*** is taken over the finite b\*. The b\* <= 0 values are kept, and the count of
   non-finite values is reported. The |log b\*| differences use the rule of 7.1 (|log b\*| = 10).
4. **The |a\*| difference** leaves out a pair with a non-finite a\*, and the count is reported. The
   protocol states the "= 10" rule only for b\*.
5. **The Hodges-Lehmann interval.** It uses the exact signed-rank critical value at every R. This
   is R's `wilcox.test(exact = TRUE)`; by default R switches to a normal approximation at R >= 50.
   With tied or zero differences the interval is approximate.
6. **Bootstrap sizes.** The median MCSE uses B = 1000 and the HL MCSE B = 500. The HL MCSE is
   computed for the labelled comparisons (TFMs, side arms, ladder); the classical arms get the mean
   paired difference with its MCSE.
7. **The data of the detection curves.**
   - Columns: every arm as fitted (classical, TFMs and side arms), on the core and stress scenarios, at N_eval. The ladder columns are left out (they belong to Q5).
   - The bootstrap cluster is one scenario x replicate.
   - The rank of the tests is taken at each of the four true-ICI points.
8. **Floors and ceilings (6.3).**
   - A proportion at 0 or 1 is at a floor or a ceiling.
   - A Q4b point is uninformative when both curves' intervals lie within 0.02 of 1 (or of 0).
   - A true-ICI comparison is at the floor when both arms' lower 95 % limits are at or below the oracle's upper limit.
   - Such cells are labelled "uninformative (floor/ceiling)" and counted apart.
9. **Q5a, the best arm.** It is chosen by the mean true ICI on the replicates where the TFM's R2
   (m = 500) column is usable. The difference is paired on those replicates.
10. **Q5b.**
    - The rule uses Cox on the R0 predictions and EDGE (G = 10) on R2 (m = 500), both on the observed Y. The truth is the R2 true ICI.
    - The intervals come from a cluster bootstrap over scenario x replicate (B = 1000, percentile).
11. **Q6 for Mitra-v2.** The 40/48 and 30/48 cuts are applied as shares of the core scenarios
    inside the region where it ran. With the device rule n^2 p <= 8e7, all 48 core scenarios are inside.
12. **The 6.4 regions** use the margin 0.01 (log 1.1 for D_logb) at every pi, as 6.4 states. The
    0.3 factor of 7.0 applies to the core cells.
13. **The size band of Q4a.** The per-scenario band uses the MCSE at the nominal 0.05 with that
    scenario's R. The pooled size is the total of rejections over the total of null sets.
14. **Figure 2** draws every arm on replicates 1-R of the TFM schedule, so all the boxes describe the same data sets.
15. **The meta-model's optimizer.** The fit is `gam(..., family = scat(), method = "REML")` as
    pre-registered, run with `optimizer = "efs"`. This is mgcv's extended Fellner-Schall update of
    the same REML criterion.
    - The default Newton optimizer had not finished one response after 45 minutes on 1,000 synthetic designs, and EFS takes 7 minutes at 300 designs.
    - `--engine bam` (bam, fREML, discretised covariates) runs in 1 minute, but it is a different
      fit: its scat degrees of freedom stay at the lower bound of 3, and its edf differ. It is
      kept only for interim and test runs, and every output it writes says so.

## Outputs

`tables/` holds the tidy inputs, `tables/summary/` every summary, `tables/meta*/` the meta-model,
and `figures/out/` the PDFs and `font_report.csv`. `tables/cache/` holds the incremental caches;
deleting it forces a full re-read.
