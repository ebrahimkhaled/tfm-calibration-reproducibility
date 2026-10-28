# R layer: external goodness-of-fit tests on frozen predictions

This folder holds the R side of the GOF layer (plan section 4.5). It reads the stored per-replicate predictions and runs the external-validation tests of `ebrahim.gof` on every arm. Nothing is refitted: each arm's predictions are frozen and only checked.

| File | What it does |
|---|---|
| `gof_layer.R` | The battery for one `(y, p, X)` triple, the reader for `results/preds/`, and the parallel, resumable run over replicates. Output goes to `results/gof/<scenario>/<rep>.csv.gz` (gzip; older uncompressed `<rep>.csv` files are still read, and rewritten compressed when added to). |
| (other files, e.g. `bench_r_arms.R`) | Model fitting and benchmarking for the statistical arms; not part of the GOF layer. |
| `size_check.R` | The oracle size study: predictions `p*` (the true risks), outcomes `Y ~ Bernoulli(p*)`, 2,000 replicates on 12 scenarios at the plan's two evaluation sizes. Output: `results/gof_size/<scenario>/repNNNN.csv` and `results/size_check.csv`. |

## Package build

**Since 2026-10-07 (b): `ebrahim.gof` 2.10.0 at commit 19480aa** of the ebrahim.gof repository (source shipped as `third_party/ebrahim.gof-2.10.0-19480aa.tar.gz`). Against 2c77f3c (the first pin of the day, which added the `Osius-Rojek (external)` row), 19480aa changes one thing: the `Stukel (offset)` row of `run.all.external()` is now the score test of Stukel's two terms at the null, with the columns scaled to their information and a rank-aware inverse, and df = 1 when no prediction reaches 0.5 (see `../STUKEL_EXTERNAL_NOTE.md`, Resolution). `R/verify_gof_build.R` sources the package's R files at the commit (`git archive`, read only) and compares every function of the installed namespace with them: 189 functions, 0 differences at 19480aa (against 2c77f3c, `run.all.external` differs). The worktree is only read, never changed.

- `use_project_lib()` refuses to run in four cases:
  - `ebrahim.gof` is loaded from any library other than `rebuild/rlib`;
  - its version is not 2.10.0;
  - any of `run.all.external`, `localize.external`, `deepgof1.external` or `edge.gof` is missing;
  - **the build is not commit 19480aa**: the SHA-256 of every namespace function (formals and body, deparsed; `gof_code_hash()`) must equal `GOF_CODE_HASH`. A build of 2c77f3c, or of any later commit, has the same version number but another hash, and is refused.
- Every output row carries `gof_ref = 2.10.0@19480aa`.
- Never reinstall while a run is going: during an install the package folder is briefly missing.
- The other packages come from `rlib`: `givitiR` 1.3, `callr`, `data.table` 1.18.2.1, `digest`, `CompQuadForm`, and `arrow` for parquet.
- **Compressed csv.** `data.table` reads `.gz` files only through `R.utils`, which is not installed, so `fread_any()` decompresses with base R first.

Unit tests: `Rscript R/tests/test_gof_layer.R` (16 checks).

## The tests (one row each in the output)

All tests run on the evaluation sample of size `N_eval = max(1000, ceiling(100/pi))`: 3,334 rows at 3 % and 1,000 rows at 10 % and 30 % (plan 4.4–4.5). Each arm is tested twice:

- `ytype = real`: on the real outcomes Y;
- `ytype = ystar`: on an exact null `Y* ~ Bernoulli(p-hat)`. This gives the size of each test on that arm's own prediction distribution.

| Source | Rows |
|---|---|
| `run.all.external(y, p, G = 10, X, include_slow = TRUE)` | `EDGE (G=10)`; `EDGE (G=auto)` (the G used is in `note`); `Cox recalibration` (LR test of a = 0, b = 1; `note` gives a and b); `Calibration-in-the-large`; `Spiegelhalter-z`; `GiViTI` (external belt test, run by the package through `callr`); `Hosmer-Lemeshow (external)` (10 groups, chi-square with 10 df); `Osius-Rojek (external)`; `Stukel (offset)` (score test at the null, chi-square with 2 df, or 1 df when no prediction reaches 0.5; `note` says "score test"); `le Cessie (external)`. It also returns the descriptive rows O/E, calibration slope and AUC, which have no p-value. |
| `edge.gof(y = y, predicted_probs = p, external = TRUE)` | `edge.gof default (G = auto)` (role verdict) and `edge.gof ten groups (G = 10)` (role check). They are the same statistics as the battery's EDGE rows, and are kept as a cross-check of the two code paths. |
| `localize.external(y, p, X, M = 999, seed = ...)` | `Localize (any part named)`: `p_value` is the smallest adjusted p-value, so `p_value <= .05` means at least one part is named; `note` lists the parts named and the highest one. `Localize <PART> (named)`: the closed-testing adjusted p-value, with `note` set to "named" or "not named". `Localize <PART> (alone)`: the single-group p-value. `PART` is one of INTERCEPT, SLOPE, LINK and COV. |
| `deepgof1.external(y, p, X, B = 199)` | `DeepGOF-1 external (combined)`, plus the `(axes)` and `(allpairs)` components. |

Which arms get which test (`--localize plan`, `--deepgof plan`, the defaults):
- localization: oracle, FLIC, GAM and the TFMs (any arm whose name contains `flic`, `gam`, `tabpfn`, `mitra` or `tabicl`), in every scenario;
- DeepGOF-1: FLIC and the TFMs, on the real Y only.

Two kinds of arm get less:
- **Recalibration-ladder columns** (`p_<arm>__R{1,2,3}_m{100,500}`, lower-cased by the reader to `flic__r2_m500` and so on) get `run.all.external` and `edge.gof` on the real Y only. The test-guided rule of plan 4.6 needs the Cox and EDGE rows after each level, not their size.
- **The side arms** `tabpfn_legacy` and `tabpfn_t1` get no DeepGOF-1.

Both localization and DeepGOF-1 are exact Monte Carlo tests, so their size under the exact null holds by construction. Their size is checked once, in `size_check.R`. Pass `--deepgof-null` to run DeepGOF-1 on Y* too. Pass `all` or `none` to change which arms get these tests.

**GiViTI equals Cox in most replicates.** The belt test starts at polynomial degree 1 and keeps that degree unless a higher one is significant. At degree 1 its p-value is the Cox test's (as `run.all.external`'s help says). Count the replicates where the two differ before reporting GiViTI as a separate test.

## Input contract: `results/preds/`

The runner globs `results/preds/<scenario>/**/<rep>.{parquet,csv,csv.gz}`.
- The file name without its extension is the replicate id (for example `rep0001`).
- Files with the same id in one scenario are merged, so the Python TFM runner and the R classical arms may write separate files, in separate sub-folders, at different times.
- Paths that contain a folder or file starting with `_` are skipped.

Columns are matched case-insensitively.

| Column | Meaning |
|---|---|
| `y` | 0/1 outcome (required) |
| `p_star` (or `p_oracle`, `p_true`) | the true risk. When it is present, an `oracle` arm is added automatically. |
| `p_<arm>` | an arm's frozen prediction, for example `p_flic`, `p_tabpfn35`, `p_mitra2` |
| `ystar_<arm>` | optional stored exact-null outcome for that arm. If it is absent, Y* is drawn in R from a per-task seed, and `ysource` says which was used. |
| `x1 … xp` (or `x_*`) | the covariates of the evaluation rows, needed for localization, DeepGOF-1 and le Cessie. Without them those tests are skipped. A categorical covariate must be one-hot coded as numeric columns. |
| `row` | the record id; it is used to align files |
| `set` | optional. If present, only rows with `eval`, `evaluation` or `test` are used. |

A long format is also accepted: the columns `row, y, p_star, arm, p_hat[, y_star]`.

**The engine's files** (`code/README.md`, "Prediction file contract") are read as they are. The engine stores its own metadata in the parquet file (key `tfmcal`), and the reader uses two parts of it:
- `col_types$cat`: categorical columns are expanded into dummies, first level as reference. In T3, `x8` (6 levels) becomes `x8_1 … x8_5`, so that localization, DeepGOF-1 and le Cessie do not treat the code as a number.
- `fingerprints$eval`: the engine's SHA-256 of the evaluation set. It must agree across the files of one replicate, and it is stored as `engine_fingerprint`.

`--subset e50` or `--subset e200` runs the layer on the nested evaluation subsamples (`in_sub_e50`, `in_sub_e200`), which have about 50 and 200 expected events. The output goes to `results/gof_e50/` and `results/gof_e200/`. This gives test power against validation size without refitting (plan 4.4).

`--preds <dir>` and `--out <dir>` point the layer at another input or output folder, for example the smoke files:

```sh
Rscript R/gof_layer.R run --preds results/smoke/preds --out results/smoke/gof
```

Every file of one replicate must hold the same `row`, `y` and `p_star`. Otherwise the replicate fails, and the reason is written to `<rep>.error.txt` next to the output.

## Output: `results/gof/<scenario>/<rep>.csv.gz`

There is one long table per replicate, with one row per (arm, ytype, test). The columns are:
- `scenario, rep, n_eval, events`;
- `fingerprint`, `engine_fingerprint`, `eval_set` (`eval`, `e50` or `e200`) and `gof_ref` (`2.10.0@19480aa`);
- `test, family, statistic, df, p_value, note`;
- `secs`: wall time of the call, on its first row only;
- `arm`, `ytype` (`real` or `ystar`; not "null", which pandas reads as a missing value), `ysource`, `events_used`;
- `seed_base`, `pred_hash`.

**Degenerate fits stay visible.** Predictions at exactly 0 or 1 are moved to 1e-9 and 1 − 1e-9, and the count is appended to every `note` as `[k p clamped]`. A test that cannot run on such predictions keeps its row, with `Not run: …` or `Failed: …` in `note` and no p-value. In the smoke run, a GAM fit at n = 200 and 3 % had 3,326 of 3,334 predictions at the boundary. EDGE and localization then failed, and that was recorded instead of hidden.

## Seeds and fingerprints

- **Seeds.** Every stochastic step has its own seed, `task_seed(<what>|scenario|rep|arm|ytype|<test>)`, which is the xxhash32 of the string. Before each draw the generator is reset to Mersenne-Twister / Inversion / Rejection, so a package that calls `set.seed()` itself cannot leak into later steps. (`localize.external(seed = )` does this, and so does `deepgof1.external`'s tie-breaking.) Results are therefore the same whichever worker runs a task, and in whatever order.
- **Fingerprints.** `fingerprint` is the xxhash64 of the evaluation sample's 0/1 outcomes. Rounded floats are not used, because a float32 and a float64 copy of the same p\* can round differently. A thousand or more Bernoulli outcomes identify a data set. `Rscript R/gof_layer.R check` flags any two replicates of one scenario that share a fingerprint, which would mean a reset random stream upstream. If a replicate's outcomes change after its output was written, the whole file is kept as `<rep>.csv.stale_<time>` and the replicate is recomputed.
- **Prediction hashes.** `pred_hash` is the xxhash64 of an arm's predictions exactly as read. When an arm's predictions change (it was refitted, or a new checkpoint was used), only that arm's rows are moved to `<rep>.csv.stale_<time>` and redone; the other arms are kept. (The size-check outputs written before 23:00 on 2026-10-06 carry an older fingerprint, a hash of y, p\* and X computed in one double-precision process. It is just as unique, and is used only for the duplicate check.)

## Resume and parallel runs

- Outputs are written atomically: a temporary file, then a rename.
- A rerun computes only the (arm, ytype) pairs that are missing, so arms can land at different times (the TFMs after the classical arms) and the layer can run as a stream: `Rscript R/gof_layer.R run` again whenever new prediction files appear. A pair that is already done is not recomputed when options change; delete its rows (or the file) to redo it.
- The workers are PSOCK (`parallel::parLapplyLB`), with at most 12 of them. The number is lowered to the free cores minus 2, measured from the CPU load at start. BLAS and OpenMP run on one thread.
- The log is `logs/gof_layer.log` (and `logs/size_check.log`).

```sh
Rscript R/gof_layer.R run                          # every scenario under results/preds
Rscript R/gof_layer.R run --scenario S3_pi10_n500 --reps 1:50
Rscript R/gof_layer.R run --subset e50             # the ~50-event subsamples -> results/gof_e50
Rscript R/gof_layer.R run --B 99                   # cheaper DeepGOF-1 (the main run's setting)
Rscript R/gof_layer.R run --reps 201:500 --arms oracle   # oracle-only replicates (schedule P4)
Rscript R/gof_layer.R check                        # fingerprints, coverage, open errors
Rscript R/size_check.R run                         # oracle size study (resumes)
Rscript R/size_check.R summarize                   # -> results/size_check.csv
```

## Size check (`size_check.R`)

**Design.**
- Predictions are the true risks p\* and Y ~ Bernoulli(p\*).
- There are 12 scenarios: shapes S1–S4 × event rates 3 %, 10 % and 30 %, at N_eval = 3,334, 1,000 and 1,000.
- Each scenario has 2,000 replicates. The MCSE of a rate near .05 is .0049.
- DeepGOF-1 runs on S3 only, at both evaluation sizes (`S3_pi03`, N = 3,334; `S3_pi10`, N = 1,000), because it costs about ten times the rest of the battery.

**DGP.** The DGP follows plan 4.2:
- p = 8: six standard normals, with x1 and x2 correlated at 0.5, and two binaries;
- S2 adds two smooth non-linear terms; S3 adds two pairwise interactions; S4 uses the cloglog link;
- the linear part is scaled for an oracle AUC of 0.80 at 10 %;
- the intercepts are solved on 10^6 draws.

The parameters are in `results/size_check_dgp.csv`. Under the oracle the tests see only p\* and X, so the main run's Python DGP need not match these coefficients. The main run repeats the check in every scenario through the `oracle` arm (real Y), with 200 or 500 replicates.

**Storage.** The data are not stored. Each replicate is regenerated from `task_seed("size", scenario, rep, "data")`, and its fingerprint is in the output.

**`results/size_check.csv`.** It has one row per scenario × test, plus rows pooled over the scenarios at each N_eval. The columns are:
- `reps`, `not_run`, `rejections`;
- `rate` (at alpha = .05), `mcse`, and `lower`/`upper` (rate ± 1.96 MCSE);
- `mean_events`, `median_secs`;
- `flag`: "OUTSIDE [0.035, 0.065]" when the rate falls outside that interval (about ±3 MCSE);
- `fingerprint_duplicates`.

**Reading `Localize <PART> (named)` rows.** A part named by closed testing is not a level-α test on its own: under the global null it is named less often than α, by design. The size of localization is the familywise row `Localize (any part named)`. The `(alone)` rows are single-group p-values, and each of them should be at α.

**Reading the GiViTI row.** In the first 1,000 oracle replicates, GiViTI's p-value differed from the Cox test's in only 0–8 % of replicates per scenario. In the rest the belt stayed at degree 1, where the two tests are the same test.

## Size-check result (run 2026-10-06 22:58 to 2026-10-07 03:22, 24,000 replicates, 12 workers)

**Coverage.** No replicate failed, and no two replicates share a data fingerprint. GiViTI was "not run" in 5 of 24,000 replicates: its `callr` call timed out during a period when the machine was overloaded.

**Flags.** Four scenario rows fall outside [0.035, 0.065]. The largest per-scenario MCSE is 0.006.

| Test | Rates outside [0.035, 0.065] | Pooled rate, N = 1,000 | Pooled rate, N = 3,334 |
|---|---|---|---|
| **Stukel (offset)** | S1 10 % 0.074; S2 3 % 0.070; S3 3 % 0.076. S3 10 % (0.062) and S4 3 % / 10 % (0.059) are also high. | 0.056 (MCSE 0.0018) | 0.063 (MCSE 0.0027) |
| Localize SLOPE (alone) | S4 3 % 0.066 (MCSE 0.0055) | 0.049 | 0.056 |

**Stukel was liberal on frozen predictions (the likelihood-ratio form of 2c77f3c and earlier).** One of its two terms is non-zero only for records with logit p ≥ 0, which are rare at low event rates and with modest risk spread. The likelihood-ratio statistic is then not well approximated by chi-square with 2 df. **Resolved in 19480aa:** the row is now the score test, whose size is 0.035-0.051 in the same null design (`results/stukel_score_size.csv`; `../STUKEL_EXTERNAL_NOTE.md`, Resolution). The size study above was run with the old form; the main run's oracle nulls give the size of the new one.

**Hosmer–Lemeshow at 3 %** is high-ish (0.061–0.064) but inside the band.

**Every other test** holds its level at both sizes. This covers the EDGE rows (G = 10: 0.043–0.055; G = auto: 0.045–0.058), Cox, GiViTI, calibration-in-the-large, Spiegelhalter, le Cessie (0.048–0.062), localization's familywise rate (0.041–0.059), and DeepGOF-1 (0.046 and 0.049).

**Two traps met on this run.**
1. An install into `rlib` by another process while workers start makes them load the user library's 2.8.0. `use_project_lib()` stops the run instead of mixing builds.
2. Editing `size_check.R` while `Rscript` is running it makes R read a fragment of the new file after `main_size()` returns. The summary had already been written; the log ends with a harmless parse error. Do not edit a script while it runs.

## Costs measured on this machine (one call, sequential)

| Step | N = 1,000 | N = 3,334 |
|---|---|---|
| `run.all.external` with le Cessie and GiViTI (callr) | about 0.5 s | about 1.6 s |
| `localize.external`, M = 999 | about 0.15 s | about 0.5 s |
| `deepgof1.external`, B = 199, 8 covariates (28 pairs) | about 10 s | about 13 s |

With 12 workers on a shared machine, a call took 1.5–5 s for the battery and 15–30 s for DeepGOF-1. The 24,000-replicate size check took 4.4 h of wall time.

**Smoke run on the engine's files.** It covered 11 replicates over S1, S3, T1, T2 and T3, with 8 workers and other jobs running, and took 4.4 min. DeepGOF-1 took 37 s a call there (13 calls, 478 s in total), against 0.6 s for a battery call and 0.35 s for a localization call.

**Main-run estimate.** A replicate-scenario with every arm needs about:
- 44 battery calls (10 arms × 2 outcome types, plus 24 ladder columns), about 45 s;
- 14 localization calls, about 4 s;
- 4 DeepGOF-1 calls, 60–150 s.

For 58 × 200 replicate-scenarios this comes to 20–45 h of wall time on 12 workers. The classical-only replicates 201–500 add about 8 h. DeepGOF-1 is half or more of the total.

**Options to cut it.**
- `--B 99` halves DeepGOF-1.
- `--deepgof none` on a second pass, combined with `--reps 1:100` for DeepGOF-1, gives 100 replicates for that test.
- `--no-lecessie` saves little.

DeepGOF-1 is the bottleneck. The plan's budget assumed about 4 s per call. Under the default rule (FLIC + 3 TFMs, real Y only, 58 scenarios × 200 replicates) it needs about 46,000 calls, which is roughly 12–20 h of wall time on 12 workers. With B = 99 (`--B 99`) the cost halves. The rest of the main-run battery (10 arms × 2 outcome types × 11,600 replicate-scenarios, about 232,000 calls) takes about 10–15 h.

## Changes of 2026-10-07 (before the pre-registration freeze)

- **Package.** `ebrahim.gof` 2.10.0 (2c77f3c, superseded the same day by 19480aa: see below). The battery gains the `Osius-Rojek (external)` row.
- **Compressed output.** Output files are `<rep>.csv.gz`, and `.csv` files are still read. Stale copies are written as `<rep>.csv.gz.stale_<time>.gz`.
- **`--arms a,b,...`** runs only those arms (lower case). The main run uses `--arms oracle` on replicates R+1 to 500.
- **DeepGOF-1 in T2.** It uses `reading = "axes"` in scenario T2 (p = 30), where all 435 pairs cost about 230 s per call. The first DeepGOF row is then named `DeepGOF-1 external (axes reading)`, and the all-pairs row stays empty.
- **Temperature arms.** `tabpfn_t1` (TabPFN-3.5 at T 0.9) and `tabicl_t1` (TabICLv2 at T 1.0) are side arms. They get the battery and localization, but not DeepGOF-1.
- **Stukel.** Stukel (offset) was liberal at low event rates. See `../STUKEL_EXTERNAL_NOTE.md` and `stukel_diagnosis.R`.

## Changes of 2026-10-07 (b), before the freeze

- **Package pin: commit 19480aa.** `GOF_REF = 2.10.0@19480aa`; `GOF_CODE_HASH` holds the code hash of that build, and `use_project_lib()` refuses any other build (unit test). Check a fresh install with `Rscript R/verify_gof_build.R`.
- **Stukel (offset) is the score test** (19480aa). Its size in the null design of `stukel_diagnosis.R` is in `results/stukel_score_size.csv` (`R/stukel_score_size.R`, 4,000 nulls per cell). It is ranked with the other tests (pre-registration 5.3, C21).
- **DeepGOF-1 reading by size.** `dg_reading_for()`: the axes reading when the design has more than 12 predictors (scenario T2 and the random designs with p > 12), else the combined reading. In the core design this is the T2 rule, unchanged.
- **Random designs (family R).** The layer reads them like any scenario (`results/preds/R0001/...`). Their categorical predictors are expanded to dummies from `col_types$cat`, as in T3.
- **The audit recipe** `audit_calibration.R` (deliverable b of pre-registration 14): `run.all.external()` + `localize.external()` on any model's (y, p[, X]); `code/audit_calibration.py` calls it from Python.
