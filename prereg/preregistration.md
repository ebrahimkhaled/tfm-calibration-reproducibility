---
title: "Calibration of tabular foundation models versus logistic regression for clinical risk prediction: pre-registration of a simulation study and a real-data analysis"
author:
  - Ebrahim Khaled Ebrahim
affiliation: "Department of Applied Statistics, Faculty of Business, Alexandria University, Alexandria, Egypt"
version: "1.0 (final text for deposit)"
date: "2026-10-08"
---

**Status.** Version 1.0, the final text for deposit. The protocol is frozen at the moment it is registered on OSF Registries. The registration must be made before the first replicate of the main simulation run is written to disk; the run scripts refuse to start until the registration is recorded (Section 9.4). After the registration this file is never edited. Every later change goes into `DEVIATIONS.md`, dated, with its reason (Section 12).

**Author.** Ebrahim Khaled Ebrahim (sole author), ebrahimkhaled@alexu.edu.eg, ORCID 0009-0006-7839-8778.

**Affiliation.** Department of Applied Statistics, Faculty of Business, Alexandria University, Alexandria, Egypt.

**Structure.** The simulation part follows the ADEMP structure of Morris, White and Crowther (2019): Aims, Data-generating mechanisms, Estimands, Methods, Performance measures. The registration uses the OSF Registries "Simulation Studies" template, which follows the same structure (Siepe et al. 2024).

---

## 0. What was done and seen before this registration

### 0.1 Work before the freeze

Before this text was frozen, the following work was done. All of it used data that the main run never uses.

1. **An earlier, unpublished pilot by the same author.** It used 3 replicates per scenario and small test sets. It is superseded, and no result from it is used.
2. **An oracle size study of the external tests** (`R/size_check.R`).
   - 2,000 replicates in each of 12 scenarios, with predictions equal to the true risk.
   - It used its own coefficients and its own random stream, not the main design's.
   - It is an auxiliary check (Section 5.3).
3. **A Phase-1 pilot of this study** (2026-10-07):
   - every model arm on 6 scenarios x 20 replicates, then the full goodness-of-fit layer;
   - TabPFN-3.5 on 5 scenarios x 20 replicates;
   - timing and memory runs.
   - The pilot **computed and inspected calibration summaries**: true slope and intercept, true ICI, and the rejection rates of the tests. They informed the changes listed in Section 0.2.
4. **A read-only study of the external Stukel test** (Section 5.3), with its own random stream.
5. **A final smoke run** of every arm, the scoring and the test layer: 2 scenarios x 2 replicates (Section 9.2).
6. **A size study of the new Stukel row** of `ebrahim.gof` 2.10.0 at commit 19480aa (Section 5.3): 4,000 nulls per cell, in the null design and with the seeds of item 4.
7. **Two engineering checks for the random-design tier** (Section 3.9), on the smoke stream:
   - a GPU memory and time probe of the three TFMs at 16 (n, p, predicted rows) cells on synthetic data, which sets the device rules;
   - the classical arms and the test layer on 50 random designs (R0006-R0055), which give the per-fit costs used to choose D.
   - Neither check computed or inspected a calibration summary.
8. **A second smoke run** (Section 9.2): every arm on 2 core scenarios x 2 replicates and on 5 random designs (R0001-R0005), then the scoring and the test layer at commit 19480aa.

**Separate seed streams.**

- The pilot and the smoke runs drew every data set from `SeedSequence([scenario_id, n_index, rep, 777])`.
- The main run draws from `SeedSequence([scenario_id, n_index, rep])`.
- The random designs (Section 3.9) draw from a stream of their own, `SeedSequence([4242, 2, d, rep])`; their smoke data add 777.
- The main run never touches the 777 stream.
- Every main-run data set carries a SHA-256 fingerprint, which is asserted before every downstream step (Section 3.8). A pilot file therefore cannot enter the main results.

### 0.2 Changes made before the freeze, and why

Each row was decided by the author on 2026-10-07, after the pilot. Rows marked "pilot" were informed by the pilot's calibration summaries. They are design decisions made before the main run, not deviations.

On 2026-10-08, before the freeze, the author changed the draft's confirmatory hypotheses into an estimation-focused protocol with pre-specified research questions and fixed reporting rules (Section 7), by his own choice to describe and estimate rather than test predictions, so that the results determine the conclusions within rules fixed in advance.

| # | Item | Before | After (this registration) | Why |
|---|---|---|---|---|
| C1 | Slope at small n (question Q1) | The plan of 2026-10-06 expected the median true slope b\* of each TFM to exceed 1.1 at n <= 500 and pi = 10 % in all four shapes (TFMs under-confident) | No expected direction. Q1 estimates each TFM's b\* and its paired distance from 1 against FLIC on the log scale, with the main cells at n <= 500 and pi = 10 %, and labels the direction it finds by a fixed rule (Section 7.1) | Pilot. In the two qualifying cells (S2 and T3 at pi = 10 %, n = 500), the TFM medians were 0.88-1.11 (MCSE 0.05-0.09), against FLIC's 0.84-0.86. At n = 200, pi = 3 %, every classical arm was over-confident (FLIC 0.49) and the TFMs were less so (0.71-1.00). These values informed the emphasis of Q1: a paired comparison with FLIC, with both directions labelled, in place of a single threshold of 1.1. |
| C2 | Rare events (question Q3) | The plan of 2026-10-06 expected each TFM to have a larger \|a\*\| than FLIC at pi = 3 % at every n | No expected direction. Q3 estimates the paired \|a\*\| difference against FLIC at pi = 3 % at every n, shows the cells at n >= 1000 first and each TFM's shipped softmax temperature beside its result, and the temperature arms estimate the temperature's effect directly (Section 7.3) | Pilot. The sign depended on the temperature. At S4_pi03_n3000, TabICLv2 and TabPFN v2 (both T = 0.9) had a\* = 0.29-0.32, against FLIC 0.02, Mitra-v2 0.00 and TabPFN-3.5 0.02. At n = 200, no TFM had a larger \|a\*\| than FLIC. These values informed the emphasis of Q3 on n >= 1000 and on the temperature. |
| C3 | Primary TabPFN arm | "TabPFN-3.5 if its licence is accepted, else the newest ungated checkpoint" | TabPFN-3.5 standard checkpoint, pinned by version and SHA-256 (Section 5.2) | The author accepted the licence, and the checkpoints are in place. The choice between the standard and Fast checkpoints is justified by timing in Section 5.2. |
| C4 | Temperature arms | TabPFN at T = 1.0 on 8 scenarios, R = 200 | `tabpfn_t1` = TabPFN-3.5 at T = 0.9. A new arm `tabicl_t1` = TabICLv2 at T = 1.0. Both on the same 8 scenarios with R = 100. | Pilot. TabPFN-3.5 already ships at T = 1.0, so the old arm would repeat the main arm. TabICLv2 and TabPFN v2 ship at T = 0.9, which multiplies every logit by 1/0.9 (measured slope 1.1105-1.1109) and lowers the mean risk by 0.012-0.017. |
| C5 | LightGBM + Platt | Platt scaling as fitted | A Platt slope <= 0 keeps the raw LightGBM scores (flag `platt_fallback`) | Pilot. At n = 200, pi = 3 %, a negative Platt slope reversed the ranking in 3 of 20 replicates. |
| C6 | Ridge-CV lambda path | glmnet's default path | 200 log-spaced values from glmnet's lambda_max down to 1e-6 x lambda_max, with the early path stop switched off; an optimum at either end is flagged | Pilot. The default path stopped early, and the CV optimum sat at its lower end in 19 of 20 fits at S3_pi30_n1000. With the new path it was interior in 49 of 50 smoke fits; the one exception is at the upper end (the null model), and it is flagged. |
| C7 | Degenerate fits | Scored as "ok" | Labelled "degenerate", scored, kept and counted (Section 8.2) | Pilot. At n = 200, pi = 3 %, more than 100 predictions sat at 0 or 1 in 12 of 20 GAM fits and 8 of 20 LR-MLE fits. |
| C8 | Mitra-v2 at T2, n = 3000 | Run like every scenario | Recorded as "does not fit in 6 GB GPU memory". It is not run there, and its context is never cut (Section 5.2). | Pilot. 20 of 20 fits ran out of memory. A CPU fit had not finished 100 rows after 15 minutes. |
| C9 | Device of TabPFN-3.5 | GPU, or CPU workers | Whole scenario groups run on CPU workers, declared in Section 9.3. T2_pi10_n3000 always runs on the CPU. | Pilot. TabPFN-3.5 runs out of GPU memory at T2, n = 3000. CPU and GPU predictions agree to 2.6e-4. The schedule needs the CPU workers to finish in about 3 days. |
| C10 | Replicates and test settings | TFMs R = 200 everywhere; legacy V2 R = 200; DeepGOF-1 B = 199 on real Y and Y\*; the battery on every classical replicate | Schedule P4 (Section 6.3): TFMs R = 200 at n <= 1000 and R = 150 at n = 3000; classical R = 500; legacy V2 R = 100; DeepGOF-1 B = 99 on real Y only, with the axes reading only in T2; the battery on every arm for reps 1-R, and on the oracle only for reps R+1 to 500 | Compute budget measured in the pilot: the plan as written needed 4.0-4.7 days |
| C11 | Stukel (offset) | One of the battery tests | Its size is reported beside its power, as for every test. The exclusion from the detection ranking decided earlier the same day is withdrawn by C21: the row is now the score test, and it is ranked like the other tests (Section 5.3). | The size study: 0.063 pooled at N = 3,334. The Stukel study traced this to the likelihood-ratio statistic of its upper-tail term, which the score form replaces. |
| C12 | The size study (review D5) | The 2,000-replicate oracle study was the size study | The main size study is the main run's own oracle nulls; the earlier study is auxiliary (Section 5.3) | The earlier study used its own coefficients |
| C13 | True ICI (review D6) | One loess estimator | The pre-registered loess estimator stays primary. `true_ll_ici` is reported beside it, and the oracle row gives the loess floor (Section 4.1). | The loess of p\* on logit p-hat cannot follow the logistic curve: the oracle gets 0.0008-0.0029, while `true_ll_ici` is 0 for it |
| C14 | Class thresholds at pi = 3 % (review D6) | Absolute thresholds | The probability-scale thresholds are multiplied by 0.3 at pi = 3 % (Section 4.5) | At a mean risk of 0.03 the absolute thresholds were met by arms that failed "weak" |
| C15 | Few-event failures (review D8) | General pairing rule | The pairing rule names the known few-event failures (Section 8.2) | Ridge needs 3 events and LightGBM 2; the m = 100 ladder can draw 0 events |
| C16 | Mitra categoricals (review D9) | "Declared categorical to every TFM" | Mitra-v2 receives integer codes (Section 5.1) | Its interface has no categorical argument |
| C17 | e200 subsample (review D13) | Every event rate | pi = 30 % only (Section 5.3) | It needs 200/pi rows: more than N_eval at 3 % and 10 % |
| C18 | Test package | ebrahim.gof 2.9.0 (+ localization from 2.10.0) | ebrahim.gof 2.10.0, which adds an "Osius-Rojek (external)" row to `run.all.external()` | A release-branch build that has every function used |
| C19 | LightGBM software | R `lightgbm` | Python LightGBM 4.7.0, the same library, with the same grid and early-stopping rule | R `lightgbm` is not in the project library |
| C20 | GOF output files | csv | gzip-compressed csv (`.csv.gz`); the reader handles both | Disk use |
| C21 | Package commit and the Stukel (offset) row | `ebrahim.gof` 2.10.0 at commit 2c77f3c. Stukel (offset) is the likelihood-ratio test, liberal at low event rates (0.061-0.078). It is excluded from the detection ranking: an exception in the size rule, and left out of the detection comparisons, the detection curves and the ranking in the summary tables. | 2.10.0 at commit **19480aa**. Stukel (offset) is the **score test** of the same two terms at the null (columns scaled to their information, rank-aware inverse, df = 1 when no prediction reaches 0.5). Its size is 0.035-0.053 in 4,000 nulls per cell (pi = 3, 10 and 30 %; normal and t3 risks), with a p-value in every replicate (Section 5.3). Stukel is in the size question (Q4a) with no exception, and in the detection curves and the summary tables, ranked like every other test. The test layer refuses any build but 19480aa (Section 5.3). | The author changed the package on 2026-10-07 after the Stukel study. The likelihood ratio was liberal (0.061-0.078); the score form holds its size. |
| C22 | Random-design robustness tier | None: 58 fixed scenarios | A robustness tier of D = 1,000 randomly drawn designs, one replicate each, with a pre-specified meta-model and a transfer check (Sections 3.9, 6.4 and 7.6) | The conclusions of a fixed factorial can be challenged as depending on the chosen settings. The tier draws every setting at random and checks whether the conclusions carry over. |
| C23 | Outputs for reuse | Per-replicate predictions deposited | (a) the per-replicate predictions and test results released as an open benchmark, with a loader and a data dictionary; (b) a calibration audit in one call, in R and from Python; (c) a minimum-n-for-calibration table by event rate for every model, as a primary output; (d) a TRIPOD+AI alignment table. TabPFN weights are never archived (Section 14). | Reuse by practitioners and methodologists |

---

## 1. Background and rationale

Tabular foundation models (TFMs) such as TabPFN, Mitra and TabICL predict a binary outcome in context. They read the development sample as a prompt and return a probability for each new row, with no gradient fitting on the user's data. They are being tried for clinical risk prediction. There, calibration (agreement between predicted and actual risk) decides whether a prediction can be used for a clinical decision.

Recent studies report TFM calibration with the Brier score, log-loss, expected calibration error, conformal coverage or a calibration plot read by eye. None of them does the following five things, and this study does:

1. score TFM predictions against the true risk;
2. check the size of the external-validation goodness-of-fit tests on the TFMs' own prediction distributions;
3. localize the miscalibration in the risk range or in covariate subgroups;
4. place TFMs on the calibration hierarchy of Van Calster et al. (2016) and on the sample-size criteria of Riley et al. (2020);
5. evaluate a test-guided recalibration rule.

Two further parts make the results easier to trust and to reuse. A tier of randomly drawn designs checks whether the conclusions depend on the chosen scenarios (Section 3.9). The predictions, a minimum-n table and a one-call calibration audit are released for others to use (Section 14).

**Prior information, disclosed.** Section 0 lists everything that was run and seen before this registration, and which design choices it informed. The pilot also informed the emphasis of research questions Q1 and Q3 (Section 7). The protocol predicts the direction of no result.

---

## 2. Aims

**A1.** Quantify how far TFM risk predictions are from the true risk, as a function of development sample size and event rate. The comparators are Firth logistic regression with intercept correction (FLIC), maximum-likelihood logistic regression, ridge regression, a generalized additive model and LightGBM.

**A2.** Find, for each model arm, the smallest development size at which it reaches pre-specified calibration targets ("minimum n for calibration"), and compare it with Riley's minimum sample size.

**A3.** Measure the size (type I error) and the detection rate of the external-validation goodness-of-fit tests when they are applied to each arm's frozen predictions.

**A4.** Measure how often the localization procedure names the correct source of miscalibration when a known covariate interaction is missed.

**A5.** Measure how much a small local sample repairs calibration (a recalibration ladder), and whether a test-guided rule picks the right level.

**A6.** Check on real clinical data (Section 10) whether the patterns found in A1-A5 appear in real patients.

**A7.** Check whether the conclusions of A1 and A3 hold across the design space, on randomly drawn designs (Section 3.9), and map the regions where each TFM is better or worse calibrated than FLIC.

**A8.** Give practitioners and methodologists outputs they can reuse (Section 14):

- a minimum-n-for-calibration table by event rate for each model (a primary output, Section 11);
- a calibration audit in one call for any model's predictions;
- the full per-replicate predictions and test results as an open benchmark;
- an alignment of the reporting with TRIPOD+AI.

---

## 3. Data-generating mechanisms (simulation)

### 3.1 Covariates

The core design has eight covariates:

- X1, ..., X6 continuous, marginally N(0, 1); corr(X1, X2) = 0.5; all other pairs independent;
- X7 ~ Bernoulli(0.5);
- X8 ~ Bernoulli(0.3).

### 3.2 Outcome models (shapes)

Let `L = 1.0 X1 + 0.8 X2 + 0.6 X3 + 0.4 X4 + 0.2 X5 + 0.0 X6 + 0.5 X7 + 0.3 X8` be the base linear term (X6 is a pure noise covariate). Then:

| Shape | True linear predictor eta | Link | Correctly specified arms |
|---|---|---|---|
| S1 linear | alpha + kappa L | logit | LR-MLE, FLIC, Ridge, GAM |
| S2 smooth non-linear | alpha + kappa [L + gamma ((X1^2 - 1)/sqrt(2) + (X3^3 - 3 X3)/sqrt(6))], gamma = 0.5 | logit | GAM |
| S3 interactions | alpha + kappa [L + 0.5 X1 X7 + 0.5 X3 X4] | logit | none of the statistical arms as fitted (main effects only) |
| S4 asymmetric link | alpha + kappa L | complementary log-log: p\* = 1 - exp(-exp(eta)) | none |

The true risk of a row is p\* = logit^{-1}(eta) in S1-S3, and 1 - exp(-exp(eta)) in S4. The outcome is Y ~ Bernoulli(p\*).

In S3, the interaction X1 x X7 defines the *true miscalibrated subgroup*. Patients with X7 = 1 have a steeper risk gradient in X1 than any main-effects model can represent. This subgroup is the target of the localization analysis (Sections 5.4 and 11).

These formulas are implemented in `code/dgp.py`. An independent re-implementation from this text agreed with it to 5.6e-16 on all 58 scenarios.

### 3.3 Signal strength and event rate

- For each shape (and each severity gamma in the stress set), the scale factor kappa is solved numerically on 10^6 fixed draws, so that the oracle AUC (the AUC of p\*) is 0.80 at an event rate of 10 %.
- For each shape and each event rate pi in {3 %, 10 %, 30 %}, the intercept alpha is then solved on the same draws so that E[p\*] = pi.
- The realized prevalence equals pi to 2e-15 in every scenario.
- Because kappa is fixed at 10 %, the oracle AUC is 0.77-0.83 at 3 % and 30 %.
- The solved values and the realized oracle AUC of every scenario are listed in Appendix A.

### 3.4 Development sample size

n in {200, 500, 1000, 3000}. Each scenario is also reported in two ways:

- as events per variable (EPV = n pi / p);
- against Riley et al.'s (2020) minimum sample size for a binary outcome, computed with the R package `pmsampsize` (expected shrinkage 0.9, Cox-Snell R^2 from the oracle AUC, p parameters).

The scenario n = 200 at pi = 3 % (about 6 events, EPV 0.75) is kept on purpose: it is the regime where TFMs might help. Degenerate fits there are an outcome, not a reason to drop the scenario.

### 3.5 Core factorial

4 shapes x 3 event rates x 4 sizes = **48 core scenarios** (ids 1-48).

### 3.6 Stress set

At pi = 10 % and n in {500, 3000}, five stress types (10 scenarios, ids 49-58):

| ID | Change from the core design | Shape |
|---|---|---|
| T1 | X1-X6 drawn as Student t with 3 df through a Gaussian copula (X1 and X2 keep correlation 0.5), scaled to unit variance | S1 |
| T2 | p = 30: the 8 core covariates plus 22 independent N(0, 1) noise columns | S1 |
| T3 | X8 replaced by a 6-level categorical with equal level probabilities and level effects (0, 0.3, -0.3, 0.6, -0.6, 0.9) inside L. It is declared categorical to TabPFN and TabICL, given as integer codes to Mitra-v2, and dummy-coded for the statistical arms. | S3 |
| T4a | S2 with severity gamma = 0.25 | S2 |
| T4b | S2 with severity gamma = 1.0 | S2 |

Together with S2 at gamma = 0.5, T4a and T4b form a severity ladder for detection-power curves.

**Total: 58 scenarios.** Each has a fixed integer `scenario_id` (Appendix A).

### 3.7 Data drawn per replicate

Each replicate draws the following sets, independently and fresh:

| Set | Size | Use |
|---|---|---|
| Development | n | fitting, or the context of a TFM |
| Evaluation | N_eval = max(1000, ceil(100/pi)): 3,334 at 3 %, 1,000 at 10 % and 30 % | oracle estimands (p\* known) and the test battery on fresh Y; at least 100 expected events (Riley et al. 2021) |
| Recalibration | 500 | recalibration ladder; the m = 100 level uses the first 100 rows |

A **probe set** of 500 rows per scenario is drawn once, from `SeedSequence([scenario_id, 99999])`, and is the same in every replicate. It is used for the prediction-instability plots.

Smaller validation samples are nested subsamples of the stored evaluation rows (the first ceil(50/pi) and ceil(200/pi) rows); no model is refitted. The e50 subsample exists at every event rate. The e200 subsample exists only at pi = 30 %, because at 3 % and 10 % it would need more rows than N_eval.

### 3.8 Random numbers and fingerprints

- **Data.**
  - Every data set is drawn in Python from `numpy.random.SeedSequence([scenario_id, n_index, rep])`, with rep >= 1.
  - The seed is spawned into independent child streams for the development, evaluation and recalibration draws.
  - Model seeds and null-outcome seeds come from the same entropy with fixed arm codes.
  - The pilot and smoke stream adds a fourth entry, 777 (Section 0.1).
- **Fingerprints.**
  - A SHA-256 fingerprint of the bytes of (X, p\*, Y) of every set is stored with the replicate.
  - Every downstream step recomputes and asserts it before using the data: adding an arm, scoring, and the R layer's cross-file checks. A mismatch stops the run.
  - Two replicates with identical data in one run stop the run, which would catch a hidden reset of the random stream.
- **R procedures.**
  - Procedures that draw random numbers (the Monte Carlo references of `localize.external()` and `deepgof1.external()`, glmnet fold assignment, null outcomes drawn in R) receive an explicit integer seed. The seed is derived from a string naming the scenario, replicate, arm and test.
  - The generator is reset to Mersenne-Twister before each such call.
  - Equal outcome fingerprints across replicates of a scenario are reported as an error.

### 3.9 Random-design robustness tier

**Purpose.** The 58 scenarios fix every factor at a few levels. A reader may ask whether the conclusions hold away from those levels. In this tier every design draws its own setting, so that the conclusions can be checked across the whole design space, and the regions where they fail can be mapped (Sections 6.4 and 7.6).

**Design.** D = 1,000 designs (`R0001`-`R1000`), each with **one replicate**. Design d draws its setting from `SeedSequence([4242, 0, d])`:

| Factor | Distribution |
|---|---|
| event rate pi | log-uniform on [0.02, 0.40] |
| development size n | log-uniform on [150, 4000], rounded |
| number of predictors p | integer uniform on [4, 30] |
| predictor correlation rho | uniform on [0, 0.6] |
| oracle AUC target | uniform on [0.65, 0.90] |
| severity of non-linearity w_nl, of interaction w_int, of link asymmetry w_link | each uniform on [0, 1] |
| fraction of binary or categorical predictors | uniform on [0, 0.5] |

**Covariates.**

- A latent Gaussian vector with exchangeable correlation rho.
- Of the p predictors, round(fraction x p) are discrete and the rest are continuous (the latent values).
- Each discrete predictor is categorical with probability 0.25 (3-6 levels, equal probabilities, cut at normal quantiles of its latent value); otherwise it is binary, 1{z > q}, with a prevalence drawn on [0.2, 0.5].
- Categorical predictors are declared to the TFMs and coded for the statistical arms as in T3 (Section 5.1).

**Outcome.** The shape is a random mixture built from the core design's own shape generators (`code/dgp.py`):

- The linear term L has coefficients drawn per design: a random sign, a size uniform on [0.2, 1], and each predictor after the second is pure noise with probability 0.25; categorical levels get N(0, 0.5^2) effects. L is rescaled to the SD of the core design's L (1.755), so that the weights below mean what gamma means in the core design.
- N is S2's non-linear term at gamma = 1, and I is S3's interaction term (Section 3.2), both evaluated by `dgp.shape_score` on the design's predictors in the core roles: x1 = the first continuous predictor, x3 = the second, x4 = the third (the first if there are only two), x7 = the first binary predictor (1{z > 0} of the last continuous predictor if there is none).
- eta = alpha + kappa [L + w_nl N + w_int I], and p\* = (1 - w_link) logit^{-1}(eta) + w_link (1 - exp(-exp(eta))): w_link = 1 is S4's complementary log-log link.
- kappa is solved so that the oracle AUC equals the design's target at its own event rate, and alpha so that E[p\*] = pi, on 200,000 calibration draws from `SeedSequence([4242, 1, d])`. Every design was solved: the realized prevalence equals pi to 6e-14, and the oracle AUC equals its target to 2e-10.

**Data per design.**

- Development set of size n.
- Evaluation set sized as in the core design: N_eval = max(1000, ceil(100/pi)), so at least 100 expected events (1,000-5,000 rows).
- No recalibration or probe set: there is one replicate and no recalibration ladder.
- The data come from `SeedSequence([4242, 2, d, 1])`, a stream that the core design and the pilot never use, and they carry the same SHA-256 fingerprints as the core design (Section 3.8).

**Arms.** Oracle, LR-MLE, FLIC, Ridge-CV, GAM, LightGBM + Platt (with its raw column), **TabPFN-3.5 (the main arm)**, TabICLv2 and Mitra-v2, with the settings of Section 5.2. There are no legacy or temperature arms and no recalibration ladder.

**Device rules** (measured on the 6 GB card with the run's 0.8 memory cap: pilot Section 2 and `results/random_memory_probe.csv`):

- **TabPFN-3.5** holds the development rows and the predicted rows together, so its size is (n + N_eval) x p, not n x p alone: a 1,500 x 20 design with 5,000 predicted rows ran out of memory, while 4,000 x 8 with 2,000 predicted rows fitted. Every measured cell up to 84,000 fitted (peak 4.4 GB), and every cell from 99,000 up ran out of memory. A design runs TabPFN-3.5 on the GPU when (n + N_eval) x p <= 80,000, and on a CPU worker otherwise (179 of the 1,000 designs). The device is recorded per design; CPU and GPU predictions agree to 2.6e-4 (Section 5.2).
- **Mitra-v2** predicts in batches, and its memory grows with n^2 x p: every measured cell up to 8.0e7 fitted (2,000 x 20 at 4.1 GB, 3,000 x 8 at 3.7 GB), and every cell from 9.4e7 up failed (2,500 x 15, 3,500 x 8, 4,000 x 6, and T2 at 3,000 x 30). Above n^2 x p = 8.0e7 Mitra-v2 is **recorded as "does not fit in 6 GB GPU memory"**, by the rule used for T2_pi10_n3000 (170 of the 1,000 designs). It is not run there, and its context is never cut. Its meta-model covers only the designs where it ran, and that region is reported.

**Test layer.** As in the core design (Section 5.3), at B = 99, with one change of reading: DeepGOF-1 uses the axes reading when the design has more than 12 predictors, and the combined reading otherwise. In the core design this rule picks out exactly T2.

**Files.** The solved designs are in `code/random_designs.csv` (`python code/dgp_random.py build --D 1000`; `check` re-solves them and compares). Appendix B summarizes them.

---

## 4. Estimands

All oracle estimands are computed on the evaluation set of each replicate, where the true risk p\* of every row is known. Predictions p-hat are clipped to [1e-6, 1 - 1e-6] for every computation. The number of predictions within 1e-6 of 0 or 1 is recorded per arm and replicate.

### 4.1 Primary estimands (oracle)

1. **True calibration slope b\*.** The slope of a quasi-binomial generalized linear model of p\* on logit(p-hat), with a logit link.
   - b\* > 1 means under-confidence (predictions too close to the mean).
   - b\* < 1 means over-confidence.
2. **True calibration intercept a\*.** The intercept of a quasi-binomial model of p\* with offset logit(p-hat) and no slope (calibration-in-the-large on the logit scale).
3. **True integrated calibration index (true ICI).** The mean over evaluation rows of |p-hat - c(p-hat)|.
   - c(.) is a loess fit (degree 2, span 0.75, Gaussian family, no robustness iterations, exact fit as R's `surface = "direct"`) of **p\*** (not Y) on logit(p-hat), evaluated at the evaluation rows.
   - This loess cannot follow the logistic curve exactly, so the estimator has a floor: 0.0008-0.0029 for the oracle in the pilot.
   - The oracle row is reported in every table as that floor.
   - The **log-scale version `true_ll_ici`** is reported beside it. It smooths logit p\* on logit p-hat and maps back. It is exactly 0 for the oracle, and it equals the OCE whenever logit p\* is linear in logit p-hat.
   - The true ICI remains the primary estimand.

### 4.2 Further oracle estimands (secondary)

- True E50, E90 and E_max: the median, 90th percentile and maximum of |p-hat - c(p-hat)|.
- **Oracle calibration error (OCE)** = mean |p-hat - p\*|. It is the individual-level error. It separates "miscalibrated" from "calibrated on average but poorly resolved".
- Mean p-hat minus mean p\* (calibration-in-the-large on the probability scale).
- True discrimination: the expected AUC of p-hat against p\*, and the oracle AUC.

### 4.3 Empirical estimands (secondary)

These are computed with the observed Y on the evaluation set, as a real validation study would:

- empirical ICI, E50, E90 and E_max (Austin and Steyerberg 2019: loess of Y on logit(p-hat), degree 2, no robustness iterations);
- empirical calibration slope (joint fit of Y on logit(p-hat)) and calibration-in-the-large (offset fit);
- Brier score, log-loss and AUC;
- 10-bin expected calibration error (reported only for comparison with the machine-learning literature);
- the gap between empirical and true ICI per replicate (how wrong a typical validation estimate is).

### 4.4 Prediction instability (secondary)

On the fixed probe set: the per-patient spread of p-hat across replicates (standard deviation and 2.5-97.5 % range) (Riley and Collins 2023).

### 4.5 Calibration class of each replicate

| Class | Rule (pi = 10 % and 30 %) | Rule at pi = 3 % |
|---|---|---|
| mean | abs(mean p-hat - mean p\*) <= 0.01 | <= 0.003 |
| weak | abs(a\*) <= 0.1 and abs(b\* - 1) <= 0.1 | the same |
| moderate | true ICI <= 0.015 | <= 0.0045 |
| strong-approx | true E90 <= 0.03 | <= 0.009 |

- **Relative thresholds at 3 %.** At pi = 3 % the three probability-scale thresholds are multiplied by pi / 0.10 = 0.3, because the absolute values are easy to meet when the mean risk is 0.03. The weak rule is on the logit scale and is not changed.
- **Nesting.** The classes are nested, as in the hierarchy of Van Calster et al. (2016). An (arm, replicate) is placed in the highest class whose rule holds *together with every lower rule*. Each rule is also stored on its own.
- **Sensitivity.** The class-based results are repeated at half and at double every threshold, and with `true_ll_ici` in place of the true ICI.

**Minimum n for calibration** of an arm in a (shape, pi, class) is the smallest n in {200, 500, 1000, 3000} at which the arm meets the class in at least 80 % of replicates. If no n qualifies, the value is "> 3000".

---

## 5. Methods (model arms)

### 5.1 Common rules

- **Paired design.** Every arm is fitted on the *same* development set of a replicate. It predicts the same evaluation, recalibration and probe rows.
- **No tuning on evaluation data.** No arm is tuned on evaluation or recalibration data. The TFMs are used with their shipped settings and are not tuned at all. The only exception is the softmax temperature of the two temperature arms.
- **Categorical covariates.** The binary covariates and the 6-level factor of T3 are declared categorical to TabPFN (`categorical_features_indices`) and to TabICLv2 (pandas categories). **Mitra-v2 receives them as integer codes**, because its interface has no categorical argument; this is how AutoGluon itself passes categoricals to Mitra.
- **Provenance.** Every version, checkpoint file, revision hash and SHA-256 is recorded per replicate and in `results/environment.json` (Section 13).

### 5.2 Arms

| Arm (code name) | Implementation | Role | Settings |
|---|---|---|---|
| Oracle (`oracle`) | p\* itself | defines the null and the truth | the "perfect model" row; the main size study (Section 5.3) |
| LR-MLE (`lr`) | R `stats::glm`, binomial, logit, main effects | classical reference | separation is flagged, not hidden |
| **FLIC** (`flic`) | R `logistf::logistf(..., flic = TRUE, pl = FALSE)` (Puhr et al. 2017), main effects | **primary statistical reference** | package defaults |
| Ridge-CV (`ridge`) | R `glmnet`, alpha = 0, lambda = `lambda.min` by 10-fold CV deviance, folds stratified on Y | penalization arm | **lambda path:** 200 log-spaced values from glmnet's own lambda_max down to 1e-6 x lambda_max, with glmnet's early path stop (`fdev`) switched off for this fit; `lambda_at_edge` flags an optimum at either end |
| GAM (`gam`) | R `mgcv::gam`, REML, `s(x, k = 5)` for each continuous covariate, linear binary and factor terms | flexible statistical arm | correctly specified for S2 |
| LightGBM + Platt (`lgbm`), and raw (`lgbm_raw`) | Python LightGBM 4.7.0 | machine-learning reference | 5-fold stratified CV over num_leaves in {7, 15, 31} x min_data_in_leaf in {10, 30}, learning rate 0.05, up to 2,000 rounds with early stopping after 50; refit on all n at the CV-best setting and round count. **Platt scaling** is a logistic fit of Y on the logit of the out-of-fold scores, applied to the refit. **If the Platt slope is <= 0, the arm keeps the raw LightGBM scores and records `platt_fallback`**. A Platt fit that does not converge is a failure. Isotonic scaling appears only in the ladder. |
| **TabPFN-3.5** (`tabpfn`) | `tabpfn` 9.1.0, `ModelVersion.V3_5`, checkpoint tabpfn-v3.5-20260909 (file, repository, revision and SHA-256 in Section 13) | primary TFM | shipped settings: `n_estimators` "auto" and softmax temperature **1.0**, both recorded per replicate; pinned by version (`V3_5`) and SHA-256 |
| TabPFN-3.5 at T = 0.9 (`tabpfn_t1`) | the same checkpoint, `softmax_temperature = 0.9`, the same seed as `tabpfn` | temperature arm | 8 scenarios (S1-S4 at pi = 10 %, n in {200, 3000}), R = 100 |
| Real-TabPFN-2, legacy (`tabpfn_legacy`) | `tabpfn` 9.1.0, `ModelVersion.V2`, its default classifier checkpoint (file, repository and SHA-256 in Section 13) | link to the earlier TabPFN literature | shipped settings (8 estimators, T = 0.9); the 16 core scenarios at pi = 10 %, R = 100 |
| **TabICLv2** (`tabicl`) | `tabicl` 2.2.0, checkpoint tabicl-classifier-v2-20260212 (file, repository, revision and SHA-256 in Section 13) | third TFM family | shipped settings (8 estimators, softmax temperature **0.9**). Its documented lower bound is 300 training rows, so results at n = 200 are reported and flagged. |
| TabICLv2 at T = 1.0 (`tabicl_t1`) | the same checkpoint, `softmax_temperature = 1.0`, the same seed as `tabicl` | temperature arm | the same 8 scenarios, R = 100 |
| **Mitra-v2** (`mitra`) | `MitraClassifier` from `autogluon.tabular` 1.6.3, weights autogluon/mitra-classifier-2 (revision and SHA-256 in Section 13) | second TFM family | in-context only (no fine-tuning, `n_estimators = 1`), with the whole development set as context. A 16-row validation set taken from the development rows is passed only so that AutoGluon holds nothing out; it does not change the predictions. See below for precision, sign flips and the T2 cell. |

**Further rules for TabPFN-3.5.**

- **Why the standard checkpoint, not Fast (timing).** Both TabPFN-3.5 checkpoints were timed on the GPU on 4 scenarios x 2 replicates of the smoke stream, with the run's prediction rows (`results/tabpfn35_fast_timing.csv`):

  | Scenario (predicted rows) | Standard, s per fit | Fast, s per fit |
  |---|---|---|
  | S1_pi03_n200 (4,334) | 23.7-30.6 | 5.2-5.6 |
  | S2_pi10_n500 (2,000) | 14.5-14.6 | 3.2-3.3 |
  | S3_pi30_n1000 (2,000) | 19.0 | 4.1-4.2 |
  | S4_pi03_n3000 (4,334) | 57.4-57.7 | 11.9-12.0 |

  - Fast is 4.5-5.4 times faster. Peak GPU memory was 2.1-2.9 GB for the standard checkpoint and 1.0-2.2 GB for Fast.
  - With the standard checkpoint, schedule P4 (Section 9.3) finishes in about 3 days.
  - The plan's budget fallback "use the Fast variant" therefore does not apply.
  - The main arm is the standard checkpoint: the default of `ModelVersion.V3_5`, and the model the vendor ships as TabPFN-3.5.
  - The Fast checkpoint (`tabpfn-v3.5-fast-20260909.safetensors`; SHA-256 in Section 13) is recorded but not used.
- **Device.** TabPFN-3.5 runs on the GPU for some scenario groups and on CPU workers for others (Section 9.3). The device is recorded per replicate. On the same replicate, CPU and GPU predictions differ by at most 2.6e-4 (mean 2.9e-5).
- **The temperature arm's device.** `tabpfn_t1` runs on the GPU. For its 8 scenarios (pi = 10 %), the main `tabpfn` arm runs on a CPU worker. The two arms therefore differ in temperature (a logit scale of 1/0.9) and, negligibly, in device.

**Further rules for Mitra-v2.**

- **Precision.** Mitra-v2 keeps its **shipped bfloat16 autocast**. The GTX 1660 Ti (Turing) has no bfloat16 hardware, so the arithmetic is emulated. Two support orders change single predictions by 6e-4 on average and by at most 0.06. The effect on the true ICI is about 1e-4, but E_max and the tails of OCE are inflated; this is reported.
- **Sign flips.** Mitra's preprocessor flips feature signs with the global numpy generator. The wrapper seeds that generator from the replicate, so every fit is reproducible.
- **T2, n = 3000.** Mitra-v2 is **not run at T2_pi10_n3000**, where it does not fit in 6 GB of GPU memory. It is recorded there as "not run: does not fit in 6 GB GPU memory". Its context is never cut. A fit whose context AutoGluon cut after an out-of-memory error, in any scenario, is recorded as a failure.
- **Pretraining limits.** The limits of 5,120 support rows and 50 features are respected in every scenario.

**Excluded arms and reasons.**

- Jev: a closed hosted decision model with no fitting interface and no published weights. It cannot be version-pinned, and it would need patient rows sent to a third party.
- TabDPT: pretrained on real public tables, a contamination risk for the real-data part.
- LimiX-2: does not fit the 6 GB GPU.
- Nori: regression only.
- Zero-shot large language models: shown not to work for this task in earlier work.

### 5.3 External goodness-of-fit battery (R, CPU)

**Software.** `ebrahim.gof` 2.10.0, built from commit **19480aa** of the 2.10.0 release branch (C21).

- **The build is checked, not only the version number.** Before the freeze, every one of the 189 functions of the installed namespace was compared with the R sources of commit 19480aa: 0 differences (`R/verify_gof_build.R`). The layer stores the SHA-256 of that code and recomputes it from the installed package at every start. Any other build is refused, including a build of 2c77f3c, which carries the same version number.
- Every output row records "2.10.0@19480aa".

**What is run on each evaluation set.** For every arm, replicate and scenario (subject to the schedule of Section 6.3):

1. **`run.all.external(y, p_hat, G = 10, X)` on the observed Y.** Its rows:
   - EDGE (G = 10), the primary EDGE row, and EDGE (G = auto);
   - the Cox recalibration likelihood-ratio test;
   - calibration-in-the-large;
   - Spiegelhalter z;
   - Osius-Rojek (external);
   - GiViTI (external mode, via `callr`);
   - Hosmer-Lemeshow (external, chi-square with G df);
   - Stukel (offset), the score test (below);
   - le Cessie (external), run when the evaluation covariates are stored (replicates 1-R);
   - three descriptive rows.
2. **`edge.gof(..., external = TRUE)`.** It is kept as a cross-check of the battery's EDGE rows.
3. **The same battery on an exact null outcome Y\* ~ Bernoulli(p-hat).** Y\* is drawn from the arm's own seed stream and stored with the predictions. Under Y\*, the arm's predictions are calibrated by construction, so the rejection rate is the size of each test *on that arm's prediction distribution*.
4. **`deepgof1.external(y, p_hat, X, B = 99)` on the observed Y only**, for FLIC and the three TFMs.
   - The "combined" reading is used, except in designs with more than 12 predictors (scenario T2, and the random designs with p > 12; Section 3.9), where only the **"axes" reading** is used. All pairs of 30 covariates (435 pairs) take about 230 s per call; the axes reading takes under 1 s.
   - DeepGOF-1 is an exact Monte Carlo test. Its size was checked in the auxiliary size study.
5. **`localize.external(y, p_hat, X, M = 999)`** (Section 5.4).

**The size study (review D5).**

- **Main size study: the oracle arm.** In every scenario, the oracle arm's real Y and its Y\* are exact nulls, at the main design's DGP. Over replicates 1-500 this gives 1,000 null evaluation sets per scenario, and the full battery is run on all of them.
- **Each arm's Y\*** (replicates 1-R) gives the size on that arm's own prediction distribution.
- **Auxiliary.** The earlier study (2,000 replicates x 12 scenarios, its own coefficients) is reported as an auxiliary check.

**Stukel (offset): the score form, ranked like every other test (C21).**

- **The problem.** Up to commit 2c77f3c the row was the likelihood-ratio test of Stukel's two terms added to the frozen logit. The auxiliary study found it liberal: 0.063 pooled at N = 3,334, and 0.0745 at S1, pi = 10 %. A separate null simulation (4,000 replicates per cell) traced the cause. The upper-tail term, 0.5 lp^2 for lp >= 0, is carried by only the 10-20 records with p >= 0.5 when events are rare, and its likelihood-ratio statistic is then not chi-square (size 0.078 at pi = 10 % with normal risks).
- **The change.** From commit 19480aa the row is the **score test of the same two terms at the null**, U' I^- U with U = Z'(y - p) and I = Z' diag(p(1 - p)) Z. Nothing is fitted. The columns are scaled to their information and the inverse is rank-aware. When no prediction reaches 0.5 the upper term is empty and the test has 1 df; the row's Note says "score test".
- **Its size.** In the null design and with the seeds of the earlier Stukel study (4,000 nulls per cell; predictions equal to the true risks; pi = 3, 10 and 30 % at N = 3,334, 1,000 and 1,000; normal and t3 risks), the package's own row rejects at 0.035-0.053 at the 0.05 level (MCSE 0.003-0.0035), with a p-value in all 24,000 replicates (`R/stukel_score_size.R`, `results/stukel_score_size.csv`). The author's own check gave 0.037-0.055. The lowest value, 0.035 at pi = 3 % with normal risks, is conservative, not liberal: there the upper term is empty in 37 % of the replicates (1 df). Under t3 risks some predictions reach 1 in double precision; they are first moved to 1 - 1e-9, as the test layer does.
- **Consequence.** Stukel is treated like every other test: it is in the size question (Q4a, Section 7.4) with no exception, in the detection curves and the ranking of tests, and in the summary tables. As for every test, its size on the main design comes from the oracle nulls and from each arm's Y\*, and every table that shows a test's power also shows its size.

**Validation-size subsamples.**

- The battery (no DeepGOF-1, no localization) is also run on the nested e50 subsample at every event rate, and on the e200 subsample at pi = 30 % only (Section 3.7).
- It is run on the observed Y and on Y\*, for every arm except the ladder columns.
- This gives detection and size against validation size.

### 5.4 Localization

- **Call.** `ebrahim.gof::localize.external(y, p_hat, X, M = 999, seed = <per-replicate seed>)`, with familywise error control at 0.05. It names zero or more of the groups INTERCEPT, SLOPE, LINK and COV.
- **Where it runs.**
  - The oracle, FLIC, GAM, the TFMs and the temperature arms, in every scenario where the covariates are stored (replicates 1-R).
  - The primary analysis is on S3 and T3.
  - The oracle rows (real Y and Y\*) give the familywise error rate.
- **T3's factor.** T3's 6-level factor enters as 5 dummy columns.

### 5.5 Recalibration ladder

For FLIC and the three TFMs, with a local sample of m in {100, 500} rows taken from the recalibration set:

| Level | Method |
|---|---|
| R0 | none |
| R1 | intercept only: `glm(y ~ 1, offset = logit(p_hat))` |
| R2 | logistic recalibration: `glm(y ~ logit(p_hat))` |
| R3 | isotonic regression (pool-adjacent-violators), with predictions clipped as in Section 4 |

After each level, the oracle estimands (Sections 4.1-4.2) and the battery on the observed Y (Section 5.3, item 1) are computed on the evaluation set. A level whose local sample has a single outcome class (m = 100 at pi = 3 %) is recorded as failed.

**Test-guided rule (evaluated, not assumed).**

1. If the Cox recalibration test on the R0 predictions rejects at 0.05 on the evaluation set, recalibrate with R2 fitted on the local sample (m = 500).
2. If EDGE (G = 10) still rejects at 0.05 on the evaluation set with the R2 predictions, flag "recalibration insufficient".

---

## 6. Performance measures and Monte Carlo error

### 6.1 Per scenario and arm

- **Summaries.** The mean, median, inter-quartile range, and 10th and 90th percentiles of each estimand across replicates.
  - Medians and IQRs are the main summaries for a\* and b\*, because single separated fits can produce extreme values. In the pilot, ridge's mean b\* was 19.7, driven by one replicate.
- **Monte Carlo standard errors.**
  - For every mean: SD / sqrt(R).
  - For every proportion: sqrt(p(1 - p) / R).
  - For medians: a distribution-free 95 % confidence interval from order statistics, and a bootstrap MCSE.
- **Paired differences against FLIC** (each arm minus FLIC on the same replicates), with the MCSE of the paired difference.
- **Classes.** The proportion of replicates in each calibration class, with MCSE.
- **Tests.** The rejection rate of every test under the observed Y (detection) and under Y\* (size), with MCSE.
- **Fit counts.** Counts of failed, degenerate and not-run fits, of `platt_fallback`, and of `lambda_at_edge` (Section 8).

### 6.2 Detection curves

- **Model.** For each test, P(reject at 0.05) is modelled against the true ICI with a logistic GAM (`mgcv`, `s(true ICI, k = 10)`, REML).
- **Data.** It is fitted across replicates, arms and scenarios at N_eval, with a separate curve for each test.
- **Uncertainty.** A cluster bootstrap over replicates (500 resamples).
- **Every test of the battery** has its curve and its rank, Stukel (offset) included (Section 5.3, C21).

### 6.3 Replicates (schedule P4) and achieved MCSE

| Arms | Scenarios | R | MCSE achieved (pilot SDs) |
|---|---|---|---|
| TabPFN-3.5, TabICLv2, Mitra-v2 | all 58 (Mitra-v2 not run at T2_pi10_n3000) | 200 at n <= 1000; **150 at n = 3000** | mean true ICI <= 0.0007 (0.0009 at n = 3000), against a target of 0.002. Rejection rate at 0.05: 0.0154 (0.0178 at n = 3000), against a target of 0.015. At 0.5: 0.035 (0.041). |
| Oracle, LR-MLE, FLIC, Ridge-CV, GAM, LightGBM (+ raw) | all 58 | 500 | mean true ICI <= 0.0005 for FLIC and 0.0011 for GAM at n = 200, pi = 3 % (target 0.001) |
| Real-TabPFN-2 legacy | 16 core scenarios at pi = 10 % | 100 | descriptive |
| `tabpfn_t1`, `tabicl_t1` | S1-S4 at pi = 10 %, n in {200, 3000} | 100 | descriptive |
| Oracle size study (oracle real Y + Y\*) | all 58 | 500 x 2 = 1,000 nulls | 0.0069 per scenario, 0.0009 pooled |
| Random-design tier: oracle, the classical arms, TabPFN-3.5, TabICLv2, Mitra-v2 | D = 1,000 random designs (Section 3.9) | 1 per design | the precision of the meta-model (Section 6.4): a simultaneous 95 % band of about +/- 0.007 on the true-ICI difference against FLIC at a typical design (Section 6.4) |

- **The n = 3000 shortfall.** At n = 3000, R = 150 misses the rejection-rate target by about 15 % (0.0178 against 0.015). This is the plan's first budget fallback, adopted before the freeze.
- **Common replicates.**
  - Every paired comparison between a TFM and a classical arm uses the replicates that both have (1-200, or 1-150 at n = 3000).
  - The 500-replicate summaries of the classical arms are reported separately.
- **Battery schedule.**
  - The battery runs on every arm for replicates 1-R.
  - It runs on the oracle only (observed Y and Y\*) for replicates R+1 to 500.
  - No pre-registered estimand needs the battery on the classical arms beyond replicate R; their oracle estimands come from all 500.

**Reporting rule.** No statement of "no difference" or "holds size" is written without first giving the MCSE (or confidence interval) of the quantity. A scenario where the quantity is at a floor or a ceiling is reported as uninformative for that comparison.

### 6.4 Meta-model of the random-design tier

The random designs (Section 3.9) are analysed with a pre-specified meta-model, in R `mgcv` (REML).

**Responses.** For each design d and each TFM m (TabPFN-3.5, TabICLv2, Mitra-v2), paired with FLIC on the same data:

- D_ICI(d, m) = true ICI(m) - true ICI(FLIC);
- D_logb(d, m) = |log b\*(m)| - |log b\*(FLIC)|, where |log b\*| is capped at log 10 (b\* <= 0 or no finite b\* counts as log 10). The meta-model uses the values themselves, so it needs this finite cap; the order-based summaries of Section 7.1 place such replicates above every finite value instead.

**Model.** One GAM per response, fitted to all designs where both arms produced usable predictions (Section 8.2):

    D ~ model + s(log n, by = model, k = 6) + s(log pi, by = model, k = 6)
          + s(p, by = model, k = 5) + s(AUC target, by = model, k = 5)
          + s(w_nl, by = model, k = 4) + s(w_int, by = model, k = 4)
          + s(w_link, by = model, k = 4) + model:rho + model:discrete fraction
          + s(design, bs = "re")

- `model` is the TFM factor; each TFM gets its own partial effects.
- The design random effect pairs the three TFMs that share a design's data.
- The family is the scaled t (`scat`), so that a few separated FLIC fits cannot dominate.

**Outputs.**

- Partial-effect plots of every smooth, per TFM, with 95 % **simultaneous** bands. The bands come from 10,000 draws of the coefficients from their approximate posterior, with the critical value of the maximum standardized deviation over the plotted grid.
- **Regions.** For each TFM, every random design's own setting is classified as "TFM better" (the simultaneous band of the predicted D_ICI lies below -0.01), "TFM worse" (above +0.01) or "within 0.01". The shares are reported, with maps on (log n, log pi) at the median of the other factors, and on the two factors with the largest partial effects.
- The same maps for D_logb, with a margin of log 1.1.
- **Detection.** For every test and every arm, the rejection at 0.05 on the observed Y (0/1) is modelled with the same terms and a logit link (binomial family, no design random effect, one model per arm and test). The size on the oracle and on each arm's Y\* is modelled the same way, and every region where the simultaneous band of the predicted size excludes [0.04, 0.06] is reported.
- **Mitra-v2** is modelled only where it ran (Section 3.9); the region where it did not fit is shown on every map.

**Precision.** One replicate per design is enough because the meta-model pools neighbouring designs. Take a residual SD of 0.012 for D_ICI (the pilot's largest SD of a paired true-ICI difference) and about 40 effective degrees of freedom per TFM. With D = 1,000, the pointwise SE of the predicted difference at a typical design is then about 0.012 x sqrt(40 / 1,000) = 0.0024, and the simultaneous band is about three times wider (+/- 0.007). A difference larger than about 0.017 is therefore classified as "better" or "worse" at the 0.01 margin. At the edges of the design space the bands are wider, and the maps show them.

---

## 7. Research questions and reporting rules

This is an estimation-focused protocol. It states no confirmatory hypothesis and predicts the direction of no result. Each research question below fixes, before any main-run result is seen, the estimands, the comparison, the summary with its Monte Carlo standard error (MCSE), and the reporting rule that decides when a model is called better, worse or equivalent. The results then determine the conclusions, and the rules cannot change after the data are seen.

### 7.0 Rules common to every question

- **"TFM".** The word means each of the three TFM arms judged separately: TabPFN-3.5 (`tabpfn`), Mitra-v2 and TabICLv2 at their shipped settings. The legacy and temperature arms are not part of the main comparison set; they are reported beside it.
- **Comparator.** Every comparison is paired against FLIC: the TFM minus FLIC on the same replicates. It uses the replicates where both arms produced usable predictions (Section 8.2).
- **Summaries.**
  - For a difference in true ICI: the mean paired difference, with MCSE = SD / sqrt(R) and the 95 % interval mean +/- 1.96 MCSE.
  - For differences in |log b\*| and |a\*|, which single separated fits can make extreme: the Hodges-Lehmann (HL) estimate of the paired differences, with its distribution-free 95 % interval (from the signed-rank statistic) and a bootstrap MCSE.
- **Classification.** For a paired difference where a negative value favours the TFM, and a margin delta:
  - **better** than FLIC: the whole 95 % interval lies below -delta;
  - **worse** than FLIC: the whole interval lies above +delta;
  - **equivalent** to FLIC: the whole interval lies inside (-delta, +delta);
  - **inconclusive**: otherwise.
- **Margins**, fixed now:
  - true ICI: delta = 0.01, the margin of Section 6.4; at pi = 3 % it is multiplied by 0.3 (delta = 0.003), by the rule for probability-scale thresholds of Section 4.5;
  - |log b\*|: delta = log 1.1, the slope margin of Section 6.4;
  - |a\*|: delta = 0.05, half the weak-calibration tolerance on a\* (Section 4.5).
- **MCSE requirement.** The reporting rule of Section 6.3 applies to every question: no "equivalent", "no difference" or "holds its size" is written without the interval or MCSE of the quantity, and a cell at a floor or a ceiling is reported as uninformative for that comparison.
- **No p-value decides a label.** The intervals are 95 % intervals, not adjusted for multiplicity. Each question reports how many cells fall in each class, so that the reader can weigh the number of cells.
- **Reporting.** Every label is reported in a summary-of-findings table with one row per question, whatever it is.

### 7.1 Q1. How far from 1, and in which direction, is each TFM's calibration slope at small n?

- **Estimands.** The true slope b\* of each TFM and of FLIC (Section 4.1), and the paired difference d = |log b\*(TFM)| - |log b\*(FLIC)|. A replicate with b\* <= 0 or no finite b\* gets |log b\*| = 10, which places it above every finite value.
- **Main cells.** pi = 10 % and n in {200, 500}, in the four shapes: 8 cells per TFM. The same summaries are reported in all 48 core scenarios.
- **Summaries.** The HL estimate of d with its 95 % interval and bootstrap MCSE; the median b\* of each arm with its distribution-free 95 % interval (Section 6.1).
- **Reporting rules.**
  - *Closeness to 1.* In each cell, the TFM is better than, worse than or equivalent to FLIC, or inconclusive, by Section 7.0 with delta = log 1.1.
  - *Direction.* In each cell, an arm is called **under-confident** when the 95 % interval of its median b\* lies above 1.1, **over-confident** when it lies below 0.9, **close to 1** when it lies inside [0.9, 1.1] (the slope tolerance of weak calibration, Section 4.5), and **unclear** otherwise. FLIC is labelled in the same way.
  - For each TFM, the number of the 8 main cells in each class is reported.

### 7.2 Q2. At what development size does each TFM reach calibration, compared with FLIC?

- **Estimands.** The minimum n for calibration (Section 4.5) of each arm, in each shape, at each event rate and for each class; and the proportion of replicates that meet each class at each n, with MCSE.
- **Comparison.** For each TFM, shape and event rate: the TFM's minimum n for weak calibration against FLIC's, both from the common replicates. "> 3000" ranks above 3000.
- **Reporting rules.**
  - The TFM **needs a smaller n** than FLIC, **the same n** or **a larger n**, by comparing the two minimum n.
  - The comparison is repeated with the 80 % criterion replaced by its lower 95 % limit (proportion - 1.96 MCSE >= 0.80). An ordering is called **clear** only when both versions give it; otherwise it is called **sensitive to Monte Carlo error**.
  - The comparison is also repeated at half and at double every threshold, and reported beside the main one.
  - Results are reported for all four shapes and all three event rates; no shape is singled out in advance.

### 7.3 Q3. How large is each TFM's calibration-in-the-large error when events are rare?

- **Estimands.** The true intercept a\* (Section 4.1), the paired difference d = |a\*|(TFM) - |a\*|(FLIC), and mean p-hat minus mean p\* (Section 4.2).
- **Cells.** pi = 3 %, the four shapes and all four n: 16 cells per TFM. The cells at n >= 1000 are shown first, because the pilot suggested that the shipped softmax temperature matters most there (Section 0.2, C2). The cells at n <= 500 are reported with the same rule. The same summaries are also reported at pi = 10 % and 30 %.
- **Summaries.** The HL estimate of d with its 95 % interval and bootstrap MCSE; the median a\* of each arm with its 95 % interval.
- **Reporting rule.** Better, worse, equivalent or inconclusive by Section 7.0, with delta = 0.05. Each TFM's shipped softmax temperature is shown beside its result.
- **Temperature arms.** At pi = 10 %, the paired effect of the temperature (`tabpfn_t1` minus `tabpfn`, and `tabicl_t1` minus `tabicl`) on a\*, b\*, mean risk and true ICI is reported with the same summaries and margins, as a direct estimate of the mechanism.

### 7.4 Q4. How do the goodness-of-fit tests behave on TFM predictions?

*Q4a. Size.*

- **Estimand.** The rejection rate at 0.05 of every battery test under the exact null Y\* ~ Bernoulli(p-hat), for every arm, with MCSE. For the oracle, its real Y and Y\* over replicates 1-500 are used (Section 5.3). Stukel (offset), now the score test, is included with no exception (C21).
- **Reporting rule**, for each (test, arm):
  - **holds its size**: the pooled size over all scenarios where the arm was run lies in [0.04, 0.06], and the per-scenario size lies inside its 99 % Monte Carlo band (0.05 +/- 2.576 MCSE) in at least 95 % of scenarios (about 1 % of scenarios are expected outside the band by chance);
  - **liberal**: the pooled size is above 0.06;
  - **conservative**: the pooled size is below 0.04;
  - **uneven**: the pooled size lies in [0.04, 0.06], but fewer than 95 % of scenarios are inside their band.
- Every (test, arm, scenario) outside its band is listed.

*Q4b. Detection.*

- **Estimands.** For every test, the detection curve P(reject at 0.05 | true ICI) of Section 6.2, and the rank of the tests.
- **Main comparison.** EDGE (G = 10) against Hosmer-Lemeshow (external): the difference P(reject | EDGE) - P(reject | HL) at true ICI in {0.01, 0.02, 0.03, 0.05}, from curves fitted on the same replicates, with cluster-bootstrap 95 % intervals.
- **Reporting rule**, at each of the four points, with a margin of 2 percentage points: EDGE **detects more often** when the interval lies above +0.02, **less often** when it lies below -0.02, **equivalently** when it lies inside (-0.02, +0.02), and the comparison is **inconclusive** otherwise.
- Every table and figure that shows a test's detection also shows its size (Section 5.3).

### 7.5 Q5. How well does recalibration repair each TFM, and does the test-guided rule pick the right cases?

*Q5a. Repair.*

- **Estimands.** The mean true ICI of each arm after each level of the ladder (Section 5.5), with MCSE.
- **Comparisons.**
  - *Against FLIC.* The TFM after R2 (m = 500) minus FLIC after R2 (m = 500), paired, labelled by Section 7.0 with the true-ICI margin.
  - *Against the best arm.* In each core scenario, the best arm is the one with the smallest mean true ICI among FLIC and the three TFMs after R2 (m = 500), and LR-MLE, Ridge, GAM and LightGBM (Platt) as fitted. The statistic is the TFM's mean true ICI after R2 minus the best arm's mean true ICI, with the paired MCSE.
- **Reporting rule against the best arm.** In a scenario, the TFM is **within 0.005 of the best arm** when the upper 95 % limit of the difference is <= 0.005, **more than 0.005 above it** when the lower limit is > 0.005, and **unclear** otherwise. For each TFM, the counts over the 48 core scenarios are reported.

*Q5b. The test-guided rule.*

- **Truth.** A replicate is "truly insufficient" if its true ICI after R2 (m = 500) is > 0.015, or > 0.0045 at pi = 3 % (Section 4.5).
- **Estimands.** PPV = P(truly insufficient | flagged) and NPV = P(not truly insufficient | Cox rejected and not flagged), pooled over the core scenarios, for each TFM and for FLIC, with cluster-bootstrap 95 % intervals.
- **Reporting rule.** The PPV is called **at least 0.8** when its lower 95 % limit is >= 0.8, **below 0.8** when its upper limit is < 0.8, and **unclear** otherwise. If fewer than 50 replicates are flagged for an arm, its PPV is reported as "not assessable" (its MCSE would exceed 0.057). The NPV is reported with its interval and no label.

### 7.6 Q6. Do the grid findings carry over to randomly drawn designs? (robustness check)

Q6 is answered by a pre-specified robustness check. It checks whether the core design's results on the true ICI transfer to the random-design tier (Section 3.9). It is a check of transfer, not a hypothesis.

- **Transfer check.** For each TFM and each of the 48 core scenarios, the meta-model of Section 6.4 predicts D_ICI at the scenario's own coordinates: its n and pi, p = 8, AUC target = its oracle AUC, rho = 0.018 (the mean pairwise correlation of the core covariates: one pair at 0.5 among 28), discrete fraction 0.25, and the severities of its shape (S1: all 0; S2: w_nl = 0.5; S3: w_int = 1; S4: w_link = 1).
- A scenario **agrees** when the core design's mean paired difference (TFM - FLIC, replicates 1-R) and the prediction differ by less than 1.96 sqrt(SE_meta^2 + MCSE_core^2).
- **Reporting rule**, per TFM:
  - **the grid transfers**: at least 40 of the 48 core scenarios agree (83 %);
  - **the grid transfers in part**: 30 to 39 of the 48 agree;
  - **the grid does not transfer**: fewer than 30 agree.
- Mitra-v2 is checked only on the core scenarios inside the region where it ran in the random tier.
- **Reported with the check, whatever it shows:** the robustness map, that is, the regions of Section 6.4 where each TFM is better or worse than FLIC by more than 0.01; and every core scenario that does not agree.
- **Interpretation, fixed in advance.** If the grid does not transfer for a TFM, the core design's fixed levels are not typical of the wider design space for that TFM. The paper then reports the random-tier maps as the general result for that TFM, and the core design as a set of special cases.

### 7.7 Equal prominence for every outcome

- If every TFM is at least weakly calibrated in at least 80 % of replicates at n = 200, in every shape and at every event rate, the paper reports that TFMs remove the small-sample calibration problem, with the same prominence as the opposite result.
- Whatever direction Q1 finds (under-confident, over-confident or close to 1) is reported as the finding, with the same prominence.

---

## 8. Exclusions, failures and stopping

### 8.1 Stopping

- The replicate counts of Section 6.3 are fixed. They are not changed after any result of the main run is seen.
- No replicate is redrawn for any reason, including a development set with no events.

### 8.2 Failures, degenerate fits and pairing

- **Failure.**
  - An arm counts as failed when it raises an error or returns a missing, non-finite or out-of-range prediction.
  - A single-class development set is recorded as the same failure for every arm except the oracle.
  - Failures are logged with the error message and kept.
  - The R arms have a limit of 1,800 s per call. The Python arms have no time limit (the pilot's longest fit was 134 s). A crashed or killed process is restarted, and the restart refits only the missing arms (resume).
- **Not run.** Mitra-v2 at T2_pi10_n3000 (Section 5.2), and in the random designs above its memory limit (Section 3.9). It is shown in every table as "not run".
- **Degenerate fit.** A fit that ran is labelled **"degenerate"** (not "ok") in either of two cases:
  - more than 100 of its evaluation predictions lie within 1e-6 of 0 or 1;
  - LR-MLE shows separation (non-convergence, fitted probabilities numerically 0 or 1, or |coefficient| > 15).

  Degenerate fits are the arm's real output. They are scored, tested and kept in every comparison, and counted in every table. LR-MLE summaries are also reported without them.
- **Other flags,** counted per arm and scenario:
  - `platt_fallback` (LightGBM);
  - `lambda_at_edge` (ridge);
  - `small_n_flag` (TabICLv2 at n < 300);
  - the number of predictions at the boundary.
- **Pairing rule** (review D8).
  - Every comparison between arms uses only the replicates where both arms produced usable (ok or degenerate) predictions.
  - The known few-event failures are part of the result, not hidden:
    - ridge fails when the development set has 2 or fewer events (glmnet needs 2 events in every CV fold), about 6 % of replicates at n = 200, pi = 3 %;
    - LightGBM fails at 1 or fewer events;
    - LR, FLIC, GAM and the TFMs run with 1 event;
    - the ladder's R1-R3 at m = 100 fail when the 100-row local sample has no event;
    - EDGE is "not run" on isotonic (R3, m = 100) predictions with too few distinct values; such rows count neither as rejections nor as non-rejections, and the test-guided rule uses R2 only.
  - **The 5 % flag.** If an arm fails in more than 5 % of the replicates of a scenario, that scenario is flagged in all tables, and its results for that arm are reported but not used in the labels of Section 7. Because failures are not redrawn, this flag is part of the result.
  - **Random designs.** Each design has one replicate, so the 5 % flag does not apply. A failed or not-run arm leaves that design out of that arm's meta-model, and the failures are mapped over the design space like any other outcome.

---

## 9. Pilot, smoke test and compute plan

### 9.1 Phase-1 pilot (before the freeze; smoke seed stream)

See Section 0. The pilot report records:

- times;
- GPU memory;
- the failure rates;
- the calibration summaries;
- the test-layer cost;
- the reproducibility checks: predictions are bit-identical across processes and under CPU load, and TabICL differs by at most 8e-7.

### 9.2 Final smoke test (2026-10-07, before the freeze)

**What was run.** Every arm on S1_pi03_n200 and T2_pi10_n500, 2 replicates each, on the smoke stream. The files were laid out exactly as in the main run: classical arms, the GPU process, and TabPFN-3.5 on a CPU worker for T2. Then the scoring and the test layer: the oracle-only mode first, then every arm.

**Results.**

- **Fits.** All 52 arm fits ran: 49 "ok" and 3 "degenerate" (LR-MLE and GAM at S1, GAM at T2), with 0 failures.
- **Temperatures and checkpoints.** The recorded temperatures were 1.0 (`tabpfn`), 0.9 (`tabpfn_t1`, `tabicl`, `tabpfn_legacy`) and 1.0 (`tabicl_t1`). Both TabPFN-3.5 arms carried the pinned SHA-256.
- **Test layer.** It wrote gzip files with 0 errors. Every arm had both real-Y and Y\* rows, including an "Osius-Rojek (external)" row with a finite p-value (104 of 104 arm x outcome pairs). DeepGOF-1 used the axes reading in T2 (0.2-0.3 s per call).
- **Pass criteria.** The pass criteria were met: no unlogged error, finite predictions, fingerprints asserted, and files written and resumable.
- **Unit tests.** 41 Python tests (in both environments), 11 R tests and the kill-and-resume check passed.

**Second smoke run (2026-10-07, after C21-C23).** The package pin moved to commit 19480aa and the random-design tier was added, so the smoke was repeated on the smoke stream (`code/smoke_freeze_b.sh`, `results/smoke_freeze_b/SUMMARY.md`):

- **What was run.** Every arm on S1_pi03_n200 and T2_pi10_n500 (2 replicates each), and the tier's arms on the random designs R0001-R0005 (one replicate each), laid out as in the main run. TabPFN-3.5 ran on the GPU for four designs and on a CPU worker for R0002 ((n + N_eval) x p = 128,910). Then the scoring and the test layer: the oracle-only mode first, then every arm.
- **Fits.** Of the 102 fits, 94 were "ok", 7 "degenerate" (LR-MLE and GAM in S1 and in two random designs, GAM in T2) and 1 "not run": Mitra-v2 in R0002 (n^2 x p = 9.8e7), recorded as "does not fit in 6 GB GPU memory". There were 0 failures.
- **Test layer.** 9 gzip files with 0 open errors, every one with the reference 2.10.0@19480aa. Every arm had real-Y and Y\* rows. All 202 Stukel (offset) rows say "score test" and have a finite p-value (df 2, or 1 where no prediction reached 0.5), and all 202 arm x outcome pairs have an "Osius-Rojek (external)" row with a finite p-value. DeepGOF-1 used the axes reading in T2 and in the three random designs with p > 12, and the combined reading in R0001 (p = 12) and R0005 (p = 8). The 52 rows marked "not run" or "failed" all belong to the degenerate GAM fits and to the isotonic ladder column `flic__R3_m100`, as Section 8.2 foresees.
- **Unit tests.** 48 Python tests (in both environments), 16 R tests, the kill-and-resume check, `dgp.py check` (the core table is unchanged, difference 0), the core-fingerprint test of the family-R code and the build check of commit 19480aa passed.

### 9.3 Schedule P4: streams, devices and budget

**Streams.** Four streams run in parallel on a 12-core / 24-thread machine with one GTX 1660 Ti 6 GB. About 5 cores stay free. A fifth stream, the random-design tier, takes over their slots as they finish.

| Stream | Cores | Work, in order |
|---|---|---|
| GPU process (`--device cuda`, 6 threads) | GPU + 1 | TabICLv2 and Mitra-v2 on all scenarios; then TabPFN-3.5 on the 16 core scenarios at pi = 3 % and on T2_pi10_n500; then legacy V2; then the two temperature arms |
| CPU worker A (TabPFN-3.5, `--device cpu`, 6 threads) | 6 | T2_pi10_n3000 first, then the 16 core scenarios at pi = 30 %, then T1 and T3 (4 scenarios) |
| CPU worker B (TabPFN-3.5, `--device cpu`, 6 threads) | 6 | the 16 core scenarios at pi = 10 %, then T4a and T4b (4 scenarios) |
| Classical and tests (6 workers) | 6 | the classical arms (R = 500) first; then the test layer and the scoring, streaming every pass until the model streams finish; then the validation-size subsamples |
| Random-design tier (Section 3.9) | the slots it takes over | when CPU worker B is done: the classical arms of the random designs (6 workers), then TabPFN-3.5 on half of the CPU designs; when worker A is done: the other half; when the GPU process is done: TabICLv2 and Mitra-v2 on every design and TabPFN-3.5 on the GPU designs; when the classical stream is done: the test layer and the scoring |

**Device rules.**

- Each scenario's TabPFN-3.5 fits run on one device only. The three groups above cover the 58 scenarios exactly once (`code/main_groups.py` asserts it).
- After about 24 hours the balance is checked:
  - a CPU worker that is clearly behind hands its *unstarted* scenarios, whole, to the GPU process (it is stopped first, so no replicate is fitted twice);
  - a worker that is ahead takes unstarted scenarios from the other.
- The device actually used for every scenario is reported in the paper. The scenarios fitted on the CPU are listed in `DEVIATIONS.md` only if they differ from this plan.
- The random designs' TabPFN-3.5 device follows the size rule of Section 3.9. Those designs are not moved in the balance check.

**Budget** (pilot measurements, about 118 GPU-hours plus about 490 core-hours before balancing):

- about 79 hours for the GPU stream, including about 8 GPU-hours for the two temperature arms;
- 63-77 hours for CPU worker A and 57-71 hours for CPU worker B (CPU/GPU time ratio 2.4-3.0);
- the classical arms take about 9 hours of the fourth stream;
- the 58 scenarios end after about 3.3-3.4 days (about 82 hours, with the last pass of the test layer).

The DeepGOF-1 axes reading in T2 removes about 70 core-hours from the pilot's estimate of the test layer (at B = 99).

**The random-design tier** (D = 1,000). The costs are measured per fit (`code/random_budget.py`, whose output is `results/random_budget.md`). Times such as "79 h" are hours from the start of the run.

- **GPU:** about 6.8 hours, from about 79 h: TabICLv2 and Mitra-v2 on every design where they run, and TabPFN-3.5 on the 821 GPU designs.
- **TabPFN-3.5 on the 179 CPU designs:** about 5.3 hours on one 6-thread worker, so about 2.7 hours on each of the two workers; done by about 73 h.
- **The classical arms:** about 4.8 core-hours (about 50 minutes on 6 workers), from about 64 h.
- **The test layer and the scoring:** about 13 core-hours (about 46 core-seconds per design at B = 99).
- **The basis:** per-fit time models fitted to the pilot's and the memory probe's GPU timings (log-scale RMSE 0.14-0.34), a CPU/GPU time ratio of 3.0 for TabPFN-3.5 on 6 threads, and the 50-design cost sample for the classical arms and the test layer, measured on a loaded machine.
- **Choice of D.** The tier was to add at most about one day, with D >= 600 and a target of 1,000. With D = 1,000 it adds about 4.3 hours (about 11 hours if every cost were twice the measured one), so D = 1,000.
- **New total.** The expected wall time of the whole run is about **3.6 days** (about 86 hours).

**Budget fallbacks of the plan.**

1. R = 150 at n = 3000: **adopted**.
2. the faster variant of TabPFN-3.5: **not needed** (Section 5.2).
3. drop the legacy arm: **not adopted**; the legacy arm is kept at R = 100.

### 9.4 Engineering

- **Writing replicates.**
  - Every replicate is written to disk as soon as it is done. The file holds the fingerprints, p\*, Y, Y\* and the p-hat of every arm as float32, and it is written atomically.
  - A lock file per replicate makes the streams safe to run together.
  - A killed stream resumes with the same command.
- **File hand-off.** The statistical arms and the test layer run in R on the saved files; the TFMs run in Python; the hand-off is by file only. The test layer writes one gzip-compressed csv per replicate and reads csv or csv.gz.
- **The start command checks:**
  - the software versions of Section 13;
  - the SHA-256 of every checkpoint (it writes `results/environment.json`);
  - that the installed `ebrahim.gof` is the build of commit 19480aa (`R/verify_gof_build.R`);
  - that the random designs re-solve to the stored table (`code/dgp_random.py check`);
  - the test-layer unit tests.
- **The deposit guard.** The run refuses to start until the registration link and timestamp are recorded in `prereg/DEPOSITED.txt`.

---

## 10. Real-data analysis plan

### 10.1 Data sets, outcomes and splits

Event rates are computed from the files and reported; they are not taken from the literature. The predictor lists below are fixed from the codebooks before any outcome is tabulated, and they are limited to variables available at the time of prediction.

| Data set (access, licence) | Outcome(s) | Predictors | Development / external split |
|---|---|---|---|
| **International Stroke Trial**, IST (Edinburgh DataShare, doi:10.7488/ds/104; ODC-By, plus the DataShare end-user licence accepted by the author). The file has been downloaded and has not been tabulated. | `ist_death14`: death within 14 days (patients with known 14-day status); `ist_dd6`: dead or dependent at 6 months (unknown status excluded) | the 21 baseline variables recorded at randomisation: age; sex; delay from onset (h); systolic blood pressure; conscious state (alert, drowsy, unconscious); symptoms on waking; atrial fibrillation; aspirin within 3 days; heparin within 24 h; CT status (not done, done without visible infarct, visible infarct); stroke subtype (TACS, PACS, LACS, POCS, other); the eight deficit items; and a flag for "a deficit could not be assessed" | primary: UK (development) against all other countries (external). Sensitivity: randomised before 1994 against 1994-1996. If the UK pool has fewer than 3,500 complete cases: UK, Ireland and the Northern European countries (Norway, Sweden, Denmark, Finland, the Netherlands). |
| **CRASH-2** (LSHTM Data Compass / freeBIRD, doi:10.17037/DATA.10; CC BY as stated by freeBIRD). The download waits for the author's freeBIRD registration. | `crash2_death28`: death in hospital within 28 days of randomisation | following Perel et al. (2012): Glasgow coma score, age, systolic blood pressure, heart rate, respiratory rate, hours since injury (continuous), injury type (blunt only or any penetrating), and sex. The data dictionary's missing-value codes are set to missing. | temporal: randomised before 1 January 2008 against 2008-2010. The public file has no country. If the early pool has fewer than 3,500 complete cases, the cut moves to the date that splits the complete-case table in half. |
| **Sepsis survival, minimal clinical records** (UCI repository id 827, doi:10.24432/C53C8N; CC BY 4.0) | `sepsis_death`: death during the admission | age, sex, and septic episode number | primary: the 91,153 admissions not meeting Sepsis-3 (development) against the 19,051 Sepsis-3 admissions (external). Sensitivity: a random half split. |
| **PhysioNet/CinC Challenge 2012** (PhysioNet, ODC-By 1.0) | `physionet_ihd`: in-hospital death | 27 summaries of the first 48 hours, following the SAPS-I variable list: age; sex; ICU type; the minimum and maximum of heart rate, systolic blood pressure, temperature, respiratory rate, white cell count, glucose, potassium and sodium; the minimum of GCS, haematocrit and bicarbonate; the maximum of BUN, creatinine and bilirubin; total urine output; any mechanical ventilation | primary: sets A + B (development) against set C (held out). Sensitivity: coronary-care, cardiac-surgery and surgical ICUs against the medical ICU. |

- **The recalibration rows.** In every external region, 500 random rows (fixed seed) form the local recalibration sample, and the remaining rows form the evaluation set.
- **If CRASH-2 cannot be obtained.** The real-data analysis proceeds with the other three data sets, and this is reported.

**Missing data.**

- IST and CRASH-2: complete-case analysis on the predictors. Excluded rows are counted per data set and split. In IST, atrial fibrillation is missing for the 984 pilot-phase patients (5.1 %).
- Sepsis: no missing values.
- PhysioNet 2012: complete cases would lose most stays. Inside each development subsample, medians are learned on the development rows, and a missing-value indicator is added for every variable missing in at least 5 % of those rows. Every arm gets the same imputed matrix.
- A sensitivity analysis keeps the incomplete rows (IST, CRASH-2) or uses the TFMs' native missing-value handling (PhysioNet).

### 10.2 Learning curves

- **Subsamples.** For each data set and outcome: B = 100 development subsamples at each n in {200, 500, 1000, 3000}, drawn by simple random sampling from the development region. The seeds are derived from (data set, split, n, b).
- **Evaluation.** Each subsample is evaluated on the **whole** external evaluation set.
- **Arms.** All arms of Section 5.2 except the oracle, with the categorical declarations of Section 5.1.
- **Small regions.** If the development region holds fewer than 3,000 rows, the largest n is the region size.
- **Few events.** Subsamples with very few events are kept.
- **Estimands:**
  - empirical calibration slope and calibration-in-the-large;
  - empirical ICI, E50 and E90;
  - Brier score and AUC;
  - the empirical analogue of the calibration class (reported as empirical, not true);
  - the battery of Section 5.3 on the external evaluation set.

### 10.3 Localization and repair on real data

- **Localization.** `localize.external()` on the external evaluation set for FLIC and the TFMs, with these covariates: age band, sex, region (where present) and a severity score (conscious state in IST, GCS in CRASH-2, SAPS-I in PhysioNet).
- **Repair.** The recalibration ladder of Section 5.5 is fitted on the 500-row local sample and evaluated on the rest of the external region.

### 10.4 Plasmode tier

A known true risk is placed on real covariate distributions, with outcomes that no TFM could have seen during pretraining.

- **Truth model.**
  - A main-effects logistic model is fitted once to the real outcome on the full complete-case table, with standardized design columns.
  - Two terms of fixed size are then added on the logit scale: a U-shaped term 0.4 (z^2 - 1) in one standardized continuous predictor, and an interaction 0.6 z_a x_b.
  - The intercept is solved by bisection so that the mean p\* equals 10 % and 30 %.

  | Data set | U-shaped term | Interaction |
  |---|---|---|
  | IST (`ist_dd6`, primary) | systolic blood pressure | age x impaired consciousness |
  | CRASH-2 | systolic blood pressure | GCS x penetrating injury |
  | PhysioNet 2012 | maximum heart rate | age x mechanical ventilation |

- **Design.** R = 200 replicates per n in {200, 500, 1000, 3000}, with development subsamples from the development region and evaluation on the external evaluation set.
- **Analyses.** The oracle estimands, the summaries of research questions Q1-Q3 (as secondary checks) and the size of the battery are computed as in the simulation.
- **Frozen truths.** Each truth is stored with a SHA-256 of p\* and checked before each use.

### 10.5 Reporting

The real-data analyses are reported following TRIPOD+AI (Collins et al. 2024). Real-data results are descriptive and supportive; they do not enter the labels of Section 7.

---

## 11. Primary, secondary and exploratory analyses

**Primary.**

1. The oracle estimands: the true calibration slope b\*, the true intercept a\* and the true ICI of every arm in the 48 core scenarios, with MCSE.
2. The paired differences in true ICI, TFM minus FLIC, per core scenario: the mean paired difference with its MCSE and 95 % interval, labelled better, worse, equivalent or inconclusive by the rule and margin of Section 7.0.
3. The answers to research questions Q1-Q5, by the reporting rules of Sections 7.1-7.5.
4. **The minimum-n-for-calibration table** (Section 4.5), a planned primary output for practitioners: for each arm, at each event rate (3, 10 and 30 %) and in each shape, the smallest development n at which the arm meets each calibration class in at least 80 % of replicates, beside Riley's minimum n and the EPV. The pooled row of each event rate gives the largest of the four shapes. The same table with the lower 95 % limit of the proportion, and at half and double the thresholds, goes to the supplement. The table is descriptive; its only labels are those of Q2.
5. **The robustness map** of the random-design tier: the regions of Section 6.4 where each TFM is better or worse than FLIC by more than 0.01, with the robustness check of Section 7.6 (Q6).

**Secondary.**

- The rest of the meta-model of Section 6.4: partial effects, the D_logb maps, and the detection and size maps.
- All other oracle and empirical estimands, `true_ll_ici`, and empirical against true ICI.
- Size and detection of every test:
  - the oracle nulls (the main size study);
  - each arm's Y\*;
  - the detection curves;
  - the validation-size subsamples;
  - Stukel's size beside its power;
  - the auxiliary size study.
- Localization: the proportion of replicates in which COV is named, on S3 and T3, for each arm. As a reference for each arm: the oracle share of the variance of logit(p\*) not explained by a smooth function of logit(p-hat). A large share means that a covariate-level error truly exists for that arm.
- The full recalibration ladder (R0-R3, m = 100 and 500) and the NPV of the test-guided rule.
- The temperature arms: the paired effect of the softmax temperature on b\*, a\*, mean risk and true ICI, within TabPFN-3.5 and within TabICLv2.
- The counts of failed, degenerate and not-run fits, and of the flags of Section 8.2.
- Real data (Section 10).

**Exploratory** (labelled as such in the paper):

- the stress set beyond the severity ladder;
- the legacy Real-TabPFN-2 arm;
- prediction instability;
- any analysis not listed in this document.

**Main-text figures and tables** are planned as:

- a design schematic;
- b\* against n;
- oracle calibration curves;
- minimum n by arm, and the minimum-n-for-calibration table;
- the random-design maps: where each TFM is better or worse than FLIC;
- detection curves with a size inset;
- a recalibration slopegraph;
- real-data learning curves and a localization map;
- a summary-of-findings table (one row per research question);
- a size table (every test on every arm).

---

## 12. Deviations policy

- `DEVIATIONS.md` is deposited empty with this registration, and again with each later version of the project archive.
- **Recording a change.** Every change after the freeze is recorded there with: the date; what changed; why; whether any main-run result had been seen when the decision was made; and its expected effect.
- **Minor deviations** are engineering changes that do not alter an estimand, a sample, a model setting or a decision rule. An example is a bug fix that is verified not to change stored predictions. They are listed and summarized in the supplement.
- **Major deviations** are any change to the DGPs, arms, settings, replicate counts, estimands, thresholds, margins, research questions or reporting rules. They are listed in the paper. Results that depend on a major deviation are labelled exploratory, and the pre-registered analysis is also reported where it can be run.
- **Bugs.** A bug found after results are seen is fixed, the affected parts are rerun in full, and both the bug and the rerun are recorded.
- **Device changes.** A change of device for a scenario group under the rule of Section 9.3 is not a deviation, but every scenario's device is reported.
- **"Pre-registered."** The paper uses the word only for the analyses in this document, and only because the registration date precedes the main run.

---

## 13. Software and hardware

| Component | Version |
|---|---|
| Python (both environments) | 3.11.15 |
| PyTorch (CUDA build) and CUDA runtime | 2.13.0+cu126, CUDA 12.6 (NVIDIA driver 591.86) |
| numpy, scipy, pandas, pyarrow | 2.4.6, 1.17.1, 2.3.3, 24.0.0 |
| scikit-learn, LightGBM (Python) | 1.9.1, 4.7.0 |
| tabpfn | 9.1.0 |
| autogluon.tabular (for `MitraClassifier`), transformers, huggingface-hub | 1.6.3, 5.19.0, 1.33.0 |
| tabicl | 2.2.0 |
| R | 4.4.1 |
| ebrahim.gof | 2.10.0 (release branch, commit 19480aa; the build is checked by the SHA-256 of its code) |
| givitiR, callr, logistf, glmnet, mgcv, pmsampsize | 1.3, 3.7.6, 1.26.1, 4.1-10, 1.9-4, 1.1.3 |
| arrow, data.table, digest, jsonlite | 23.0.1.2, 1.18.2.1, 0.6.39, 2.0.0 |
| Hardware | Windows 11 Pro; AMD Ryzen 9 3900X (12 cores, 24 threads); 32 GB RAM; NVIDIA GeForce GTX 1660 Ti 6 GB (Turing; bfloat16 emulated) |

**Model checkpoints.** Revision = the Hugging Face commit of the repository in the project cache.

- TabPFN-3.5, standard (arms `tabpfn`, `tabpfn_t1`): `tabpfn-v3.5-20260909.safetensors`, repository `Prior-Labs/tabpfn_3_5`, revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`.
    - SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`
- TabPFN-3.5-Fast (recorded, not used): `tabpfn-v3.5-fast-20260909.safetensors`, same repository and revision.
    - SHA-256 `14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550`
- Real-TabPFN-2 (arm `tabpfn_legacy`): `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, repository `Prior-Labs/TabPFN-v2-clf`.
    - SHA-256 `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`
- TabICLv2 (arms `tabicl`, `tabicl_t1`): `tabicl-classifier-v2-20260212.ckpt`, repository `jingang/TabICL`, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
    - SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
- Mitra-v2 (arm `mitra`): `model.safetensors`, repository `autogluon/mitra-classifier-2`, revision `edada0d20759c58ada8c8605c25f22f6e98ea5f0`.
    - SHA-256 `5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09`

At the start of the main run, all of these are written to `results/environment.json` and checked against this table. A mismatch stops the run.

---

## 14. Archive and availability plan

- **This pre-registration** is registered on OSF Registries before the main run.
- **Code** goes in a public GitHub repository under the MIT licence; **results, figures and tables** under CC BY 4.0. The repository holds `CITATION.cff` and `.zenodo.json` with the author and his ORCID.
- **Zenodo**: a version DOI from a GitHub release. The paper cites the concept DOI.
- **The ebrahim.gof 2.10.0 source** used here (commit 19480aa) is archived with the code.

**Outputs for reuse (aim A8).**

- **(a) An open benchmark of calibrated and miscalibrated predictions.**
  - The full per-replicate prediction and test dataset of the simulation (the 58 scenarios and the random designs): for every evaluation record, p\*, Y, the covariates where stored, and the p-hat and the exact-null Y\* of every arm, as float32 parquet; and every row of the test layer.
  - It is deposited on Zenodo under **CC BY 4.0** as its own record, with its own DOI, so that others can test a new calibration method or test on exactly the same predictions, with the truth known.
  - It comes with a **loader** (`code/benchmark_loader.py`: wide and long views of a replicate, iteration over a family, the study's own test rows) and a **data dictionary** of every column and metadata field, with the design tables `scenarios.csv` and `random_designs.csv`.
- **(b) A calibration audit in one call.**
  - R: `ebrahim.gof::run.all.external()` followed by `localize.external()`, packaged as the recipe `R/audit_calibration.R` (`audit_calibration(y, p, X)`).
  - Python: `code/audit_calibration.py`, a command-line tool and function that takes y and p (and optionally X) from any model and runs the R recipe through a temporary file; rpy2 is not needed.
  - Both are in the code archive, with an example on one benchmark replicate.
- **(c) A minimum-n-for-calibration table** by event rate for each model (Section 11, primary output 4), deposited as a csv beside the paper's table.
- **(d) A TRIPOD+AI alignment table**: each TRIPOD+AI item (Collins et al. 2024) that applies to the real-data analyses (Section 10.5), with the section of the paper or supplement that answers it, or the reason it does not apply.

**What is never archived.**

- **TabPFN weights.** TabPFN-3.5 is used under the TABPFN-3.5 License v1.0, which allows non-commercial research use only and does not allow redistribution. No archive of this study holds any TabPFN weights. The archive holds instead `code/get_tabpfn_weights.py`, which prints the licence notice, downloads the checkpoints from Prior Labs after the users have accepted the licence themselves, and checks their SHA-256 against Section 13. The other checkpoints are referenced by repository, file name, revision and SHA-256.
- **Real patient data.** They are not redistributed. The repository gives loaders and access instructions. Derived per-patient predictions on real data are deposited only for data sets whose licence allows redistribution with attribution (Sepsis: CC BY 4.0; PhysioNet 2012: ODC-By 1.0). For IST and CRASH-2, only aggregate results are deposited.
- **A preprint** is posted when the results and the manuscript are final.

---

## 15. References

- Austin PC, Steyerberg EW (2019). The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. *Statistics in Medicine* 38:4051-4065. doi:10.1002/sim.8281
- Collins GS, et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 385:e078378. doi:10.1136/bmj-2023-078378
- Ebrahim EK, Hussein OA, El-Kotory A (2026). [EDGE] A grouped calibration test for logistic regression that tolerates a few corrupted records. arXiv:2608.20511. doi:10.48550/arXiv.2608.20511
- Hollmann N, Müller S, Purucker L, Krishnakumar A, Körfer M, Hoo SB, Schirrmeister RT, Hutter F (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637:319-326. doi:10.1038/s41586-024-08328-6
- Morris TP, White IR, Crowther MJ (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine* 38:2074-2102. doi:10.1002/sim.8086
- Nattino G, Finazzi S, Bertolini G (2014). A new calibration test and a reappraisal of the calibration belt for the assessment of prediction models based on dichotomous outcomes. *Statistics in Medicine* 33:2390-2407. doi:10.1002/sim.6100
- Osius G, Rojek D (1992). Normal goodness-of-fit tests for multinomial models with large degrees of freedom. *Journal of the American Statistical Association* 87:1145-1152. doi:10.1080/01621459.1992.10476271
- Perel P, et al. (2012). Predicting early death in patients with traumatic bleeding: development and validation of prognostic model. *BMJ* 345:e5166. doi:10.1136/bmj.e5166
- Prior Labs (2026). TabPFN-3.5 model card and TABPFN-3.5 License v1.0. Hugging Face repository Prior-Labs/tabpfn_3_5 (checkpoint tabpfn-v3.5-20260909).
- Puhr R, Heinze G, Nold M, Lusa L, Geroldinger A (2017). Firth's logistic regression with rare events: accurate effect estimates and predictions? *Statistics in Medicine* 36:2302-2317. doi:10.1002/sim.7273
- Qu J, Holzmüller D, Varoquaux G, Le Morvan M (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv:2602.11139.
- Riley RD, et al. (2020). Calculating the sample size required for developing a clinical prediction model. *BMJ* 368:m441. doi:10.1136/bmj.m441
- Riley RD, et al. (2021). Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Statistics in Medicine* 40:4230-4251. doi:10.1002/sim.9025
- Riley RD, Collins GS (2023). Stability of clinical prediction models developed using statistical or machine learning methods. *Biometrical Journal* 65:2200302. doi:10.1002/bimj.202200302
- Siepe BS, Bartoš F, Morris TP, Boulesteix A-L, Heck DW, Pawel S (2024). Simulation studies for methodological research in psychology: a standardized template for planning, preregistration, and reporting. *Psychological Methods*. doi:10.1037/met0000695
- Stukel TA (1988). Generalized logistic models. *Journal of the American Statistical Association* 83:426-431. doi:10.1080/01621459.1988.10478613
- Tao Y, Zhang X, Liu X, Han B, Maddix D, et al. (2026). Mitra-v2 technical report. arXiv:2609.04540.
- Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW (2016). A calibration hierarchy for risk models was defined: from utopia to empirical data. *Journal of Clinical Epidemiology* 74:167-176. doi:10.1016/j.jclinepi.2015.12.005

**Checks.**

- The journal DOIs were checked on Crossref: on 2026-10-06, and on 2026-10-07 for Osius-Rojek, Stukel, Hollmann et al. and Siepe et al.
- The arXiv titles and authors were checked on their arXiv records on 2026-10-07.
- The dataset DOIs were checked on DataCite on 2026-10-06.

---

## Appendix A. Scenario table

Produced by `code/dgp.py build` (`code/scenarios.csv`):

- kappa and alpha are solved on 10^6 fixed draws;
- the realized prevalence equals pi to 2e-15 in every row;
- EPV = n pi / p;
- the Riley minimum n comes from R `pmsampsize` (shrinkage 0.9, Cox-Snell R^2 from the oracle AUC).

The scenario name gives the family, the event rate (pi03 = 3 %) and n. Shapes, links and gamma are in Sections 3.2 and 3.6.

| id | scenario | N_eval | kappa | alpha | oracle AUC | EPV | Riley min n |
|---|---|---|---|---|---|---|---|
| 1 | S1_pi03_n200 | 3334 | 0.7327 | -4.4586 | 0.810 | 0.75 | 1662 |
| 2 | S1_pi03_n500 | 3334 | 0.7327 | -4.4586 | 0.810 | 1.88 | 1662 |
| 3 | S1_pi03_n1000 | 3334 | 0.7327 | -4.4586 | 0.810 | 3.75 | 1662 |
| 4 | S1_pi03_n3000 | 3334 | 0.7327 | -4.4586 | 0.810 | 11.25 | 1662 |
| 5 | S1_pi10_n200 | 1000 | 0.7327 | -3.0255 | 0.800 | 2.50 | 609 |
| 6 | S1_pi10_n500 | 1000 | 0.7327 | -3.0255 | 0.800 | 6.25 | 609 |
| 7 | S1_pi10_n1000 | 1000 | 0.7327 | -3.0255 | 0.800 | 12.50 | 609 |
| 8 | S1_pi10_n3000 | 1000 | 0.7327 | -3.0255 | 0.800 | 37.50 | 609 |
| 9 | S1_pi30_n200 | 1000 | 0.7327 | -1.3603 | 0.788 | 7.50 | 323 |
| 10 | S1_pi30_n500 | 1000 | 0.7327 | -1.3603 | 0.788 | 18.75 | 323 |
| 11 | S1_pi30_n1000 | 1000 | 0.7327 | -1.3603 | 0.788 | 37.50 | 323 |
| 12 | S1_pi30_n3000 | 1000 | 0.7327 | -1.3603 | 0.788 | 112.50 | 323 |
| 13 | S2_pi03_n200 | 3334 | 0.6230 | -4.4357 | 0.827 | 0.75 | 1454 |
| 14 | S2_pi03_n500 | 3334 | 0.6230 | -4.4357 | 0.827 | 1.88 | 1454 |
| 15 | S2_pi03_n1000 | 3334 | 0.6230 | -4.4357 | 0.827 | 3.75 | 1454 |
| 16 | S2_pi03_n3000 | 3334 | 0.6230 | -4.4357 | 0.827 | 11.25 | 1454 |
| 17 | S2_pi10_n200 | 1000 | 0.6230 | -2.9203 | 0.800 | 2.50 | 609 |
| 18 | S2_pi10_n500 | 1000 | 0.6230 | -2.9203 | 0.800 | 6.25 | 609 |
| 19 | S2_pi10_n1000 | 1000 | 0.6230 | -2.9203 | 0.800 | 12.50 | 609 |
| 20 | S2_pi10_n3000 | 1000 | 0.6230 | -2.9203 | 0.800 | 37.50 | 609 |
| 21 | S2_pi30_n200 | 1000 | 0.6230 | -1.2353 | 0.769 | 7.50 | 349 |
| 22 | S2_pi30_n500 | 1000 | 0.6230 | -1.2353 | 0.769 | 18.75 | 349 |
| 23 | S2_pi30_n1000 | 1000 | 0.6230 | -1.2353 | 0.769 | 37.50 | 349 |
| 24 | S2_pi30_n3000 | 1000 | 0.6230 | -1.2353 | 0.769 | 112.50 | 349 |
| 25 | S3_pi03_n200 | 3334 | 0.6142 | -4.4168 | 0.814 | 0.75 | 1612 |
| 26 | S3_pi03_n500 | 3334 | 0.6142 | -4.4168 | 0.814 | 1.88 | 1612 |
| 27 | S3_pi03_n1000 | 3334 | 0.6142 | -4.4168 | 0.814 | 3.75 | 1612 |
| 28 | S3_pi03_n3000 | 3334 | 0.6142 | -4.4168 | 0.814 | 11.25 | 1612 |
| 29 | S3_pi10_n200 | 1000 | 0.6142 | -2.9651 | 0.800 | 2.50 | 609 |
| 30 | S3_pi10_n500 | 1000 | 0.6142 | -2.9651 | 0.800 | 6.25 | 609 |
| 31 | S3_pi10_n1000 | 1000 | 0.6142 | -2.9651 | 0.800 | 12.50 | 609 |
| 32 | S3_pi10_n3000 | 1000 | 0.6142 | -2.9651 | 0.800 | 37.50 | 609 |
| 33 | S3_pi30_n200 | 1000 | 0.6142 | -1.2956 | 0.783 | 7.50 | 323 |
| 34 | S3_pi30_n500 | 1000 | 0.6142 | -1.2956 | 0.783 | 18.75 | 323 |
| 35 | S3_pi30_n1000 | 1000 | 0.6142 | -1.2956 | 0.783 | 37.50 | 323 |
| 36 | S3_pi30_n3000 | 1000 | 0.6142 | -1.2956 | 0.783 | 112.50 | 323 |
| 37 | S4_pi03_n200 | 3334 | 0.6697 | -4.3699 | 0.798 | 0.75 | 1845 |
| 38 | S4_pi03_n500 | 3334 | 0.6697 | -4.3699 | 0.798 | 1.88 | 1845 |
| 39 | S4_pi03_n1000 | 3334 | 0.6697 | -4.3699 | 0.798 | 3.75 | 1845 |
| 40 | S4_pi03_n3000 | 3334 | 0.6697 | -4.3699 | 0.798 | 11.25 | 1845 |
| 41 | S4_pi10_n200 | 1000 | 0.6697 | -3.0419 | 0.800 | 2.50 | 609 |
| 42 | S4_pi10_n500 | 1000 | 0.6697 | -3.0419 | 0.800 | 6.25 | 609 |
| 43 | S4_pi10_n1000 | 1000 | 0.6697 | -3.0419 | 0.800 | 12.50 | 609 |
| 44 | S4_pi10_n3000 | 1000 | 0.6697 | -3.0419 | 0.800 | 37.50 | 609 |
| 45 | S4_pi30_n200 | 1000 | 0.6697 | -1.5964 | 0.814 | 7.50 | 323 |
| 46 | S4_pi30_n500 | 1000 | 0.6697 | -1.5964 | 0.814 | 18.75 | 323 |
| 47 | S4_pi30_n1000 | 1000 | 0.6697 | -1.5964 | 0.814 | 37.50 | 323 |
| 48 | S4_pi30_n3000 | 1000 | 0.6697 | -1.5964 | 0.814 | 112.50 | 323 |
| 49 | T1_pi10_n500 | 1000 | 0.8205 | -3.0442 | 0.800 | 6.25 | 609 |
| 50 | T1_pi10_n3000 | 1000 | 0.8205 | -3.0442 | 0.800 | 37.50 | 609 |
| 51 | T2_pi10_n500 | 1000 | 0.7327 | -3.0255 | 0.800 | 1.67 | 2284 |
| 52 | T2_pi10_n3000 | 1000 | 0.7327 | -3.0255 | 0.800 | 10.00 | 2284 |
| 53 | T3_pi10_n500 | 1000 | 0.5992 | -2.9981 | 0.800 | 4.17 | 914 |
| 54 | T3_pi10_n3000 | 1000 | 0.5992 | -2.9981 | 0.800 | 25.00 | 914 |
| 55 | T4a_pi10_n500 | 1000 | 0.6762 | -2.9646 | 0.800 | 6.25 | 609 |
| 56 | T4a_pi10_n3000 | 1000 | 0.6762 | -2.9646 | 0.800 | 37.50 | 609 |
| 57 | T4b_pi10_n500 | 1000 | 0.5297 | -2.8686 | 0.800 | 6.25 | 609 |
| 58 | T4b_pi10_n3000 | 1000 | 0.5297 | -2.8686 | 0.800 | 37.50 | 609 |

## Appendix B. The random designs

Produced by `code/dgp_random.py build --D 1000` (`code/random_designs.csv`, one row per design with its full specification). Every design was solved: the realized prevalence equals pi to 6e-14 and the oracle AUC equals its target to 2e-10, on 200,000 calibration draws per design.

| Factor | min | 10 % | median | 90 % | max |
|---|---|---|---|---|---|
| event rate pi | 0.020 | 0.027 | 0.088 | 0.290 | 0.400 |
| development n | 150 | 200 | 738 | 2732 | 3993 |
| predictors p | 4 | 6 | 17 | 28 | 30 |
| correlation rho | 0.00 | 0.06 | 0.29 | 0.53 | 0.60 |
| oracle AUC | 0.650 | 0.673 | 0.774 | 0.870 | 0.900 |
| w_nl | 0.00 | 0.10 | 0.51 | 0.91 | 1.00 |
| w_int | 0.00 | 0.11 | 0.51 | 0.92 | 1.00 |
| w_link | 0.00 | 0.10 | 0.50 | 0.90 | 1.00 |
| discrete fraction | 0.00 | 0.05 | 0.24 | 0.44 | 0.50 |
| N_eval | 1000 | 1000 | 1130 | 3730 | 4996 |
| expected development events (n pi) | 3.2 | 12.6 | 67.6 | 369.5 | 1255.4 |
| EPV | 0.12 | 0.59 | 4.01 | 26.11 | 187.37 |
| share of p\* >= 0.5 | 0.000 | 0.000 | 0.014 | 0.165 | 0.337 |

- The 1,000 designs hold 3,111 binary and 1,052 categorical predictors; 594 designs have at least one categorical predictor.
- 658 designs have p > 12, so DeepGOF-1 uses its axes reading there.
- TabPFN-3.5 runs on the GPU in 821 designs and on a CPU worker in 179.
- Mitra-v2 is recorded as "does not fit in 6 GB GPU memory" in 170 designs.
- In 722 designs n is below Riley's minimum n, and 67 designs expect fewer than 10 development events.
