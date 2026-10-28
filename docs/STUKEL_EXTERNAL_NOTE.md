# Why Stukel (offset) in `run.all.external()` is liberal

**Resolved the same day: from commit 19480aa the row is the score test, which holds its size (section 7).**

Date: 2026-10-07. A read-only diagnosis: `ebrahim.gof` was not edited.
- Source read: `R/run_all_external.R` at commit 2c77f3c (release-2.10.0 worktree, `git show HEAD:...`).
- Script: `R/stukel_diagnosis.R`.
- Output: `results/stukel_diagnosis.csv`, log `logs/stukel_diagnosis.log`.

## 1. What the row computes

With frozen predictions p and lp = logit(p), the row fits

    glm(y ~ 0 + za + zb, offset = lp, family = binomial)
    za = 0.5 lp^2 for lp >= 0 (else 0),   zb = -0.5 lp^2 for lp < 0 (else 0)

and refers the likelihood-ratio statistic (null deviance minus the fitted deviance) to chi-square(2). No intercept or slope is fitted, which is right for frozen predictions.

- The **za** term lives only on records with p >= 0.5.
- The **zb** term lives on records with p < 0.5.

## 2. What was simulated

- **Null.** The predictions are the true risks, y ~ Bernoulli(p), so the null holds exactly.
- **Risks.** logit p = alpha + 1.2 z, where z is either:
  - normal (oracle AUC about 0.80, as in the main design), or
  - t on 3 df scaled to unit variance (heavy tails).
- **Event rate and size.** alpha is solved for pi = 3 %, 10 % and 30 %, at the main design's evaluation sizes (N = 3,334 at 3 %; 1,000 otherwise).
- **Replicates.** 4,000 per cell; the MCSE of a size near 0.05 is 0.0034.
- **Re-implementation check.** The script's LR2 equals the package row to 1e-8 in every cell (one draw per cell, asserted).

Four statistics are compared:
- LR2: the package's statistic, chi-square(2).
- Score2: the score test of the same two terms at the null, U' I^-1 U, with U = Z'(y - p) and I = Z' diag(p(1-p)) Z. No fit is needed.
- LRa: the za term on its own, chi-square(1).
- LRb: the zb term on its own, chi-square(1).

## 3. Results (rejection rate at 0.05; MCSE 0.0034)

| risk tails | pi | N | median records with p >= 0.5 | LR2 (package) | Score2 | LRa (za alone) | LRb (zb alone) | no valid statistic (NA) / glm not converged |
|---|---|---|---|---|---|---|---|---|
| normal | 3 % | 3,334 | 1 | 0.034 | 0.035 | 0.020 | 0.048 | 0 / 0 |
| normal | 10 % | 1,000 | 12 | **0.078** | 0.051 | **0.088** | 0.053 | 0 / 0.05 % |
| normal | 30 % | 1,000 | 183 | 0.051 | 0.049 | 0.051 | 0.055 | 0 / 0 |
| t3 | 3 % | 3,334 | 16 | **0.061** | 0.049 | **0.071** | 0.044 | 1.7 % / 1.0 % |
| t3 | 10 % | 1,000 | 17 | **0.057** | 0.053 | **0.063** | 0.054 | 0.7 % / 0.4 % |
| t3 | 30 % | 1,000 | 122 | 0.042 | 0.050 | 0.043 | 0.046 | 0.7 % / 0 |

This matches the earlier oracle size check (`results/size_check.csv`, 2,000 replicates per scenario):
- Stukel is liberal at pi = 10 % (S1: 0.0745) and in some pi = 3 % shapes (S3: 0.0765).
- It is on target at 30 %.
- It is 0.063 pooled at N = 3,334.

## 4. Diagnosis

1. **The excess comes from the za term (the upper tail, p >= 0.5), and only when few records carry it.**
   - When the event rate is low, only about 10-20 records have p >= 0.5. The za coefficient is then estimated from those few records, and its likelihood-ratio statistic is not chi-square(1).
   - Its mean is 1.34 at pi = 10 % (normal), not 1, and its size is 0.088.
   - With about 1 such record (normal, 3 %) the term is almost empty, and the test becomes conservative (0.020).
   - With more than 100 records (30 %) the chi-square reference is accurate.
   - The size therefore depends on how many records sit above p = 0.5. That count is set by the event rate and the tail of the risk distribution, which is why heavy tails move the problem to pi = 3 %.
2. **It is a likelihood-ratio problem, not a problem with the lp^2 terms.**
   - The score test of the same two terms holds its size in every cell (0.035-0.053). The 0.035 at pi = 3 % (normal) is the same near-empty za term.
   - The zb term alone holds its size everywhere (0.044-0.055), even though its values reach |zb| = 35 at 3 %.
   - The large values of lp^2 matter in one way only: under heavy tails, glm returns no valid statistic in up to 1.7 % of replicates and reports non-convergence in up to 1.0 %.
3. **Why the size check found it and the pilot did not.** The pilot pooled 6 scenarios x 20 replicates (MCSE 0.006 pooled), which cannot separate 0.06 from 0.05.

## 5. Consequences for this study (already in the pre-registration)

- Stukel (offset) is reported with its size beside its power in every table: the oracle nulls and Y\* of the main run.
- It is excluded from the detection ranking (H4b, the detection curves and the summary of findings). It is kept as a descriptive row.

## 6. Options for the package (for the author; nothing changed here)

- Use the score statistic (one line, no fit, no convergence failures). It holds its size at every event rate simulated here.
- Or drop za, with a note, when fewer than about 50 records have p >= 0.5, and refer zb alone to chi-square(1).
- Or give the row a Monte Carlo reference from Bernoulli(p), as `localize.external()` does.

## 7. Resolution (2026-10-07, commit 19480aa)

The author took the first option of section 6 into the package. `ebrahim.gof` 2.10.0 at commit **19480aa** (release-2.10.0 worktree, read only) changes one row of `run.all.external()`:

- **Stukel (offset) is now the score test** of the two terms at the null, U' I^- U with U = Z'(y - p) and I = Z' diag(p(1 - p)) Z. Nothing is fitted, so it cannot fail to converge.
- The columns are scaled to their own information first (lp^2 reaches the thousands where p(1 - p) is near zero), and the inverse keeps only the eigenvalues above 1e-10 of the largest (a rank-aware inverse).
- A tail with no records carries no term: when no prediction reaches 0.5 the test has **1 df**. The row's Note says "score test, terms added to logit(p)" or "score test, one tail has no records".
- The package NEWS gives the likelihood ratio's size as 0.078 at pi = 10 % and 0.061 at 3 % with heavy-tailed risks, and the author measured the score form at 0.037-0.055 (4,000 nulls per cell; pi 3/10/30 %; normal and t3 risks; no failures).

**Checked here.**

- **The build.** `R/verify_gof_build.R` compares every function of the installed namespace with the R sources of the commit (`git archive`): 189 functions, 0 differences at 19480aa, while against 2c77f3c `run.all.external` differs. The test layer now refuses any build whose code hash differs from 19480aa's (`GOF_CODE_HASH` in `R/gof_layer.R`; unit test).
- **The statistic.** The package row equals an independent score statistic (`R/tests/test_gof_layer.R`), and the re-implementation check of `stukel_diagnosis.R` now compares the package row with Score2.
- **The size**, with the package's own row, in the null design and with the seeds of section 2 (`R/stukel_score_size.R`, `results/stukel_score_size.csv`, 4,000 nulls per cell, MCSE about 0.0035):

| risk tails | pi | N | size (MCSE) | replicates with 1 df | replicates with p clamped |
|---|---|---|---|---|---|
| normal | 3 % | 3,334 | 0.035 (0.0029) | 37 % | 0 |
| normal | 10 % | 1,000 | 0.051 (0.0035) | 0 | 0 |
| normal | 30 % | 1,000 | 0.049 (0.0034) | 0 | 0 |
| t3 | 3 % | 3,334 | 0.048 (0.0034) | 0 | 1,139 |
| t3 | 10 % | 1,000 | 0.053 (0.0035) | 0 | 374 |
| t3 | 30 % | 1,000 | 0.050 (0.0034) | 0 | 281 |

- Every one of the 24,000 replicates gave a p-value. Under t3 risks some predictions are exactly 1 in double precision, which `run.all.external()` refuses; they were moved to 1 - 1e-9 first, as the test layer's `clamp_p()` does (the first run of the script, without that step, lost 1.7 % of the t3 replicates at 3 % to this input check, not to the test).
- The range is 0.035-0.053. The lowest value is the near-empty upper term at pi = 3 % with normal risks: conservative, as section 4 predicted, not liberal. The likelihood ratio's 0.078 and 0.061 are gone.

**What changes in the study** (pre-registration C11, C21 and Section 5.3):

- Stukel (offset) is no longer excluded from the detection ranking. It is in H4a with no exception, and in the detection curves and the summary tables, ranked like every other test.
- As for every test, its size on the main design comes from the oracle nulls and from each arm's Y*, beside its power.
- Sections 1-6 above describe the likelihood-ratio row of 2c77f3c and are kept as the record of why the row changed.
