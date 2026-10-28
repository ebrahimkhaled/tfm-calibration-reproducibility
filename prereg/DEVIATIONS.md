# Deviations from the pre-registration

Registration: OSF Registries, https://osf.io/8hmtz, DOI 10.17605/OSF.IO/8HMTZ, registered 2026-10-08 14:25 (public). The deposited copy of this file is the frozen one; this project copy is the live log.

This file starts empty at the freeze. Changes decided before the freeze are not deviations; they are listed in Section 0.2 of the pre-registration. Every change made after the freeze is added below, newest last, in the format of the template. See Section 12 of the pre-registration for what counts as minor or major.

## Template

### D-NN (YYYY-MM-DD) — short title

- **Type:** minor / major
- **What changed:**
- **Why:**
- **Had any result been seen when this was decided?** yes / no (say which)
- **Expected effect on the results:**
- **Parts rerun:**

---

### D-01 (2026-10-11) — CRASH-2: hospitals held out instead of a temporal split (major)

- **Type:** major (originally recorded as minor) (one real-data split; the simulation is unaffected). Reclassified as **major** on 2026-10-27 under Section 12 of the protocol (it changes a real-data split); results that depend on it are labelled exploratory in the paper.
- **What changed:** Section 10.1 fixed a temporal split for CRASH-2 (randomised before 2008-01-01 = development, later = external). The data file released through freeBIRD (CRASH-2_Data_anonymised_dataset_w.csv, 20,207 patients, 43 variables) carries no dates, no country and no treatment code, so that split cannot be formed. It is replaced by a hospital-held-out split: the 270 hospitals (iSiteID) that remain after the outcome exclusion are put in a random order with the pre-registered CRASH-2 seed (20261007), and whole hospitals are taken until at least 30 % of patients are in the external region. Result: 74 held-out hospitals; development 13,555 patients (2,035 deaths), external 6,048 (815 deaths), with 500 of them the local recalibration sample; no hospital is in both. The outcome is defined from daysrandtodeath (death within 28 days); patients with neither a discharge status nor a recorded death (83) are excluded, as patients without an outcome form were in the plan. The 3,076 deaths within 28 days match the published trial total.
- **Why:** the pre-registered split needs randomisation dates, which the released file does not contain. Holding out whole hospitals is a standard external (geographic) validation and is fully reproducible from the file and the seed.
- **Had any result been seen when this was decided?** No. No model had been fitted to CRASH-2; the decision used only the file's column list, site sizes and the outcome counts.
- **Expected effect on the results:** the external region now differs from development by hospital rather than by calendar time, so it tests transport across centres. The paper states this split and this deviation.
- **Parts rerun:** none (the CRASH-2 analysis had not started).
- **Correction (2026-10-23):** the hospitals put in random order are the 270 that remain after the outcome exclusion (the file lists 274 recruiting hospitals); 74 are held out. The split itself is unchanged (code/realdata.py, build_crash2_freebird).


### D-02 (2026-10-12) - CRASH-2 data terms: correction of the registered licence statement

- **Type:** minor (a factual correction; no analysis changes).
- **What changed:** Section 10.1 of the registered protocol describes CRASH-2 as available "CC BY as stated by freeBIRD". That is wrong. The data are released under the freeBIRD Data Sharing Agreement, which requires: confidential and secure storage; no transmission to other parties; no attempt to identify patients or hospitals; acknowledgement of the CRASH-2 Trial Collaborators; submission of the final publication and the analysis code to the LSHTM Clinical Trials Unit; and the permission of the trial's Chief Investigators before results are publicised. Consequences: (i) no CRASH-2 per-patient data or per-patient predictions are deposited in the benchmark release (only aggregate results, as the protocol already planned for this data set); (ii) hospital identifiers are used only to form the split and no per-hospital result is reported; (iii) the CRASH-2 results enter the paper only with the Chief Investigators' written permission; if permission is not given, the real-data analysis is reported for the other three data sets, as Section 10.1 already allows when CRASH-2 is unavailable.
- **Why:** the agreement text became available to the author only on registration with freeBIRD (2026-10-11).
- **Had any result been seen when this was decided?** No. No model had been fitted to CRASH-2.
- **Expected effect on the results:** none on the simulation; CRASH-2 may drop out of the real-data part.
- **Parts rerun:** none.
- **Update (2026-10-23):** Permission to publish granted by the CRASH-2 Chief Investigators (reported by the author 2026-10-23; written confirmation filed with the author). CRASH-2 aggregate results therefore enter the paper. The other terms still hold: aggregate results only, no per-hospital result, no per-patient data or predictions in any deposit (results/realdata/crash2_death28/ keeps its DO_NOT_DEPOSIT marker and stays on the deposit exclusion list).


### D-03 (2026-10-16) - Real data: Mitra-v2 not run, and TabPFN-3.5 on CPU, where the 6 GB card cannot hold the model (major)

- **Type:** major (originally recorded as minor) (three real-data cells; the simulation is unaffected). Reclassified as **major** on 2026-10-27 under Section 12 of the protocol (it changes a real-data arm set and device); results that depend on it are labelled exploratory in the paper.
- **What changed:** Section 5.2 records Mitra-v2 as "not run: does not fit in 6 GB GPU memory" only at T2_pi10_n3000 (30 covariates, n = 3000), and Section 9.3 runs TabPFN-3.5 on CPU workers for the scenario groups the card cannot hold. The same two rules are now applied to the real-data cells that have the same problem: (i) Mitra-v2 is recorded as not run at n = 3000 for IST (both outcomes; 21 predictor columns) and PhysioNet 2012 (31 columns), in the learning curves and the plasmode tier; (ii) TabPFN-3.5 and its T = 0.9 arm run on a CPU worker (6 threads) at PhysioNet n = 3000. The device is recorded per replicate, as in the simulation. Mitra-v2 runs at every other real-data cell, including n = 3000 for CRASH-2 and Sepsis (8 and 3 columns).
- **Why:** measured in the real-data smoke test (results/smoke/realdata, 2026-10-16). At those cells AutoGluon cut Mitra-v2's context to 1,024 of 3,000 development rows after a GPU out-of-memory error even when only 500 rows were predicted per call, and the protocol never accepts a cut context. TabPFN-3.5 ran out of GPU memory at PhysioNet n = 3000 with 500 predicted rows per call; on a CPU worker it took 203 s per fit.
- **Had any result been seen when this was decided?** No. Only run status, timings and memory were looked at; no calibration metric had been computed for these cells.
- **Expected effect on the results:** Mitra-v2 has no learning-curve or plasmode point at n = 3000 for IST and PhysioNet (reported as "not run", as for T2 in the simulation). CPU and GPU predictions of TabPFN-3.5 differ by at most 2.6e-4 on the same replicate (Section 5.2).
- **Parts rerun:** none (the real-data stream had not started).
- **Correction (2026-10-28):** IST has 20 predictor columns, not 21 (code/realdata.py, build_ist: 3 continuous, 14 binary and 3 categorical). Section 10.1 of the protocol lists exactly these 20 variables but calls them "the 21 baseline variables"; the count was wrong, the list is right. The TFMs receive these 20 columns. Nothing in the analysis changes.

### D-04 (2026-10-16) - Real data: implementation details that Section 10 leaves open

- **Type:** minor (implementation; no estimand, arm, split or replicate count changes).
- **What changed:** the real-data runner (code/run_realdata.py, R/realdata_gof.R) fixes the following.
  1. **Regression arms (LR-MLE, FLIC, ridge, GAM).** Categorical predictors are passed as treatment dummies (the design R builds from a factor; identical fits when every level is present). A design column that is constant in the development subsample, or a linear combination of earlier columns (what glm reports as aliased), is dropped, so its coefficient is 0. Example: IST stroke subtype "other" (57 of 19,435 patients) is absent from about half of the n = 200 subsamples; its external rows are then predicted like the reference subtype. PhysioNet's missing-value indicators of minimum and maximum respiratory rate are identical, so the second is dropped. Without this, glm, logistf and mgcv cannot predict an unseen level, and logistf stops on a singular information matrix. The TFMs and LightGBM keep the categorical columns unchanged.
  2. **Distinct rows are predicted once.** Every arm predicts each distinct external covariate row once, and the value is copied to its duplicates (Sepsis: fewer than 1,000 covariate patterns among 19,051 external rows). A fitted model is a fixed function of the row, so the predictions are the same.
  3. **Out-of-memory chunking.** A TFM predicts the whole external region in one call. Only after a CUDA out-of-memory error does it predict in halving chunks (same seed, same fit; at least 500 rows). Chunking changed TabICLv2 predictions by at most 1e-6 and TabPFN-3.5 by at most 1e-3; Mitra-v2 changed by up to 0.045, its documented batch dependence under emulated bfloat16 (Section 5.2). The chunk size is recorded per arm and replicate.
  4. **Infrastructure errors** (CUDA runtime errors, a crashed Rscript, failed host-memory allocations, and an R test that cannot allocate memory) are not model failures (Section 8.2). The arm or test is left unrecorded and computed again; it is never stored as a failure or as "Not run".
  5. **Localization covariates (10.3)**, built on the external region:
     - age band: quartiles of age;
     - sex;
     - region (IST only): four groups of the external countries — Ireland and the Nordic countries with the Netherlands (the pre-registered fallback pool), the rest of Western and Southern Europe, Central and Eastern Europe with Turkey, and countries outside Europe;
     - severity: IST conscious state; CRASH-2 GCS; PhysioNet SAPS-I. Sepsis has no severity score, so the septic episode number stands in for it.

     Missing values take the external median (PhysioNet: 4 sex and 166 SAPS-I values). Categoricals enter as dummies. Le Cessie and DeepGOF-1 use the full predictor matrix the arms saw, as in the simulation; DeepGOF-1 uses the axes reading above 12 predictors (IST, PhysioNet).
  6. **Battery (10.2).** Run as in the simulation (Section 5.3): on Y, and on each arm's exact null Y* ~ Bernoulli(p-hat); DeepGOF-1 (B = 99) on Y only; the ladder columns on Y only.
  7. **Ladder (10.3).** Both local sizes of Section 5.5 are used: m = 500 (the whole local sample) and m = 100 (its first 100 rows). Only the learning curves have a ladder, because 10.4 does not ask for repair in the plasmode tier.
  8. **Plasmode (10.4).** The development subsample has its own seed (key, prevalence, n, replicate), separate from the learning curves. Y* for replicate r is drawn once over the whole table (realdata.plasmode_outcomes), so the four n of one replicate share their evaluation outcomes. The oracle arm (p\*) is included.
  9. **Arms.** All arms of Section 5.2 except the oracle (10.2), including the three side arms: TabPFN-3.5 at T = 0.9, TabICLv2 at T = 1.0, and Real-TabPFN-2. The main arms run first.
  10. **Empirical calibration class (10.2).** The rules of 4.5 are applied to the empirical estimands with the absolute thresholds; no real-data outcome is near the 3 % event rate. Reported as empirical, never as true.
  11. **IST loader.** RDATE is written as Polish month abbreviations ("sty-91" ... "gru-96"), which the loader did not parse. Fixed. This affects only the temporal sensitivity split, whose development pool is then 3,251 (death at 14 days) and 3,236 (dead or dependent) complete cases. That is below the 3,500 rule, and the protocol gives a fallback only for the UK pool (decision for the author). The primary geographic split is unaffected.
- **Why:** Section 10 does not specify these points. Items 1 and 3 were forced by failures seen in the smoke test.
- **Had any result been seen when this was decided?** No calibration result. Only fit status, error messages, timings and memory from the smoke test were used.
- **Expected effect on the results:** none intended.
- **Parts rerun:** none.

### D-05 (2026-10-23) - Real data: side arms dropped, plasmode foundation models at R = 50 (major)

- **Type:** major (originally recorded as minor) (real-data part only; Section 10.5 makes the real-data results descriptive and supportive; the simulation is unaffected). Reclassified as **major** on 2026-10-27 under Section 12 of the protocol (it changes a real-data arm set and replicate count); results that depend on it are labelled exploratory in the paper.
- **What changed:** (i) The three side arms of Section 5.2 are dropped on all real data, in the learning curves and the plasmode tier: TabPFN-3.5 at T = 0.9, TabICLv2 at T = 1.0, and Real-TabPFN-2. Section 10.2 had asked for every arm except the oracle. Side-arm fits written before the decision (part of the learning curves) stay in the files but are not scored, tested or reported. (ii) In the plasmode tier (Section 10.4), the three main foundation models (TabPFN-3.5, TabICLv2, Mitra-v2) run at R = 50 per n (replicates 1-50) instead of R = 200. The classical arms and the oracle had already been fitted at R = 200; paired comparisons use replicates 1-50, and the classical summaries are also reported over all 200. (iii) As D-04 item 11 anticipated, the IST temporal sensitivity split is run with the complete cases available: development pools of 3,251 (death at 14 days) and 3,236 (dead or dependent at 6 months), below the 3,500 rule. It uses the main arms only, B = 100, and is declared as such. Its goodness-of-fit battery is run on the R0 predictions only (repair, 10.3, belongs to the primary split; its ladder columns are scored but not tested), to keep the shared CPU budget. The learning curves of the main arms keep B = 100 on every data set, and their repair and localization analyses are unchanged.
- **Why:** GPU budget. The full real-data plan needs about 400 GPU-hours on the single 6 GB card (measured per replicate in the smoke test, 2026-10-16): about 115 hours for the learning curves with all arms, and about 290 hours for the plasmode tier with all arms at R = 200. The real-data results are supportive under Section 10.5, and the questions the side arms answer (temperature and model version) are answered by the simulation, where these arms were run as pre-registered. With these changes the remaining GPU work is about 30 hours for the plasmode tier and about 30 hours for the temporal split.
- **Had any result been seen when this was decided?** No real-data calibration result had been examined; the decision used run status and timings only.
- **Expected effect on the results:** the side arms have no real-data rows. The plasmode Monte Carlo standard errors of the foundation models are twice those planned (sqrt(200/50) = 2). The plasmode tier enters no label of Section 7, so its role as a secondary check is unchanged.
- **Parts rerun:** none; the remaining work follows the new plan. D-03 and D-04 were confirmed by the author on 2026-10-23, together with two defaults: GAM on Sepsis stays as registered (s(x, k = 5); a subsample with fewer than five distinct episode values is a failed fit and is counted), and the IST temporal split is run as described above.

### D-06 (2026-10-28) - Real data: three registered sensitivity analyses not run (major)

- **Type:** major (real-data part only; the simulation is unaffected), because it removes registered analyses (two splits and a missing-data analysis), the same kind of change for which D-01 was classed major under Section 12.
- **What changed:** Section 10.1 registered three real-data sensitivity analyses. None of them was run; no fit, score or test exists for any of them.
  - (a) Sepsis: "a random half split", as a check on the primary split (non-Sepsis-3 admissions against Sepsis-3 admissions).
  - (b) PhysioNet 2012: "coronary-care, cardiac-surgery and surgical ICUs against the medical ICU", as a check on the primary split (sets A + B against set C).
  - (c) Missing data: "A sensitivity analysis keeps the incomplete rows (IST, CRASH-2) or uses the TFMs' native missing-value handling (PhysioNet)."

  The only real-data sensitivity analysis that was run is the IST temporal split (D-04 item 11, D-05 (iii)).
- **Why:** computing time. Section 10.5 makes the real-data results descriptive and supportive; they enter no label of Section 7. All real-data work ran on the single 6 GB GPU machine, which had repeated NVIDIA driver errors (nvlddmkm event 153 on 2026-10-25 and 2026-10-27), a driver power-state bugcheck (0x9F) on 2026-10-23 and several restarts during the real-data run (logs/main/INCIDENTS.md). The remaining time was used to finish the primary splits, the plasmode tier and the IST temporal split.
- **Had any result been seen when this was decided?** Yes. The decision was made on 2026-10-28, after the main real-data results (learning curves, plasmode tier and IST temporal split) were available and had been seen.
- **Expected effect on the results:** no reported number changes. The real-data conclusions rest on the primary splits only (and, for IST, also on the temporal split). Whether they hold under a random split of the Sepsis data, under a case-mix split of the PhysioNet data, and with incomplete rows kept or with the TFMs' own missing-value handling is not known. These checks remain open. The released code already forms both splits (columns split_random and split_icu, code/realdata.py) and keeps the incomplete-row tables of IST and CRASH-2 (*_all.parquet), so the checks can be run later from the released code; the runner (code/run_realdata.py) needs each new split or table added to its list of analysis parts (PART_SPLIT). Section 12 asks for a pre-registered analysis to be reported where it can be run; these three could be run but were not, and the paper says so.
- **Parts rerun:** none.

### D-07 (2026-10-28) - Localization on the Sepsis data not run; main-text display plan adapted to eight items (major)

- **Type:** major, because part (a) drops a registered analysis on one data set, the same kind of change as D-06. Part (b) alone would be minor: it changes where results are shown, not what is computed.
- **What changed:** (a) Section 10.3 registered `localize.external()` on the external evaluation set for FLIC and the TFMs. It was run for IST, CRASH-2 and PhysioNet 2012 but not for the Sepsis data; no localization row exists for Sepsis. Section 10.3 names a severity score for the other three data sets (conscious state, GCS, SAPS-I) but none for Sepsis, whose released file holds only age, sex and episode number. (b) Section 11 planned ten main-text figures and tables. The target journal allows eight display items, so the plan was adapted: b* against n is reported in the Supplementary Information (the true-risk estimand table, by n, and the distribution of b* in Supplementary Fig. S1), as are the size table (every test on every arm) and the real-data localization results (Table S18d); the design schematic is replaced by the scenario table in the Supplementary Information; separate oracle calibration curves are not drawn, because the true ICI against n (main-text Fig. 1) and the true-risk estimand tables summarise the same distance between predicted and true risk. A figure on calibration-in-the-large and the softmax temperature, not in the plan, is in the main text because it carries the TabICLv2 finding of Q1.
- **Why:** (a) the Sepsis covariates were not run through localization when the real-data run was finished under the computing limits described in D-06; with no registered severity score, the covariate set would have been age band and sex only. (b) the journal's limit on display items.
- **Had any result been seen when this was decided?** Yes, for both parts: the decision was recorded on 2026-10-28, after the main real-data results had been seen. Part (b) followed the journal's limit and was applied while the manuscript was written; it is logged here because it had not been recorded before.
- **Expected effect on the results:** no reported number changes. Whether the shift in event rate of the Sepsis data (4.9 % against 19 %) would be localized to INTERCEPT, as it was in IST death or dependency, is not known. Every planned display item that was computed is reported in the Supplementary Information.
- **Parts rerun:** none.

## Analysis implementation choices (recorded 2026-10-11, before any foundation-model result was examined)

Not deviations: the protocol leaves these details open. They are listed in full in analysis/README.md, section "Choices the protocol leaves open" (items 1-15), and were accepted by the author's standing instruction on 2026-10-11. Item 15 runs the pre-registered gam(..., family = scat(), method = "REML") meta-model with mgcv's optimizer = "efs", which maximises the same REML criterion; the faster bam approximation is used only for interim and test runs and is never reported.
