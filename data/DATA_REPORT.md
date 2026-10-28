# Real-data report: sources, outcomes, predictors, splits

Date: 2026-10-06. Plan reference: `AUDIT_2026-10/REBUILD_PLAN.md` §5 (real data, plasmode tier) and §4.4 (local recalibration sample of 500).
Code: `code/realdata.py`. Raw files and hashes: `data/raw/`, `data/raw/MANIFEST_SHA256.txt`. Clean tables: `data/clean/<key>.parquet` with `<key>.meta.json` (flow, predictor types, missing counts, split counts, raw SHA256, table fingerprint).

Every number below for Sepsis and PhysioNet was produced by `realdata.py build` on the downloaded files. IST and CRASH-2 numbers wait for the author's downloads (`data/AUTHOR_STEPS.md`); the loaders were tested on mock files with the documented layout.

---

## 1. Status

| Source | Key(s) | Access | Status |
|---|---|---|---|
| International Stroke Trial | `ist_death14`, `ist_dd6` | Edinburgh DataShare, end-user licence (author) | loader ready, file awaited |
| CRASH-2 | `crash2_death28` | freeBIRD, registration (author) | loader ready, file awaited |
| Sepsis survival (UCI 827) | `sepsis_death` (+ `sepsis_korea`) | open, CC BY 4.0 | **built** |
| PhysioNet/CinC Challenge 2012 | `physionet_ihd` | open, ODC-By 1.0 | **built** |

## 2. Sources, licences and citations

All DOIs were resolved on Crossref or DataCite on 2026-10-06.

| Source | Licence | Cite (dataset) | Cite (paper) |
|---|---|---|---|
| IST | ODC-By v1.0 per DataCite; the DataShare item also carries an End-user Licence that the author must read | Sandercock P, Niewada M, Czlonkowska A. International Stroke Trial database (version 2). University of Edinburgh, 2011. doi:10.7488/ds/104 | International Stroke Trial Collaborative Group. The International Stroke Trial (IST): a randomised trial of aspirin, subcutaneous heparin, both, or neither among 19 435 patients with acute ischaemic stroke. *Lancet* 1997;349:1569–1581. doi:10.1016/S0140-6736(97)04011-7 · Sandercock PAG, Niewada M, Członkowska A, IST Collaborative Group. The International Stroke Trial database. *Trials* 2011;12:101. doi:10.1186/1745-6215-12-101 · Erratum *Trials* 2012;13:24. doi:10.1186/1745-6215-13-24 |
| CRASH-2 | CC BY 3.0 (stated by freeBIRD; author to confirm on download) | CRASH-2 Collaborators. CRASH-2 – Clinical Randomisation of an Antifibrinolytic in Significant Haemorrhage [dataset]. LSHTM, 2010. doi:10.17037/DATA.10 | CRASH-2 trial collaborators. Effects of tranexamic acid on death, vascular occlusive events, and blood transfusion in trauma patients with significant haemorrhage (CRASH-2). *Lancet* 2010;376:23–32. doi:10.1016/S0140-6736(10)60835-5 · Perel P, et al. Predicting early death in patients with traumatic bleeding: development and validation of prognostic model. *BMJ* 2012;345:e5166. doi:10.1136/bmj.e5166 · Roberts I, et al. *Health Technol Assess* 2013;17(10). doi:10.3310/hta17100 |
| Sepsis (UCI 827) | CC BY 4.0 | Chicco D, Jurman G. Sepsis Survival Minimal Clinical Records. UCI Machine Learning Repository, 2020. doi:10.24432/C53C8N | Chicco D, Jurman G. Survival prediction of patients with sepsis from age, sex, and septic episode number alone. *Sci Rep* 2020;10:17156. doi:10.1038/s41598-020-73558-3 · Knoop ST, Skrede S, Langeland N, Flaatten HK. Epidemiology and impact on all-cause mortality of sepsis in Norwegian hospitals: a national retrospective study. *PLoS One* 2017;12:e0187990. doi:10.1371/journal.pone.0187990 |
| PhysioNet 2012 | Open Data Commons Attribution v1.0, open access | Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012, v1.0.0, https://physionet.org/content/challenge-2012/1.0.0/ (no dataset DOI shown on the page) | Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality of ICU patients: the PhysioNet/Computing in Cardiology Challenge 2012. *Comput Cardiol* 2012;39:245–248 (PMID 24678516, PMCID PMC3965265; no DOI) · PhysioNet platform citation now requested by the page: Pollard T, et al. PhysioNet as a global platform for biomedical research. *Nature Health* 2026;1(8):792–795. doi:10.1038/s44360-026-00096-z · Classic platform citation: Goldberger AL, et al. *Circulation* 2000;101:e215–e220. doi:10.1161/01.CIR.101.23.e215 |

Other references used for design choices: Counsell C, et al. *Stroke* 2002;33:1041–1047, doi:10.1161/hs0402.105909; Weir NU, et al. Variations between countries in outcome after stroke in the IST. *Stroke* 2001;32:1370–1377, doi:10.1161/01.str.32.6.1370; Le Gall JR, et al. A simplified acute physiology score for ICU patients. *Crit Care Med* 1984;12:975–977, doi:10.1097/00003246-198411000-00012; TRIPOD+AI, Collins GS, et al. *BMJ* 2024;385:e078378, doi:10.1136/bmj-2023-078378.

None of these four datasets appears in the author's corpus (plan §5).

## 3. Outcomes, predictors and exclusions

### 3.1 IST (19,435 randomised; 467 hospitals; 36 countries; 1991–1996)

- **Outcomes.**
  - `ist_death14`: death within 14 days (`ID14 = 1`), among patients with known 14-day status (`SET14D = 1`). Low prevalence.
  - `ist_dd6`: dead or dependent at 6 months (`OCCODE` 1 or 2 vs 3 or 4). `OCCODE` 0 or 9 (unknown) excluded. Common outcome.
- **Predictors (baseline only, all recorded before randomisation; 20 columns).** Age, sex, delay from onset (h), systolic BP, conscious state (alert / drowsy / unconscious), symptoms on waking, atrial fibrillation, aspirin within 3 days, heparin within 24 h, CT status (not done / done without visible infarct / visible infarct), OCSP stroke subtype (TACS, PACS, LACS, POCS, other), the eight deficit items (face, arm, leg, dysphasia, hemianopia, visuospatial, brainstem/cerebellar, other) and one flag for "deficit could not be assessed".
  - This is the full set of baseline variables recorded at randomisation (Sandercock et al. 2011, Table 2). It covers the prognostic domains of the Counsell et al. (2002) six-variable model (age, consciousness, motor deficit); pre-stroke independence and living alone are not recorded in IST. Which baseline variables Weir et al. (2001) used for case-mix adjustment is not yet checked against their paper.
  - Deficit coding: Y = 1, N = 0; "C" (cannot assess) = 0 for the item plus `def_unassessable = 1`.
  - Not predictors (kept as columns): country, date of randomisation, trial arms (`rxasp`, `rxhep`; randomised, so balanced), and IST's own predicted risks `EXPD14`/`EXPDD` (derivation not documented in the database paper; usable only as a descriptive comparator).
- **Exclusions.** Unknown outcome (above). Complete case on predictors: the known gap is atrial fibrillation, not recorded for the 984 pilot-phase patients (5.1 %). Exact counts appear in `flow` in the meta file once the data are built.

### 3.2 CRASH-2 (20,207 patients in the public file; 274 hospitals; 40 countries; 2005–2010)

- **Outcome** `crash2_death28`: death in hospital within 28 days of randomisation (`ddeath` present and at most 28 days after `trandomised`). Patients without an outcome form are excluded. Deaths dated after day 28 are counted in `flow` and coded 0.
- **Predictors (8)**, following the final model of Perel et al. (2012): Glasgow coma score (total, or the sum of its three components), age, systolic BP, heart rate, respiratory rate, hours since injury (continuous; Perel et al. dichotomised it at 3 h), injury type (blunt only vs any penetrating), plus sex. Capillary refill time is kept as a column but excluded, as in Perel et al.
- **Missing-value codes** from the data dictionary: SBP 0/999, respiratory rate 0, hours since injury 0/999 are set to missing.
- **Limits of the public file.** No country and no treatment code (the dictionary says the treatment code is withheld, and data for one site are excluded). The income-region adjustment and the treatment term of Perel et al. cannot be reproduced; our model is marginal over the tranexamic-acid arms (14.5 % vs 16.0 % mortality in the trial).
- **Exclusions.** Complete case on predictors (Perel et al. did the same; under 1 % missing for each predictor except capillary refill, which we do not use).

### 3.3 Sepsis survival (UCI 827), **built**

- **Outcome** `sepsis_death`: death (source column `hospital_outcome_1alive_0dead = 0`), about 9 days after admission.
- **Predictors (3)**: age (0–100 years), sex, septic episode number (1–5). Nothing is missing.
- **Unit of analysis**: admissions, not patients. The 110,204 admissions belong to 84,811 people (the count of `episode = 1` rows is exactly 84,811). There is no identifier, so repeated admissions cannot be clustered. This is reported as a limitation.
- **Cohorts.** The study cohort (19,051 Sepsis-3 admissions) is a subset of the primary cohort (110,204). With only four columns, identical rows are interchangeable, so removing the study rows pattern by pattern gives exactly the 91,153 non-Sepsis-3 admissions. The loader checks that the subset relation holds for every pattern; it does.
- There are only 975 distinct covariate patterns. The observed death rate per pattern in the external set is therefore a near-exact reference curve for calibration (an empirical "truth" without a model).

| Cohort | n | Deaths | Rate |
|---|---|---|---|
| Primary, not Sepsis-3 | 91,153 | 4,499 | 4.9 % |
| Study cohort (Sepsis-3) | 19,051 | 3,606 | 18.9 % |
| Korean validation cohort (`sepsis_korea`) | 137 | 24 | 17.5 % |

### 3.4 PhysioNet/CinC Challenge 2012, **built**

- **Population**: 12,000 adult ICU stays of at least 48 h from MIMIC-II (sets A, B, C, 4,000 each; outcome files for all three are now public). Coronary care 1,769, cardiac surgery recovery 2,530, medical 4,293, surgical 3,408.
- **Outcome** `physionet_ihd`: in-hospital death (`In-hospital_death`). 1,707 deaths (14.2 %).
- **Predictors (27)**, built from the first 48 h, following the SAPS-I variable list (Le Gall et al. 1984): age, sex, ICU type, minimum and maximum of heart rate, systolic BP (invasive, else non-invasive), temperature, respiratory rate, white cell count, glucose, potassium and sodium; minimum GCS, haematocrit and bicarbonate; maximum BUN, creatinine and bilirubin; total urine output; any mechanical ventilation. Values outside physiological ranges (listed in `PN_RANGE` in the code) are set to missing. SAPS-I and SOFA from the outcome files are kept as columns but never used as predictors.
- **Missing data.** Complete case would lose most stays: respiratory rate is missing for 8,672 stays (mostly ventilated patients) and bilirubin for 6,787. About 180–300 stays miss each other variable, and 15 stays have no time-series data at all.
  - Policy: missing values stay as NaN in the table.
  - In each replicate `impute_simple()` fills medians learned on the development rows and adds a missing-value indicator for every variable with at least 5 % missing in those rows. With n = 500 this added indicators for respiratory rate (min, max) and bilirubin.
  - Every arm gets the same imputed matrix. A sensitivity run with the TFMs' native NaN handling is possible but not primary.

## 4. Missing-data policy (summary)

| Source | Primary | Sensitivity |
|---|---|---|
| IST | complete case on predictors | `<key>_all.parquet` keeps the missing rows (e.g. AF unknown as its own category) |
| CRASH-2 | complete case (as Perel et al.) | `crash2_death28_all.parquet` |
| Sepsis | none missing | — |
| PhysioNet | dev-fitted median + indicators, identical for all arms | TFM native NaN handling |

Outcome-missing rows are always excluded and counted in `flow`. Imputation is fitted inside each development subsample, never on the validation rows.

## 5. Splits for the real-data tier

Each table carries one or more split columns with values `dev_pool`, `ext_recal` (500 random external rows, fixed seed, for the recalibration ladder at m = 100 and 500) and `ext_eval` (all other external rows, for calibration and GOF testing). The split is made after the complete-case step, so `ext_recal` always holds 500 usable rows.

Learning curves: `subsample_dev(df, key, n, b)` draws subsample b (b = 0…99) of size n ∈ {200, 500, 1000, 3000} from `dev_pool` by simple random sampling. The seed is derived from (key, split, n, b), so the draws can be rebuilt exactly. Subsamples with very few events are kept, because degenerate fits are an outcome to report. The builder warns if any development pool is below 3,500 rows.

| Key | Split column | Development | External | Type |
|---|---|---|---|---|
| `ist_*` | `split` (primary) | UK (country code 27) | all other 35 countries | geographic; Weir et al. 2001 documented real between-country outcome differences |
| `ist_*` | `split_temporal` | randomised before 1994 | 1994–1996 | temporal |
| `crash2_death28` | `split` | randomised before 1 Jan 2008 | 2008–2010 | temporal (the public file has no country) |
| `sepsis_death` | `split` (primary) | 91,153 non-Sepsis-3 admissions (4,499 deaths) | 19,051 Sepsis-3 admissions: 500 recal (82 deaths) + 18,551 eval (3,524 deaths) | case-mix / definition shift; a large intercept shift (4.9 % → 18.9 %) is expected |
| `sepsis_death` | `split_random` | random half (54,985) | other half: 500 + 54,719 | random, sensitivity |
| `physionet_ihd` | `split` (primary) | sets A + B: 8,000 (1,122 deaths) | set C: 500 recal (68) + 3,500 eval (517) | held-out set. The three sets are random thirds of one source, so this is a held-out, not an external, validation. |
| `physionet_ihd` | `split_icu` | CCU + CSRU + SICU: 7,707 (855 deaths) | medical ICU: 500 recal (83) + 3,793 eval (769) | case-mix shift across unit types |

Pre-specified fallbacks, to be applied only if the warning fires once the IST and CRASH-2 files are in:
- IST UK pool below 3,500: development = UK + Ireland + Northern Europe (Norway, Sweden, Denmark, Finland, Netherlands).
- CRASH-2 pre-2008 pool below 3,500: move the cut to the date that splits the complete-case table in half.

All validation sets have at least 500 events except PhysioNet set C (585) and the ICU split (852), which are still above Riley et al.'s 100-event minimum for validation. Riley minimum development sizes (`pmsampsize`) will be computed in P1 with R, per outcome, from the observed prevalence and an anticipated Cox–Snell R² taken from a full-pool logistic fit.

## 6. Plasmode tier

Purpose (plan §5): a known true risk p\* on real covariate distributions, and outcomes that no TFM could have seen during pretraining.

- **Truth model.** Fit a main-effect logistic model once to the real outcome on the full complete-case table, with standardised design columns, to get realistic coefficients. Then add two terms of fixed size on the logit scale:
  - one U-shaped term, γ_nl·(z² − 1) with γ_nl = 0.4, where z is a standardised continuous predictor;
  - one interaction, γ_int·z_a·x_b with γ_int = 0.6.
  - The intercept is solved by bisection so that the mean p\* hits each target prevalence, 10 % and 30 %.

| Key | Non-linear term | Interaction |
|---|---|---|
| `ist_dd6`, `ist_death14` (primary plasmode) | systolic BP (U-shape) | age × impaired consciousness |
| `crash2_death28` | systolic BP | GCS × penetrating injury |
| `physionet_ihd` (available now; covariates median-filled for the truth only) | maximum heart rate | age × mechanical ventilation |

- **Freezing.** `realdata.py plasmode <key>` writes `plasmode_<key>.json` with the following, plus `plasmode_<key>_p10.npy` and `_p30.npy`:
  - the spec;
  - the main-effect coefficients;
  - the source-table fingerprint;
  - the intercept shift;
  - a SHA256 of p\*.
- **Outcomes per replicate.** `plasmode_outcomes(key, tag, rep)` first checks p\* against its stored hash, then draws Y\* ~ Bernoulli(p\*) with a seed derived from (key, prevalence, rep).
- **Per-replicate design.** Development subsamples n ∈ {200, 500, 1000, 3000} come from `dev_pool`, and evaluation is on `ext_eval`. The oracle estimands of plan §4.3 (true ICI, b\*, a\*, OCE) are computed against p\*.
- The coefficient sizes (0.4 and 0.6) are proposals for the pre-registration. They should be checked in P1 so that the deviation from the main-effect model is detectable but not extreme; for example, report the oracle AUC and the true ICI of a main-effect LR fitted at n = 3000.

The PhysioNet plasmode truth is already frozen: mean p\* = 0.100 and 0.300.

## 7. Open points for the author

1. IST and CRASH-2 downloads (`AUTHOR_STEPS.md`). For CRASH-2, report whether the file has a country column.
2. Primary split for PhysioNet: the held-out set C (default) or the medical-ICU transport split?
3. Plasmode effect sizes (γ_nl = 0.4, γ_int = 0.6): accept for the pre-registration, or tune in P1.
4. Sepsis: its primary split carries a large prevalence shift. That suits the recalibration ladder (intercept repair), but it will dominate the calibration-in-the-large results. Keep it as primary, or use `split_random`?
