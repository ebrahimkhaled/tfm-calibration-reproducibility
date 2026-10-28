# Reproduction materials: Tabular foundation models are not a shortcut to calibrated clinical risk prediction

Code for the article "Tabular foundation models are not a shortcut to calibrated clinical risk prediction: a pre-registered simulation against the true risk" by Ebrahim Khaled Ebrahim (Department of Applied Statistics, Faculty of Business, Alexandria University, Alexandria, Egypt; ORCID 0009-0006-7839-8778).

The study was pre-registered on OSF Registries before the main run: https://osf.io/8hmtz (doi:10.17605/OSF.IO/8HMTZ). Changes made after registration are listed in `prereg/DEVIATIONS.md` (D-01 to D-07).

The per-replicate predictions and test results of the simulation, the per-replicate results for the two open real data sets and the aggregate real-data tables are a separate Zenodo record (the benchmark record, CC BY 4.0). This repository holds the code that produced them.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | The simulation engine (Python): the 58 scenarios and the random-design tier (`dgp.py`, `dgp_random.py`, `scenarios.csv`, `random_designs.csv`), the model arms (`models.py`, `fit_r_arms.R`), the recalibration ladder, the estimands and the run loop; the real-data loaders and runner (`realdata.py`, `run_realdata.py`); the benchmark loader (`benchmark_loader.py`); the calibration audit tool (`audit_calibration.py`); the TabPFN weight download (`get_tabpfn_weights.py`); unit tests. See `code/README.md`. |
| `R/` | The goodness-of-fit layer (`gof_layer.R`, `realdata_gof.R`), which runs the external-validation battery of ebrahim.gof 2.10.0 on the frozen predictions; the size checks and the Stukel diagnosis. See `R/README.md`. |
| `analysis/` | Every table and figure of the article and its supplement from the run's outputs (`run_all.sh`). See `analysis/README.md`. |
| `prereg/` | The registered protocol (`preregistration.md`, `.tex`, `.pdf`), the frozen checksums, the registration record and the deviations log. |
| `data/` | No data. Where to get the four real data sets, a download script for the two open ones, the SHA-256 of every raw file used, and `DATA_REPORT.md` (sources, licences, outcomes, predictors, splits). |
| `setup/` | Building the Python environment (`make_venv_gpu.sh`, `requirements-gpu.lock.txt`), the R library (`install_rlib.R`) and ebrahim.gof (`install_ebrahim_gof.sh`). |
| `third_party/` | The source of the R package ebrahim.gof at commit 19480aa, the build used by the study. Its R code is identical to CRAN release 2.10.0 (doi:10.32614/CRAN.package.ebrahim.gof). |
| `environment/` | The recorded software environment of the main run (package versions, R packages, checkpoint SHA-256). |
| `docs/` | The pilot report, the main-run instructions and the note on the Stukel test. |

## Model weights

No model weights are included. TabPFN-3.5 is distributed by Prior Labs under the TABPFN-3.5 License v1.0, which does not allow redistribution. Accept the licence on Hugging Face (Prior-Labs/tabpfn_3_5), log in with `hf auth login`, and run `python code/get_tabpfn_weights.py`; the script checks the SHA-256 of each file against Section 13 of the pre-registration. TabICLv2 and Mitra-v2 are downloaded from Hugging Face at the revisions and checksums listed in `environment/environment.json` and Supplementary Table S2 of the article.

## How to reproduce

The study ran on Windows 11 with Git Bash, Python 3.11.15, R 4.4.1 and one 6 GB NVIDIA GPU.

1. **Environment.** `source env.sh` (or `. ./env.ps1`), then `bash setup/make_venv_gpu.sh`, `Rscript setup/install_rlib.R` and `bash setup/install_ebrahim_gof.sh`. Get the weights as above.
2. **Tests.** `python code/tests/run_tests.py`, `Rscript R/tests/test_gof_layer.R`, `python analysis/tests/test_rules.py`.
3. **Simulation.** `bash code/run_main.sh preflight`, then the streams `gpu`, `cpu_a`, `cpu_b`, `classical` and `random` (see `docs/RUN_MAIN.md`). The main run took about 3.6 days on one GPU and 19 threads. Every data set is rebuilt from its seed and checked against its stored SHA-256 fingerprint, so a rerun writes the same data.
4. **Real data.** Get the files (`data/README.md`), build the tables with `python code/realdata.py build`, then run `bash code/run_realdata.sh` with its streams.
5. **Tables and figures.** `bash analysis/run_all.sh`.

To reproduce the tables and figures without refitting any model, download the benchmark record and place its `results/` and `analysis/tables/` folders in the root of this repository, then run step 5.

## Licence

Code: MIT (see `LICENSE`). The pre-registration and the documents: CC BY 4.0. ebrahim.gof keeps its GPL-3 licence.

## Citation

See `CITATION.cff`. Please cite the article, this archive and, if you use the predictions, the benchmark record.
