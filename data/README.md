# Real data: where to get them

No real data are in this repository. Each data set is available from its provider under its own terms. The loaders in `code/realdata.py` read the files from `data/raw/<source>/` and check them against `data/raw/MANIFEST_SHA256.txt`, the SHA-256 of the files used in the study. A table is rebuilt exactly when the file is the same. Sources, licences, outcomes, predictors and splits are documented in `DATA_REPORT.md`.

| Source | Provider and terms | Files expected | How |
|---|---|---|---|
| Sepsis Survival Minimal Clinical Records | UCI Machine Learning Repository, doi:10.24432/C53C8N, CC BY 4.0 | `data/raw/uci_sepsis/s41598-020-73558-3_sepsis_survival_{primary,study,validation}_cohort.csv` | `bash data/fetch_open_data.sh` |
| PhysioNet/CinC Challenge 2012, v1.0.0 | https://physionet.org/content/challenge-2012/1.0.0/, Open Data Commons Attribution v1.0 | `data/raw/physionet2012/set-{a,b,c}.tar.gz`, `Outcomes-{a,b,c}.txt` | `bash data/fetch_open_data.sh` |
| International Stroke Trial database, version 2 | Edinburgh DataShare, doi:10.7488/ds/104 (read the End-user Licence on the item page) | `data/raw/ist/IST_corrected.csv` | download by hand |
| CRASH-2 | LSHTM Clinical Trials Unit, freeBIRD repository (https://freebird.lshtm.ac.uk), doi:10.17037/DATA.10, freeBIRD Data Sharing Agreement (registration; no redistribution) | `data/raw/crash2/CRASH-2_Data_anonymised_dataset_w.csv` | request and download by hand |

Then build the analysis tables and the plasmode truths:

```sh
python code/realdata.py build                     # every source whose files are present
python code/realdata.py plasmode ist_dd6 ist_death14 crash2_death28 physionet_ihd
python code/realdata.py summary
```

The CRASH-2 results of the study were published with the permission of the trial's Chief Investigators. Anyone who reruns the CRASH-2 analysis is bound by the freeBIRD Data Sharing Agreement, including its rules on publication.
