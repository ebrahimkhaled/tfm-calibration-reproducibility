#!/usr/bin/env bash
# Download the two real data sets that can be fetched without registration and check them
# against data/raw/MANIFEST_SHA256.txt (the SHA-256 of the files the study used).
#
#   bash data/fetch_open_data.sh
#
# The International Stroke Trial and CRASH-2 files must be downloaded by hand, under the
# providers' terms; see data/README.md. Nothing here accepts a licence for you.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$HERE/data/raw"
mkdir -p "$RAW/physionet2012" "$RAW/uci_sepsis"

# PhysioNet/Computing in Cardiology Challenge 2012, v1.0.0 (Open Data Commons Attribution v1.0)
PN="https://physionet.org/files/challenge-2012/1.0.0"
for f in set-a.tar.gz set-b.tar.gz set-c.tar.gz Outcomes-a.txt Outcomes-b.txt Outcomes-c.txt; do
  [ -f "$RAW/physionet2012/$f" ] || curl -fL -o "$RAW/physionet2012/$f" "$PN/$f"
done

# Sepsis Survival Minimal Clinical Records, UCI Machine Learning Repository 827 (CC BY 4.0)
UCI="https://archive.ics.uci.edu/static/public/827/sepsis+survival+minimal+clinical+records.zip"
if [ ! -f "$RAW/uci_sepsis/s41598-020-73558-3_sepsis_survival_primary_cohort.csv" ]; then
  curl -fL -o "$RAW/uci_sepsis/sepsis.zip" "$UCI"
  (cd "$RAW/uci_sepsis" && unzip -o sepsis.zip)
  # the UCI archive may hold a second zip with the three cohort files
  for z in "$RAW"/uci_sepsis/*.zip; do
    [ "$(basename "$z")" = "sepsis.zip" ] || (cd "$RAW/uci_sepsis" && unzip -o "$(basename "$z")")
  done
fi

# check the files against the study's hashes (lines for files that are present)
cd "$RAW"
grep -E "physionet2012/|uci_sepsis/s41598-020-73558-3_sepsis_survival_(primary|study|validation)_cohort.csv" \
  "$RAW/MANIFEST_SHA256.txt" | sha256sum -c -
echo "Open data in place. Build the tables with: python code/realdata.py build sepsis physionet"
