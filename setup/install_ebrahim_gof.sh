#!/usr/bin/env bash
# Install ebrahim.gof 2.10.0 at the study's commit (19480aa) into the project R library.
# The source is shipped in third_party/ebrahim.gof-2.10.0-19480aa.tar.gz (a git archive of
# that commit). Its R code is identical to the CRAN release 2.10.0; only DESCRIPTION's date,
# cran-comments.md and one test tolerance differ.
#
#   bash setup/install_ebrahim_gof.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/setup/src"
RLIB="$(cygpath -m "$HERE/rlib" 2>/dev/null || echo "$HERE/rlib")"   # R on Windows wants C:/...
mkdir -p "$SRC" "$HERE/rlib"

rm -rf "$SRC/ebrahim.gof" "$SRC"/ebrahim.gof_*.tar.gz
tar -xzf "$HERE/third_party/ebrahim.gof-2.10.0-19480aa.tar.gz" -C "$SRC"
echo "19480aa" > "$SRC/ebrahim.gof/GIT_REF"

cd "$SRC"
# vignettes and the manual are not needed for the analysis
R CMD build --no-build-vignettes --no-manual ebrahim.gof
TARBALL=$(ls ebrahim.gof_*.tar.gz | head -1)
R_LIBS="$RLIB" R CMD INSTALL --no-multiarch -l "$RLIB" "$TARBALL"
sha256sum "$TARBALL" > "$TARBALL.sha256"
cat "$TARBALL.sha256"
Rscript -e ".libPaths(c('$RLIB', .Library)); library(ebrahim.gof); cat('ebrahim.gof', as.character(packageVersion('ebrahim.gof')), 'from', find.package('ebrahim.gof'), '\n')"
