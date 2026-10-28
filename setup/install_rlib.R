# Project R library for the TFM calibration rebuild.
# Everything goes into rebuild/rlib so the analysis does not depend on the
# user library (which holds ebrahim.gof 2.8.0 and other versions).
#
# Run from rebuild/:  Rscript setup/install_rlib.R
# ebrahim.gof itself is installed separately by setup/install_ebrahim_gof.sh
# (git archive of ref b248d83, never the working tree).

rlib <- normalizePath("rlib", winslash = "/", mustWork = TRUE)
.libPaths(c(rlib, .Library))   # project library first, base R last, no user library
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(install.packages.check.source = "no")

pkgs <- c(
  # analysis arms
  "mgcv", "logistf", "glmnet",
  # calibration curves / ICI
  "CalibrationCurves",
  # I/O
  "jsonlite", "arrow",
  # ebrahim.gof imports and the suggests used by the external battery
  "CompQuadForm", "ResourceSelection", "statmod", "givitiR", "callr",
  "randomForest", "dcov", "TH.data", "BAGofT",
  # sample size and misc
  "pmsampsize", "data.table"
)

have <- rownames(installed.packages(lib.loc = rlib))
todo <- setdiff(pkgs, have)
cat("To install:", paste(todo, collapse = ", "), "\n")
if (length(todo)) {
  install.packages(todo, lib = rlib, type = "binary",
                   dependencies = c("Depends", "Imports", "LinkingTo"))
}

# anything that had no binary: try source
left <- setdiff(pkgs, rownames(installed.packages(lib.loc = rlib)))
if (length(left)) {
  cat("No binary for:", paste(left, collapse = ", "), "- trying source\n")
  try(install.packages(left, lib = rlib, type = "source",
                       dependencies = c("Depends", "Imports", "LinkingTo")))
}

ip <- installed.packages(lib.loc = rlib)
miss <- setdiff(pkgs, rownames(ip))
cat("\nInstalled in rlib:\n")
print(ip[intersect(pkgs, rownames(ip)), c("Package", "Version")], quote = FALSE)
if (length(miss)) cat("\nMISSING:", paste(miss, collapse = ", "), "\n") else cat("\nAll requested packages present.\n")
