## riley_pmsampsize.R -- Riley et al. (2020) minimum development size of every scenario with
## pmsampsize (binary outcome, R^2 from the oracle c-statistic, shrinkage 0.9), as the
## pre-registration (section 3.4) specifies. Called by `python dgp.py build`:
##   Rscript riley_pmsampsize.R <scenarios.csv> <out.csv>

local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  rlib <- file.path(dirname(dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/"))), "rlib")
  if (dir.exists(rlib)) .libPaths(c(rlib, .Library))
})
args <- commandArgs(TRUE)
sc <- read.csv(args[1], stringsAsFactors = FALSE)
one <- function(i) {
  r <- sc[i, ]
  out <- tryCatch({
    s <- suppressMessages(utils::capture.output(
      ps <- pmsampsize::pmsampsize(type = "b", cstatistic = r$oracle_auc, parameters = r$n_params,
                                   prevalence = r$pi, shrinkage = 0.9, seed = 123456)))
    c(ps$sample_size, ps$results_table[1, "CS_Rsq"])
  }, error = function(e) c(NA, NA))
  data.frame(scenario_id = r$scenario_id, riley_n_min_pmsampsize = out[1], r2cs_pmsampsize = out[2])
}
res <- do.call(rbind, lapply(seq_len(nrow(sc)), one))
write.csv(res, args[2], row.names = FALSE)
cat("pmsampsize", as.character(packageVersion("pmsampsize")), "\n")
