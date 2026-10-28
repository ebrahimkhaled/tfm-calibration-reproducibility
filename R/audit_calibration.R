## audit_calibration.R -- a calibration audit of any model's frozen predictions, in one call.
##
## The recipe of the study (pre-registration section 14, deliverable b): the external battery of
## ebrahim.gof 2.10.0 (run.all.external) on validation outcomes y and predictions p from any model,
## then localize.external() to name which part is wrong (intercept, slope, link, covariates), with
## familywise error control at 0.05.
##
## In R:
##   source("R/audit_calibration.R"); audit_calibration(y, p, X = NULL, seed = 1)
## From the command line (this is what code/audit_calibration.py calls):
##   Rscript R/audit_calibration.R input.csv output.csv [--seed 1] [--M 999] [--G 10]
## input.csv has columns y (0/1) and p (predicted risk), and optionally covariates x_* (numeric) for
## localization over covariate subgroups and for le Cessie's test.

audit_calibration <- function(y, p, X = NULL, seed = 1L, M = 999L, G = 10L) {
  if (!requireNamespace("ebrahim.gof", quietly = TRUE)) stop("install ebrahim.gof (>= 2.10.0) first")
  y <- as.integer(y); p <- as.numeric(p)
  if (length(y) != length(p) || anyNA(y) || anyNA(p)) stop("y and p must have the same length, no NA")
  if (!all(y %in% 0:1)) stop("y must be 0/1")
  if (any(p <= 0 | p >= 1)) {
    k <- sum(p <= 0 | p >= 1)
    message(k, " predictions at 0 or 1 were moved to 1e-9 / 1 - 1e-9")
    p <- pmin(pmax(p, 1e-9), 1 - 1e-9)
  }
  b <- as.data.frame(ebrahim.gof::run.all.external(y, p, G = G, X = X, include_slow = !is.null(X)))
  out <- data.frame(part = "battery", test = b$Test, family = b$Family, statistic = b$Statistic,
                    df = b$df, p_value = b$p_value, note = b$Note, stringsAsFactors = FALSE)
  ## localization needs the covariates: the COV part looks for misfit within covariate subgroups
  L <- if (is.null(X)) simpleError("localization needs covariates (columns x_* in the input)") else
    tryCatch(ebrahim.gof::localize.external(y, p, X, M = M, seed = seed, plot = FALSE), error = function(e) e)
  if (inherits(L, "error")) {
    out <- rbind(out, data.frame(part = "localize", test = "Localize", family = "Localize", statistic = NA,
                                 df = NA, p_value = NA, note = paste("Not run:", conditionMessage(L))))
  } else {
    g <- names(L$single)
    out <- rbind(out, data.frame(
      part = "localize", test = c("Localize (any part named)", paste0("Localize ", g, " (named)")),
      family = "Localize", statistic = NA, df = NA, p_value = c(min(L$adjusted), L$adjusted),
      note = c(sprintf("named=%s; highest=%s; M=%d", if (length(L$named)) paste(L$named, collapse = "+") else "none",
                       ifelse(is.na(L$highest), "none", L$highest), L$draws),
               ifelse(g %in% L$named, "named", "not named"))))
  }
  attr(out, "ebrahim.gof") <- as.character(utils::packageVersion("ebrahim.gof"))
  rownames(out) <- NULL
  out
}

if (sys.nframe() == 0L) {
  a <- commandArgs(TRUE)
  if (length(a) < 2) stop("usage: Rscript R/audit_calibration.R input.csv output.csv [--seed 1] [--M 999] [--G 10]")
  opt <- function(k, v) { i <- match(k, a); if (is.na(i)) v else as.integer(a[i + 1]) }
  d <- utils::read.csv(a[1])
  names(d) <- tolower(names(d))
  xc <- grep("^x", names(d), value = TRUE)
  X <- if (length(xc)) d[, xc, drop = FALSE] else NULL
  res <- audit_calibration(d$y, d$p, X, seed = opt("--seed", 1L), M = opt("--M", 999L), G = opt("--G", 10L))
  utils::write.csv(res, a[2], row.names = FALSE)
  cat(sprintf("ebrahim.gof %s: %d rows written to %s\n", attr(res, "ebrahim.gof"), nrow(res), a[2]))
}
