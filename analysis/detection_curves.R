# Detection curves of the goodness-of-fit tests (pre-registration 6.2 and Q4b, 7.4).
#
#   Rscript analysis/detection_curves.R [--B 500] [--workers 1] [--tables analysis/tables]
#
# For each test, P(reject at 0.05) on the observed Y is modelled against the true ICI with a
# logistic GAM, gam(reject ~ s(true_ici, k = 10), family = binomial, method = "REML"), fitted across
# replicates, arms and scenarios at N_eval (core and stress scenarios, every arm as fitted; the
# ladder columns belong to Q5 and are not used). Uncertainty: a cluster bootstrap over replicates
# (a cluster is one scenario x replicate, which holds every arm fitted on the same data), B = 500.
# A resample is fitted with the cluster counts as prior weights, which for a binomial likelihood is
# the same as repeating the rows.
#
# Q4b: EDGE (G = 10) minus Hosmer-Lemeshow (external) at true ICI 0.01, 0.02, 0.03 and 0.05, from
# curves fitted on the same rows (both tests with a p-value), with the same resamples.
#
# Outputs (analysis/tables/summary/): detection_curves.csv, detection_points.csv,
# detection_edge_vs_hl.csv, detection_fit_info.csv.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "_common.R"))
suppressPackageStartupMessages({ library(mgcv); library(arrow); library(data.table) })

B <- as.integer(arg_value("--B", "500"))
workers <- min(as.integer(Sys.getenv("ANALYSIS_MAX_WORKERS", "2")), as.integer(arg_value("--workers", "1")))
tab_dir <- arg_value("--tables", file.path(analysis_dir, "tables"))
out_dir <- check_out(file.path(tab_dir, "summary"))
POINTS <- c(0.01, 0.02, 0.03, 0.05)

g <- as.data.table(read_parquet(file.path(tab_dir, "gof.parquet"),
                                col_select = c("scenario", "rep", "eval_set", "column", "ladder", "ytype",
                                               "test", "reject", "tier")))
g <- g[eval_set == "eval" & ytype == "real" & ladder == "R0" & tier != "random" & !is.na(reject) &
         test %in% c(BATTERY, EXTRA)]
s <- as.data.table(read_parquet(file.path(tab_dir, "scores.parquet"),
                                col_select = c("scenario", "rep", "column", "o_true_ici", "usable")))
s <- s[usable == TRUE & is.finite(o_true_ici)]
d <- merge(g, s[, .(scenario, rep, column, ici = o_true_ici)], by = c("scenario", "rep", "column"))
d[, cluster := paste(scenario, rep, sep = "|")]
cat(sprintf("detection data: %d rows, %d tests, %d clusters\n", nrow(d), uniqueN(d$test), uniqueN(d$cluster)))
if (!nrow(d)) quit(status = 0)

grid_max <- as.numeric(quantile(d$ici, 0.99))
GRID <- sort(unique(c(seq(0, grid_max, length.out = 101), POINTS)))

fit_curve <- function(dd, w = NULL, at = GRID) {
  k <- min(10L, length(unique(dd$ici)) - 1L)
  if (k < 3) return(rep(NA_real_, length(at)))
  m <- if (is.null(w)) gam(reject ~ s(ici, k = k), family = binomial, method = "REML", data = dd)
       else gam(reject ~ s(ici, k = k), family = binomial, method = "REML", data = dd, weights = w)
  as.numeric(predict(m, newdata = data.frame(ici = at), type = "response"))
}

boot_weights <- function(clusters, b, seed) {
  u <- unique(clusters)
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  idx <- match(clusters, u)
  lapply(seq_len(b), function(i) tabulate(sample.int(length(u), length(u), replace = TRUE), length(u))[idx])
}

run_boot <- function(dd, fun, seed) {
  force(fun)       # an unforced promise reaches the workers as a bare name they cannot resolve
  W <- boot_weights(dd$cluster, B, seed)
  one <- function(w) { keep <- w > 0; tryCatch(fun(dd[keep], w[keep]), error = function(e) conditionMessage(e)) }
  if (workers > 1) {
    cl <- parallel::makeCluster(workers)
    on.exit(parallel::stopCluster(cl))
    parallel::clusterExport(cl, c("GRID", "POINTS", "fit_curve", "rebuild_dir"), envir = globalenv())
    parallel::clusterEvalQ(cl, { Sys.setenv(OMP_NUM_THREADS = "1"); .libPaths(c(file.path(rebuild_dir, "rlib"), .Library))
      suppressPackageStartupMessages({ library(mgcv); library(data.table) }); NULL })
    res <- parallel::parLapply(cl, W, one)
  } else res <- lapply(W, one)
  # a failed resample comes back as its error message: count and show them, never drop silently
  bad <- vapply(res, is.character, logical(1))
  if (any(bad)) message(sprintf("  %d of %d resamples failed; first error: %s", sum(bad), length(res), res[bad][[1]]))
  res[!bad]
}

t0 <- Sys.time()
curves <- list(); info <- list(); pts <- list()
Q4B_ONLY <- "--q4b-only" %in% commandArgs(TRUE)       # re-run only the EDGE vs HL comparison
for (tt in if (Q4B_ONLY) character(0) else intersect(c(BATTERY, EXTRA), unique(d$test))) {
  dd <- d[test == tt]
  est <- fit_curve(dd)
  bs <- run_boot(dd, function(x, w) fit_curve(x, w), task_seed(paste("detection", tt)))
  M <- do.call(rbind, bs)
  lo <- apply(M, 2, quantile, 0.025, na.rm = TRUE); hi <- apply(M, 2, quantile, 0.975, na.rm = TRUE)
  curves[[tt]] <- data.table(test = tt, ici = GRID, fit = est, lo = lo, hi = hi)
  sel <- match(POINTS, GRID)
  pts[[tt]] <- data.table(test = tt, ici = POINTS, fit = est[sel], lo = lo[sel], hi = hi[sel])
  info[[tt]] <- data.table(test = tt, rows = nrow(dd), clusters = uniqueN(dd$cluster),
                           arms = paste(sort(unique(dd$column)), collapse = ","), B_ok = nrow(M),
                           secs = round(as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  cat(sprintf("  %-32s rows %6d  B_ok %d  [%.0f s]\n", tt, nrow(dd), nrow(M),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
if (!Q4B_ONLY) {
  pts <- rbindlist(pts)
  # the rank of the tests at each point (1 = most often rejects), from the fitted curves
  pts[, rank := frank(-fit, ties.method = "min"), by = ici]
  fwrite(rbindlist(curves), file.path(out_dir, "detection_curves.csv"))
  fwrite(pts, file.path(out_dir, "detection_points.csv"))
  fwrite(rbindlist(info), file.path(out_dir, "detection_fit_info.csv"))
}

# Q4b: EDGE (G = 10) against HL on the same rows
e <- d[test == "EDGE (G=10)", .(scenario, rep, column, cluster, ici, edge = reject)]
h <- d[test == "Hosmer-Lemeshow (external)", .(scenario, rep, column, hl = reject)]
eh <- merge(e, h, by = c("scenario", "rep", "column"))
if (nrow(eh) > 50) {
  both <- function(x, w = NULL) {
    a <- fit_curve(data.table(ici = x$ici, reject = x$edge), w, POINTS)
    b <- fit_curve(data.table(ici = x$ici, reject = x$hl), w, POINTS)
    c(a, b, a - b)
  }
  est <- both(eh)
  # pass `both` itself: a wrapper defined here would look `both` up in the workers' global
  # environment, where it does not exist, and every resample silently returned NULL (10-16)
  bs <- run_boot(eh, both, task_seed("detection EDGE vs HL"))
  if (!length(bs)) stop("EDGE vs HL: every bootstrap resample failed")
  M <- do.call(rbind, bs)
  q <- function(j, p) as.numeric(quantile(M[, j], p, na.rm = TRUE))
  k <- length(POINTS)
  out <- data.table(ici = POINTS, edge = est[1:k], hl = est[k + 1:k], diff = est[2 * k + 1:k],
                    edge_lo = sapply(1:k, q, 0.025), edge_hi = sapply(1:k, q, 0.975),
                    hl_lo = sapply(k + 1:k, q, 0.025), hl_hi = sapply(k + 1:k, q, 0.975),
                    diff_lo = sapply(2 * k + 1:k, q, 0.025), diff_hi = sapply(2 * k + 1:k, q, 0.975),
                    rows = nrow(eh), clusters = uniqueN(eh$cluster), B_ok = nrow(M))
  fwrite(out, file.path(out_dir, "detection_edge_vs_hl.csv"))
}
cat(sprintf("detection curves done in %.0f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
