# Timing benchmark for the statistical arms (CPU, one core each):
#   LR-MLE (glm), FLIC (logistf, flic = TRUE), ridge (cv.glmnet, alpha = 0,
#   10-fold CV deviance), GAM (mgcv, REML, one s() per continuous predictor).
# Reads the parquet files written by code/bench_00_make_data.py, checks the
# checksum in the manifest, and appends one row per fit to
# results/bench/timings_r.csv (rows already there are skipped, so it resumes).
#
# Run from rebuild/:  Rscript R/bench_r_arms.R

rlib <- normalizePath("rlib", winslash = "/", mustWork = TRUE)
.libPaths(c(rlib, .Library))
suppressPackageStartupMessages({
  library(arrow); library(jsonlite); library(logistf); library(glmnet); library(mgcv)
})

data_dir <- "results/bench/data"
out_csv  <- "results/bench/timings_r.csv"
man <- fromJSON(file.path(data_dir, "manifest.json"))

checksum <- function(df, p) {
  x <- as.matrix(df[, paste0("x", seq_len(p))])
  w_row <- seq_len(nrow(df)) / nrow(df)
  sum(w_row * (x %*% seq_len(p))) + sum(w_row * df$y)
}

read_checked <- function(name, p) {
  df <- as.data.frame(read_parquet(file.path(data_dir, paste0(name, ".parquet"))))
  cs <- checksum(df, p)
  stopifnot("data fingerprint mismatch" = abs(cs - man[[name]]$checksum) < 1e-6 * max(1, abs(cs)))
  df
}

# GAM basis size: k = 10 unless that would give more coefficients than n / 2
gam_k <- function(n, p) max(3L, min(10L, floor(n / (2 * p)) + 1L))

fit_predict <- function(arm, tr, ev, p) {
  xn <- paste0("x", seq_len(p))
  warn <- character(0)
  wh <- function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }
  t0 <- as.numeric(Sys.time())
  fit <- withCallingHandlers(switch(arm,
    LR    = glm(reformulate(xn, "y"), family = binomial, data = tr),
    FLIC  = logistf(reformulate(xn, "y"), data = tr, flic = TRUE),
    RIDGE = cv.glmnet(as.matrix(tr[, xn]), tr$y, family = "binomial", alpha = 0,
                      nfolds = 10, type.measure = "deviance"),
    GAM   = {
      k <- gam_k(nrow(tr), p)
      f <- as.formula(paste("y ~", paste0("s(", xn, ", k = ", k, ")", collapse = " + ")))
      gam(f, family = binomial, data = tr, method = "REML")
    }), warning = wh)
  t1 <- as.numeric(Sys.time())
  phat <- switch(arm,
    LR    = predict(fit, newdata = ev, type = "response"),
    FLIC  = predict(fit, newdata = ev, type = "response"),
    RIDGE = as.numeric(predict(fit, newx = as.matrix(ev[, xn]), s = "lambda.min", type = "response")),
    GAM   = as.numeric(predict(fit, newdata = ev, type = "response")))
  t2 <- as.numeric(Sys.time())
  list(fit_s = t1 - t0, predict_s = t2 - t1, phat = as.numeric(phat),
       note = paste(unique(warn), collapse = " | "))
}

auc <- function(y, p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

done <- if (file.exists(out_csv)) read.csv(out_csv, stringsAsFactors = FALSE) else NULL
key_of <- function(arm, n, p, rep) paste(arm, "cpu", n, p, rep, sep = "_")
done_keys <- if (is.null(done)) character(0) else
  with(done[done$status == "ok", ], key_of(model, n_train, p, rep))

for (p in c(5, 20)) for (rep in 1:3) {
  ev <- read_checked(sprintf("eval_p%d_r%d", p, rep), p)
  for (n in c(100, 500, 2000)) {
    tr <- read_checked(sprintf("train_p%d_n%d_r%d", p, n, rep), p)
    for (arm in c("LR", "FLIC", "RIDGE", "GAM")) {
      if (key_of(arm, n, p, rep) %in% done_keys) next
      set.seed(1000 * p + 10 * n + rep)   # cv.glmnet folds
      res <- tryCatch(fit_predict(arm, tr, ev, p),
                      error = function(e) list(error = conditionMessage(e)))
      ok <- is.null(res$error)
      if (ok) write_parquet(data.frame(phat = res$phat),
                            sprintf("results/bench/preds/%s_cpu_p%d_n%d_r%d.parquet", arm, p, n, rep))
      row <- data.frame(
        model = arm, impl = "R", device = "cpu", threads = 1L,
        n_train = n, p = p, n_eval = nrow(ev), rep = rep,
        fit_s = if (ok) res$fit_s else NA, predict_s = if (ok) res$predict_s else NA,
        total_s = if (ok) res$fit_s + res$predict_s else NA,
        peak_gpu_mb = NA, status = if (ok) "ok" else "error",
        auc = if (ok) auc(ev$y, res$phat) else NA,
        mean_phat = if (ok) mean(res$phat) else NA,
        note = if (ok) substr(res$note, 1, 200) else substr(res$error, 1, 200),
        stringsAsFactors = FALSE)
      write.table(row, out_csv, sep = ",", row.names = FALSE,
                  col.names = !file.exists(out_csv), append = file.exists(out_csv))
      cat(sprintf("%-5s p=%2d n=%4d rep=%d  %s  total %.3f s\n", arm, p, n, rep, row$status, row$total_s))
    }
  }
}
cat("R arms done\n")
