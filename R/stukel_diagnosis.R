## stukel_diagnosis.R -- why is Stukel (offset) in run.all.external() liberal? (read-only study)
##
## The external Stukel row adds za = 0.5 lp^2 (lp >= 0) and zb = -0.5 lp^2 (lp < 0) to the frozen
## linear predictor lp = logit(p) as an offset and refers the likelihood ratio to chi-square(2).
## Here the predictions are the true risks, so the null holds exactly, and we compare:
##   LR2      the package's statistic (re-implemented line for line, checked against the package)
##   Score2   the score test of the same two terms at the null (no fitting at all)
##   LRa/LRb  each term on its own, LR on 1 df
## together with how many records carry the za term (lp >= 0).
##
##   Rscript R/stukel_diagnosis.R [reps] [workers]   -> results/stukel_diagnosis.csv
## Nothing in ebrahim.gof is changed; the package is only called once to confirm the re-implementation.

a <- commandArgs(TRUE)
REPS <- if (length(a) >= 1) as.integer(a[1]) else 4000L
WORKERS <- if (length(a) >= 2) as.integer(a[2]) else 6L
root <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(dirname(normalizePath(sub("^--file=", "", f[1]), winslash = "/")))
})
.libPaths(c(file.path(root, "rlib"), .Library))

## the four statistics on one (y, p) pair
stukel_stats <- function(y, p) {
  lp <- qlogis(p); v <- p * (1 - p)
  dev0 <- -2 * sum(y * log(p) + (1 - y) * log(1 - p))
  za <- 0.5 * lp^2 * (lp >= 0); zb <- -0.5 * lp^2 * (lp < 0)
  conv <- TRUE
  lr <- function(Z) {
    ## glm itself can stop with an error when an extreme term overflows (heavy tails); that
    ## replicate is counted as not converged, with no statistic
    f <- tryCatch(suppressWarnings(glm(y ~ 0 + Z, offset = lp, family = binomial())),
                  error = function(e) NULL)
    if (is.null(f) || !f$converged) { conv <<- FALSE; if (is.null(f)) return(NA_real_) }
    dev0 - f$deviance
  }
  Z <- cbind(za, zb)
  keep <- colSums(Z^2) > 0                       # za is all zero when no record has p >= 0.5
  U <- crossprod(Z[, keep, drop = FALSE], y - p)
  I <- crossprod(Z[, keep, drop = FALSE] * v, Z[, keep, drop = FALSE])
  sc <- tryCatch(drop(t(U) %*% solve(I, U)), error = function(e) NA_real_)
  lr2 <- lr(Z); conv2 <- conv
  c(LR2 = lr2, conv_LR2 = conv2, Score2 = sc, df_score = sum(keep),
    LRa = if (keep[1]) lr(za) else NA_real_, LRb = lr(zb),
    n_pos = sum(lp >= 0), ev_pos = sum(p[lp >= 0]),
    max_abs_zb = max(abs(zb)), info_share_top1 = {           # share of zb information in the top 1 %
      w <- v * zb^2; k <- max(1L, floor(0.01 * length(w))); sum(sort(w, TRUE)[1:k]) / sum(w) })
}

## risk distributions: logit-normal with an oracle AUC near 0.80 (sd 1.2, as in the main design) and a
## heavier-tailed version (logit = t on 3 df scaled to the same sd), at the main design's event rates
## and evaluation sizes
cells <- expand.grid(pi = c(0.03, 0.10, 0.30), tail = c("normal", "t3"), stringsAsFactors = FALSE)
cells$N <- ifelse(cells$pi == 0.03, 3334L, 1000L)
draw_lp <- function(N, tail) {
  z <- if (tail == "normal") rnorm(N) else rt(N, 3) / sqrt(3)
  1.2 * z
}
cells$alpha <- vapply(seq_len(nrow(cells)), function(i) {
  set.seed(1)
  z <- draw_lp(1e6, cells$tail[i])
  uniroot(function(a) mean(plogis(a + z)) - cells$pi[i], c(-15, 5))$root
}, 0)

one <- function(r, cell) {
  set.seed(100000L * cell$id + r)
  p <- plogis(cell$alpha + draw_lp(cell$N, cell$tail))
  y <- rbinom(cell$N, 1L, p)
  stukel_stats(y, p)
}

## the re-implementation equals the package row (one draw per cell). Up to commit 2c77f3c the row was
## the likelihood ratio (LR2); from 19480aa on it is the score test (Score2), and its Note says so.
suppressPackageStartupMessages(library(ebrahim.gof))
for (i in seq_len(nrow(cells))) {
  set.seed(i); p <- plogis(cells$alpha[i] + draw_lp(cells$N[i], cells$tail[i])); y <- rbinom(length(p), 1, p)
  b <- as.data.frame(run.all.external(y, p))
  pk <- b$Statistic[b$Test == "Stukel (offset)"]
  form <- if (grepl("score test", b$Note[b$Test == "Stukel (offset)"])) "Score2" else "LR2"
  stopifnot(abs(pk - stukel_stats(y, p)[[form]]) < 1e-6 * max(1, abs(pk)))
}
cat("re-implementation matches run.all.external() in all", nrow(cells), "cells (package row:", form, ")\n")

library(parallel)
cl <- makePSOCKcluster(WORKERS)
clusterExport(cl, c("stukel_stats", "draw_lp", "one"))
res <- list()
for (i in seq_len(nrow(cells))) {
  cell <- c(as.list(cells[i, ]), id = i)
  m <- do.call(rbind, parLapply(cl, seq_len(REPS), one, cell = cell))
  rej <- function(s, df) mean(pchisq(pmax(s, 0), df, lower.tail = FALSE) < 0.05, na.rm = TRUE)
  rr <- c(LR2 = rej(m[, "LR2"], 2), Score2 = mean(pchisq(m[, "Score2"], m[, "df_score"], lower.tail = FALSE) < 0.05, na.rm = TRUE),
          LRa = rej(m[, "LRa"], 1), LRb = rej(m[, "LRb"], 1))
  res[[i]] <- data.frame(pi = cell$pi, tail = cell$tail, N = cell$N, reps = REPS,
                         size_LR2 = rr[["LR2"]], size_Score2 = rr[["Score2"]],
                         size_LRa = rr[["LRa"]], size_LRb = rr[["LRb"]],
                         mcse = sqrt(0.05 * 0.95 / REPS),
                         nonconv_LR2 = mean(m[, "conv_LR2"] == 0), negative_LR2 = mean(m[, "LR2"] < -1e-6),
                         size_LR2_converged = rej(m[m[, "conv_LR2"] == 1, "LR2"], 2),
                         median_LR2 = median(m[, "LR2"]),
                         mean_LRa = mean(m[, "LRa"], na.rm = TRUE), mean_LRb = mean(m[, "LRb"], na.rm = TRUE),
                         glm_error_LR2 = mean(is.na(m[, "LR2"])),
                         median_n_pos = median(m[, "n_pos"]), median_events_pos = median(m[, "ev_pos"]),
                         median_max_abs_zb = median(m[, "max_abs_zb"]),
                         median_info_share_top1 = median(m[, "info_share_top1"]))
  cat(sprintf("%s pi=%.2f: LR2 %.4f  Score2 %.4f  LRa %.4f  LRb %.4f  (n_pos %g)\n", cell$tail, cell$pi,
              rr[["LR2"]], rr[["Score2"]], rr[["LRa"]], rr[["LRb"]], median(m[, "n_pos"])))
}
stopCluster(cl)
out <- do.call(rbind, res)
write.csv(out, file.path(root, "results", "stukel_diagnosis.csv"), row.names = FALSE)
print(out, digits = 4)
