## stukel_score_size.R -- size of the Stukel (offset) row of ebrahim.gof 2.10.0@19480aa (score form)
##
## The same null design as stukel_diagnosis.R (predictions = true risks, logit p = alpha + 1.2 z with
## z normal or t3 scaled to unit variance; pi = 3, 10, 30 %; N = 3,334 at 3 % and 1,000 otherwise;
## the same seeds), but the statistic is the package's own row, read from run.all.external().
##
##   Rscript R/stukel_score_size.R [reps] [workers]   -> results/stukel_score_size.csv

a <- commandArgs(TRUE)
REPS <- if (length(a) >= 1) as.integer(a[1]) else 4000L
WORKERS <- if (length(a) >= 2) as.integer(a[2]) else 6L
root <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(dirname(normalizePath(sub("^--file=", "", f[1]), winslash = "/")))
})
Sys.setenv(GOF_LAYER_NO_MAIN = "1", TFM_REBUILD_ROOT = root)
source(file.path(root, "R", "gof_layer.R"))
use_project_lib()                               # refuses any build but 19480aa

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
  set.seed(100000L * cell$id + r)               # the seeds of stukel_diagnosis.R
  p <- plogis(cell$alpha + draw_lp(cell$N, cell$tail))
  y <- rbinom(cell$N, 1L, p)
  ## t3 risks reach p = 1 in double precision now and then, which run.all.external() refuses; the test
  ## layer moves such predictions to 1e-9 / 1 - 1e-9 (clamp_p), and so does this study
  nclamp <- sum(p <= 1e-9 | p >= 1 - 1e-9)
  p <- pmin(pmax(p, 1e-9), 1 - 1e-9)
  b <- tryCatch(as.data.frame(suppressWarnings(ebrahim.gof::run.all.external(y, p))), error = function(e) NULL)
  if (is.null(b)) return(c(stat = NA, df = NA, p = NA, score = NA, clamped = nclamp))
  k <- b$Test == "Stukel (offset)"
  c(stat = b$Statistic[k], df = b$df[k], p = b$p_value[k], score = as.numeric(grepl("score test", b$Note[k])),
    clamped = nclamp)
}

library(parallel)
cl <- makePSOCKcluster(WORKERS)
clusterCall(cl, function(lib) { .libPaths(c(lib, .libPaths())); library(ebrahim.gof); TRUE }, RLIB)
clusterExport(cl, c("draw_lp", "one"))
res <- list()
for (i in seq_len(nrow(cells))) {
  cell <- c(as.list(cells[i, ]), id = i)
  m <- do.call(rbind, parLapply(cl, seq_len(REPS), one, cell = cell))
  ok <- is.finite(m[, "p"])
  rate <- mean(m[ok, "p"] < 0.05)
  res[[i]] <- data.frame(pi = cell$pi, tail = cell$tail, N = cell$N, reps = REPS, valid = sum(ok),
                         size = rate, mcse = sqrt(rate * (1 - rate) / sum(ok)),
                         share_df1 = mean(m[ok, "df"] == 1), all_score_form = all(m[ok, "score"] == 1),
                         reps_with_clamped_p = sum(m[, "clamped"] > 0),
                         gof_ref = GOF_REF)
  cat(sprintf("%s pi=%.2f: size %.4f (MCSE %.4f), %d valid, df=1 in %.3f\n", cell$tail, cell$pi, rate,
              res[[i]]$mcse, sum(ok), res[[i]]$share_df1))
}
stopCluster(cl)
out <- do.call(rbind, res)
write.csv(out, file.path(root, "results", "stukel_score_size.csv"), row.names = FALSE)
print(out, digits = 4)
