## size_check.R -- size of every external test when the predictions are the true risks (plan 4.5).
##
## The oracle predicts p* and the outcomes are Y ~ Bernoulli(p*), so every test's null holds exactly.
## 12 scenarios (shapes S1-S4 x event rate 3 %, 10 %, 30 %) at the plan's evaluation sizes
## N_eval = max(1000, ceiling(100 / pi)): 3,334 at 3 %, 1,000 at 10 % and 30 %. 2,000 replicates each.
##
##   Rscript R/size_check.R calibrate              solve the DGP intercepts and scale (once)
##   Rscript R/size_check.R time [--reps 3]        cost per replicate, sequential
##   Rscript R/size_check.R run  [--workers 12] [--reps 1:2000] [--scenario ...]
##   Rscript R/size_check.R summarize              -> results/size_check.csv
##
## The covariates and the four shapes follow plan 4.2 (p = 8: six continuous, x1 and x2 correlated at
## 0.5, two binary; oracle AUC about 0.80; S4 on the cloglog link). The main run's DGP lives in the
## Python engine; under the oracle the tests see only p* and X, so this check does not depend on its
## exact coefficients. The main run repeats the check per scenario through the oracle arm.
## Data are not stored: each replicate is regenerated from its seed, and its fingerprint is kept.

Sys.setenv(GOF_LAYER_NO_MAIN = "1")
local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/")) else
    file.path(Sys.getenv("TFM_REBUILD_ROOT"), "R")
  if (!exists("gof_battery")) sys.source(file.path(here, "gof_layer.R"), envir = globalenv())
})

SIZE_DIR    <- file.path(ROOT, "results", "gof_size")
SIZE_PARAMS <- file.path(ROOT, "results", "size_check_dgp.csv")
SIZE_OUT    <- file.path(ROOT, "results", "size_check.csv")
SIZE_REPS   <- 2000L

SHAPES <- c("S1", "S2", "S3", "S4")
PIS    <- c(0.03, 0.10, 0.30)
BETA   <- c(0.8, 0.6, 0.5, -0.5, 0.4, -0.3, 0.6, -0.4)   # direction of the linear part; scaled below

## DeepGOF-1 costs 10-13 s per call (B = 199, 28 covariate pairs), ten times the rest of the battery, so
## its size is checked at the two evaluation sizes on one shape (S3, the interaction scenario that
## localization and DeepGOF are aimed at) rather than on all twelve scenarios
DG_SCEN <- c("S3_pi10", "S3_pi03")

n_eval_of <- function(pi) max(1000L, as.integer(ceiling(100 / pi)))
size_scenarios <- function()
  do.call(rbind, lapply(SHAPES, function(s) data.frame(shape = s, pi = PIS, stringsAsFactors = FALSE)))
scen_name <- function(shape, pi) sprintf("%s_pi%02d", shape, round(100 * pi))

draw_X <- function(n) {
  z <- matrix(stats::rnorm(n * 6), n, 6)
  z[, 2] <- 0.5 * z[, 1] + sqrt(0.75) * z[, 2]          # x1 and x2 correlated at 0.5
  X <- cbind(z, stats::rbinom(n, 1, 0.5), stats::rbinom(n, 1, 0.3))
  colnames(X) <- paste0("x", 1:8)
  X
}

## the shape part of the score, before the intercept; s scales the linear part
shape_eta <- function(X, shape, s) {
  lin <- s * drop(X %*% BETA)
  switch(shape,
         S1 = lin,
         S2 = lin + 0.5 * (X[, 3]^2 - 1) + 0.5 * X[, 4]^3 / 3,            # two smooth non-linear terms
         S3 = lin + 0.8 * X[, 1] * X[, 5] + 0.8 * X[, 2] * X[, 7],        # two pairwise interactions
         S4 = lin)                                                       # same score, cloglog link
}
inv_link <- function(eta, shape) if (shape == "S4") 1 - exp(-exp(eta)) else stats::plogis(eta)

## the expected AUC of the oracle, with the outcomes integrated out: P(p_i > p_j) over pairs weighted by
## p_i (1 - p_j), i != j. Deterministic, so the root finder below sees a smooth function.
oracle_auc <- function(p) {
  p <- sort(p); q <- 1 - p
  below <- cumsum(q) - q                       # sum of (1 - p_j) over records with smaller p
  (sum(p * below)) / (sum(p) * sum(q) - sum(p * q))
}

## scale s for oracle AUC 0.80 at pi = 10 % (2e5 draws), then the intercept for every pi (1e6 draws)
calibrate_dgp <- function() {
  rows <- list()
  for (sh in SHAPES) {
    reset_rng(task_seed("size-calibrate", sh, "X"))
    Xa <- draw_X(2e5); Xb <- draw_X(1e6)
    icpt <- function(e, pi) stats::uniroot(function(a) mean(inv_link(a + e, sh)) - pi, c(-15, 10), tol = 1e-10)$root
    auc_at <- function(s) {
      e <- shape_eta(Xa, sh, s); a <- icpt(e, 0.10)
      oracle_auc(inv_link(a + e, sh))
    }
    s <- stats::uniroot(function(s) auc_at(s) - 0.80, c(0.05, 3), tol = 1e-4)$root
    eb <- shape_eta(Xb, sh, s)
    for (pi in PIS) {
      a <- icpt(eb, pi)
      rows[[paste(sh, pi)]] <- data.frame(shape = sh, pi = pi, n_eval = n_eval_of(pi), scale = s, intercept = a,
                                          check_prevalence = mean(inv_link(a + eb, sh)),
                                          check_auc_pi10 = if (pi == 0.10) auc_at(s) else NA_real_)
    }
  }
  out <- do.call(rbind, rows); rownames(out) <- NULL
  dir.create(dirname(SIZE_PARAMS), recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(out, SIZE_PARAMS)
  print(out)
  invisible(out)
}

## one replicate of one scenario: data from its own seed, oracle predictions, the whole battery
run_size_rep <- function(task, opts) {
  use_project_lib()
  sc <- scen_name(task$shape, task$pi)
  out_file <- file.path(SIZE_DIR, sc, sprintf("rep%04d.csv", task$rep))
  if (file.exists(out_file)) return(sprintf("%s/%d exists", sc, task$rep))
  tryCatch({
    pr <- task$par
    reset_rng(task_seed("size", sc, task$rep, "data"))
    X <- draw_X(pr$n_eval)
    pstar <- inv_link(pr$intercept + shape_eta(X, task$shape, pr$scale), task$shape)
    y <- stats::rbinom(length(pstar), 1L, pstar)
    fp <- data_fingerprint(y)
    sb <- paste("size", sc, task$rep, sep = "|")
    b <- gof_battery(y, pstar, X, seed_base = sb, do_lecessie = opts$lecessie, do_localize = opts$localize,
                     do_deepgof = opts$deepgof && task$scenario_dg, M_loc = opts$M_loc, B_dg = opts$B_dg)
    b <- cbind(scenario = sc, shape = task$shape, pi = task$pi, rep = task$rep, n_eval = length(y),
               events = sum(y), fingerprint = fp, gof_ref = GOF_REF, arm = "oracle", ytype = "oracle null",
               seed_base = sb, b)
    atomic_write(b, out_file)
    sprintf("%s/%d ok %.1fs", sc, task$rep, sum(b$secs, na.rm = TRUE))
  }, error = function(e) {
    writeLines(c(format(Sys.time()), conditionMessage(e)), sub("\\.csv$", ".error.txt", out_file))
    sprintf("%s/%d FAILED: %s", sc, task$rep, conditionMessage(e))
  })
}

size_tasks <- function(reps, scen = NULL) {
  par <- as.data.frame(data.table::fread(SIZE_PARAMS))
  if (!is.null(scen)) par <- par[scen_name(par$shape, par$pi) %in% scen, , drop = FALSE]
  tasks <- list()
  ## replicate-major order, so a partial run covers every scenario evenly
  for (r in reps) for (i in seq_len(nrow(par)))
    tasks[[length(tasks) + 1L]] <- list(shape = par$shape[i], pi = par$pi[i], rep = r, par = as.list(par[i, ]),
                                        scenario_dg = scen_name(par$shape[i], par$pi[i]) %in% DG_SCEN)
  done <- vapply(tasks, function(t) file.exists(file.path(SIZE_DIR, scen_name(t$shape, t$pi), sprintf("rep%04d.csv", t$rep))), TRUE)
  tasks[!done]
}

## rejection rate at .05 with its Monte Carlo SE; flagged outside [0.035, 0.065]
summarize_size <- function(alpha = 0.05, lo = 0.035, hi = 0.065) {
  f <- list.files(SIZE_DIR, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE)
  if (!length(f)) stop("no size-check results yet")
  d <- data.table::rbindlist(lapply(f, data.table::fread, colClasses = list(character = c("fingerprint", "note"))),
                             fill = TRUE)
  ## the same data twice would mean a reset random stream
  fpd <- d[, .(fp = fingerprint[1]), by = .(scenario, rep)]
  ndup <- sum(duplicated(fpd$fp))
  if (ndup) warning(ndup, " replicate(s) share a data fingerprint: check the seeds")
  d <- d[!is.na(d$p_value) | grepl("^(Not run|Failed)", d$note)]
  agg <- function(dd, by) dd[, {
    ok <- !is.na(p_value); R <- sum(ok); rej <- sum(p_value[ok] <= alpha)
    rate <- if (R) rej / R else NA_real_
    .(reps = R, not_run = sum(!ok), rejections = rej, rate = rate,
      mcse = if (R) sqrt(rate * (1 - rate) / R) else NA_real_,
      mean_events = mean(events), median_secs = stats::median(secs, na.rm = TRUE))
  }, by = by]
  per <- agg(d, c("n_eval", "scenario", "test", "family"))
  pooled <- agg(d, c("n_eval", "test", "family"))
  pooled[, scenario := sprintf("all shapes and rates at N_eval = %d", n_eval)]
  out <- rbind(per, pooled, use.names = TRUE)
  out[, lower := rate - 1.96 * mcse][, upper := rate + 1.96 * mcse]
  out[, flag := ifelse(is.na(rate), "not run", ifelse(rate < lo | rate > hi, "OUTSIDE [0.035, 0.065]", ""))]
  ## a part named by closed testing is not a level-alpha test of its own: under the global null its
  ## rate must sit below alpha, and the familywise rate is the "any part named" row
  out[grepl("^Localize .*\\(named\\)$", test),
      flag := ifelse(rate > hi, "ABOVE 0.065", "closed-testing component: below alpha by design; FWER = 'any part named'")]
  out[, fingerprint_duplicates := ndup]
  data.table::setorder(out, n_eval, scenario, family, test)
  data.table::fwrite(out, SIZE_OUT)
  message(sprintf("%d rows -> %s; flagged: %d; duplicate fingerprints: %d", nrow(out), SIZE_OUT,
                  sum(out$flag %in% c("OUTSIDE [0.035, 0.065]", "ABOVE 0.065")), ndup))
  invisible(out)
}

main_size <- function(a = commandArgs(TRUE)) {
  o <- parse_args(a)
  o$localize <- !identical(o$localize, "none"); o$deepgof <- !identical(o$deepgof, "none")
  use_project_lib()
  suppressPackageStartupMessages(library(data.table))
  if (o$cmd == "calibrate") return(invisible(calibrate_dgp()))
  if (!file.exists(SIZE_PARAMS)) calibrate_dgp()
  if (o$cmd == "summarize") return(invisible(summarize_size()))
  scen <- if (is.na(o$scenario)) NULL else strsplit(o$scenario, ",")[[1]]
  if (o$cmd == "time") {
    reps <- if (is.na(o$reps)) 1:2 else eval(parse(text = o$reps))
    for (t in size_tasks(reps, scen)) {
      t0 <- proc.time()[["elapsed"]]
      m <- run_size_rep(t, o)
      message(m, sprintf("  wall %.1fs", proc.time()[["elapsed"]] - t0))
    }
    return(invisible())
  }
  if (o$cmd != "run") stop("unknown command: ", o$cmd)
  reps <- if (is.na(o$reps)) seq_len(SIZE_REPS) else eval(parse(text = o$reps))
  tasks <- size_tasks(reps, scen)
  message(length(tasks), " size-check tasks to run")
  me <- file.path(ROOT, "R", "size_check.R")
  run_tasks(tasks, run_size_rep, o, pick_workers(o$workers), file.path(ROOT, "logs", "size_check.log"),
            sources = c(file.path(ROOT, "R", "gof_layer.R"), me))
  summarize_size()
}

if (sys.nframe() == 0L) main_size()
