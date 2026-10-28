## realdata_gof.R -- the external battery of gof_layer.R on the real-data stream (pre-registration
## 10.2-10.4). Everything is the simulation's layer (same ebrahim.gof build check, battery, seeds,
## resume and output format); two functions are wrapped:
##
##   parse_pred_file  also reads the localization covariates of 10.3 (columns l_*: age band, sex,
##                    region where present, severity score; categoricals dummy-coded) and attaches
##                    them to the covariate frame as attribute "X_loc"
##   gof_battery      runs localize.external on those covariates; le Cessie and DeepGOF-1 keep the
##                    full predictor matrix the arms saw (columns x_*), as in the simulation
##
##   Rscript R/realdata_gof.R run   [--keys ist_dd6,...] [--parts lc,lct,pl] [--ns 200,500]
##                                  [--reps 1:100] [--workers 12] [--B 99] [--arms ...]
##   Rscript R/realdata_gof.R check [--keys ...]
##
## Input:  results/realdata/<key>/preds/<setting>/<group>/<rep>.parquet (code/run_realdata.py)
## Output: results/realdata/<key>/gof/<setting>/<rep>.csv.gz, log logs/realdata/gof.log
## REALDATA_SMOKE=1 reads and writes under results/smoke/realdata instead.

local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/")) else NA
  root <- Sys.getenv("TFM_REBUILD_ROOT", if (!is.na(here) && basename(here) == "R") dirname(here) else ".")
  Sys.setenv(TFM_REBUILD_ROOT = normalizePath(root, winslash = "/"))
})
source(file.path(Sys.getenv("TFM_REBUILD_ROOT"), "R", "gof_layer.R"))

RD_RESULTS <- file.path(ROOT, "results", if (Sys.getenv("REALDATA_SMOKE") == "1") "smoke/realdata" else "realdata")
RD_KEYS <- c("ist_death14", "ist_dd6", "crash2_death28", "sepsis_death", "physionet_ihd")
## the arms tested by default: the side arms were dropped from all real data on 2026-10-23
## (DEVIATIONS D-05), so their columns, where an earlier pass wrote them, are not tested
RD_ARMS <- c("oracle", "lr", "flic", "ridge", "gam", "lgbm", "lgbm_raw", "tabpfn", "tabicl", "mitra",
             as.vector(outer(c("flic", "tabpfn", "tabicl", "mitra"),
                             c("r1_m100", "r2_m100", "r3_m100", "r1_m500", "r2_m500", "r3_m500"),
                             paste, sep = "__")))

## ---------------------------------------------------------------- wrapped functions ----

.parse_pred_file_layer <- parse_pred_file
parse_pred_file <- function(f, subset = NULL) {
  out <- .parse_pred_file_layer(f, subset)
  d <- read_any(f)
  meta <- attr(d, "meta")
  names(d) <- tolower(names(d))
  d <- d[tolower(d$set) == "eval", , drop = FALSE]
  d <- d[order(d$row), , drop = FALSE]
  lcol <- grep("^l_", names(d), value = TRUE)
  if (length(lcol) && !is.null(out$X)) {
    XL <- as.data.frame(lapply(d[, lcol, drop = FALSE], as.numeric))
    if (length(meta$loc_cat)) XL <- expand_categoricals(XL, meta$loc_cat)
    if (anyNA(XL)) stop("missing values in the localization covariates of ", f)
    attr(out$X, "X_loc") <- XL
  }
  out
}

.gof_battery_layer <- gof_battery
gof_battery <- function(y, p, X = NULL, seed_base, do_lecessie = TRUE, do_localize = TRUE,
                        do_deepgof = TRUE, M_loc = 999L, B_dg = 199L, dg_reading = "combined") {
  XL <- if (!is.null(X)) attr(X, "X_loc") else NULL
  res <- .gof_battery_layer(y, p, X, seed_base = seed_base, do_lecessie = do_lecessie, do_localize = FALSE,
                            do_deepgof = do_deepgof, M_loc = M_loc, B_dg = B_dg, dg_reading = dg_reading)
  if (do_localize && !is.null(XL)) res <- rbind(res, localize_rows(y, p, XL, seed_base, M_loc))
  ## a test that could not get memory is a failure of the machine, not a result: stop, so that
  ## nothing of this replicate is written and the next pass computes it again (run_one_rep keeps
  ## the message in <rep>.error.txt)
  oom <- grepl("cannot allocate|memory exhausted|std::bad_alloc", res$note, ignore.case = TRUE)
  if (any(oom)) stop("out of memory in ", paste(unique(res$test[oom]), collapse = ", "), ": ",
                     res$note[oom][1], call. = FALSE)
  res
}

## block (3) of gof_battery, on the 10.3 covariates (same call, same seed, same rows)
localize_rows <- function(y, p, XL, seed_base, M_loc) {
  y <- as.numeric(y)
  nclamp <- sum(p <= 1e-9 | p >= 1 - 1e-9)
  p <- clamp_p(p)
  XL <- clean_X(XL)
  if (is.null(XL)) return(one_row("Localize (any part named)", "Localize", note = "Not run: no covariates"))
  r <- timed(localize.external(y, p, XL, M = M_loc, seed = task_seed(seed_base, "localize"), plot = FALSE))
  if (inherits(r$val, "gof_fail")) {
    out <- one_row("Localize (any part named)", "Localize", note = paste("Failed:", r$val), secs = r$secs)
  } else {
    L <- r$val
    g <- names(L$single)
    named <- if (length(L$named)) paste(L$named, collapse = "+") else "none"
    out <- rbind(
      one_row("Localize (any part named)", "Localize", length(L$named), NA, min(L$adjusted),
              sprintf("named=%s; highest=%s; M=%d; covariates %s", named,
                      ifelse(is.na(L$highest), "none", L$highest), L$draws, paste(names(XL), collapse = "+")),
              r$secs),
      one_row(paste0("Localize ", g, " (named)"), "Localize", NA, NA, L$adjusted,
              ifelse(g %in% L$named, "named", "not named"), NA),
      one_row(paste0("Localize ", g, " (alone)"), "Localize", NA, NA, L$single, "single-group p", NA))
  }
  if (nclamp) out$note <- paste0(out$note, sprintf(" [%d p clamped]", nclamp))
  out
}

## ---------------------------------------------------------------- driver ----

rd_args <- function(a) {
  get <- function(flag, default) { i <- match(flag, a); if (is.na(i)) default else a[i + 1] }
  spl <- function(v) if (is.na(v)) character() else strsplit(v, ",")[[1]]
  list(cmd = if (length(a)) a[1] else "run",
       keys = { v <- spl(get("--keys", NA)); if (length(v)) v else RD_KEYS },
       parts = { v <- spl(get("--parts", NA)); if (length(v)) v else c("lc", "pl") },
       ns = spl(get("--ns", NA)),
       reps = get("--reps", NA),
       workers = as.integer(get("--workers", 12)),
       B_dg = as.integer(get("--B", 99)),
       arms = { v <- tolower(spl(get("--arms", NA))); if (length(v)) v else RD_ARMS })
}

rd_settings <- function(key, parts, ns) {
  d <- file.path(RD_RESULTS, key, "preds")
  if (!dir.exists(d)) return(character())
  s <- list.dirs(d, recursive = FALSE, full.names = FALSE)
  part <- ifelse(grepl("__lct_n", s), "lct", ifelse(grepl("__lc_n", s), "lc", "pl"))
  n <- sub(".*_n", "", s)
  s[part %in% parts & (!length(ns) | n %in% ns)]
}

## learning-curve replicates of one setting share the external outcomes, so equal fingerprints are
## expected there; a plasmode setting must have one outcome vector per replicate
rd_check <- function(keys) {
  for (key in keys) {
    d <- file.path(RD_RESULTS, key, "gof")
    for (s in list.dirs(d, recursive = FALSE, full.names = FALSE)) {
      f <- list.files(file.path(d, s), pattern = "\\.csv\\.gz$", full.names = TRUE)
      err <- list.files(file.path(d, s), pattern = "\\.error\\.txt$")
      fp <- vapply(f, function(x) as.character(fread_any(x, select = "fingerprint", nrows = 1,
                                                          colClasses = list(character = "fingerprint"))$fingerprint), "")
      flag <- if (grepl("__pl", s) && anyDuplicated(fp)) "  DUPLICATE OUTCOMES" else
        if (grepl("__lc", s) && length(unique(fp)) > 1) "  OUTCOMES DIFFER ACROSS REPLICATES" else ""   # lc and lct
      message(sprintf("%s: %d replicate files, %d unique outcome fingerprints, %d open errors%s",
                      s, length(f), length(unique(fp)), length(err), flag))
    }
  }
}

rd_main <- function(args = commandArgs(TRUE)) {
  o <- rd_args(args)
  use_project_lib()
  if (o$cmd == "check") return(invisible(rd_check(o$keys)))
  if (o$cmd != "run") stop("unknown command: ", o$cmd)
  log_file <- file.path(ROOT, "logs", "realdata", if (Sys.getenv("REALDATA_SMOKE") == "1") "gof_smoke.log" else "gof.log")
  workers <- pick_workers(o$workers)
  for (key in o$keys) {
    pred_dir <- file.path(RD_RESULTS, key, "preds")
    tasks <- list()
    for (s in rd_settings(key, o$parts, o$ns)) {
      reps <- list_rep_files(s, pred_dir)
      if (!is.na(o$reps)) {
        r <- eval(parse(text = o$reps))
        reps <- reps[as.integer(gsub("[^0-9]", "", names(reps))) %in% r]
      }
      for (r in names(reps)) tasks[[length(tasks) + 1L]] <- list(scenario = s, rep = r, files = reps[[r]])
    }
    if (!length(tasks)) next
    ## the simulation's options: battery on Y and on each arm's Y*, le Cessie with the covariates,
    ## localization on oracle / FLIC / GAM / TFMs, DeepGOF-1 (B = 99) on FLIC and the main TFMs (Y only)
    ## the IST temporal sensitivity split is tested on its R0 predictions only (DEVIATIONS D-05):
    ## repair (10.3) belongs to the primary split; the temporal ladder columns are still scored
    arms <- if (identical(o$parts, "lct")) o$arms[!grepl("__", o$arms, fixed = TRUE)] else o$arms
    opts <- list(null = TRUE, lecessie = TRUE, localize = "plan", deepgof = "plan", dg_null = FALSE,
                 M_loc = 999L, B_dg = o$B_dg, arms = arms, subset = NULL, preds = pred_dir,
                 out = file.path(RD_RESULTS, key, "gof"))
    message(sprintf("%s: %d replicate tasks", key, length(tasks)))
    run_tasks(tasks, run_one_rep, opts, workers, log_file,
              sources = file.path(ROOT, "R", "realdata_gof.R"))
  }
  rd_check(o$keys)
}

if (sys.nframe() == 0L) rd_main()
