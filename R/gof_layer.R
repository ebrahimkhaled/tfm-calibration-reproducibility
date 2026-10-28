## gof_layer.R -- the external-validation goodness-of-fit layer (plan section 4.5).
##
## Reads the frozen per-replicate predictions in results/preds/<scenario>/... and, for every arm,
## runs the external battery of ebrahim.gof 2.10.0 (release-2.10.0 commit 19480aa, installed in rebuild/rlib)
## on the evaluation sample, twice:
##   ytype = "real"  the outcomes Y that were drawn from the true risk p*
##   ytype = "ystar" an exact null Y* ~ Bernoulli(p-hat), the size of each test on that arm's
##                   own prediction distribution
## and writes one long gzip-compressed csv per replicate to results/gof/<scenario>/<rep>.csv.gz
## (older uncompressed <rep>.csv files are still read, and rewritten compressed when added to).
##
##   Rscript R/gof_layer.R run [--scenario S1_pi10_n200,...] [--workers 12] [--reps 1:200]
##                             [--no-null] [--no-lecessie] [--localize plan|all|none]
##                             [--deepgof plan|all|none] [--deepgof-null] [--M 999] [--B 199]
##                             [--arms oracle,flic,...]   only these arms (e.g. the oracle on reps 201-500)
##                             [--preds DIR] [--out DIR] [--subset e50|e200]
##   Rscript R/gof_layer.R check [--scenario ...]      fingerprints and coverage of the outputs
##
## The oracle size study lives in size_check.R, which sources this file for the battery.
## Input contract, output columns and seeds: see R/README.md.

## ---------------------------------------------------------------- paths and library ----

gof_root <- function() {
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  here <- if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/")) else
    normalizePath(".", winslash = "/")
  ## the script sits in rebuild/R; the root is one level up
  if (basename(here) == "R") dirname(here) else here
}
ROOT <- Sys.getenv("TFM_REBUILD_ROOT", gof_root())
RLIB <- Sys.getenv("TFM_RLIB", file.path(ROOT, "rlib"))
GOF_REF <- "2.10.0@19480aa"
GOF_VERSION <- "2.10.0"
GOF_COMMIT <- "19480aa"
## hash of every function of the installed namespace (formals and body, deparsed), for the build of
## commit 19480aa in rebuild/rlib under R 4.4.1. A build of any other commit gives another hash, and
## the layer refuses to run. R/verify_gof_build.R checks the build against the commit's sources and
## prints this value.
GOF_CODE_HASH <- "69f11ec8034bc66bb2f99c79c5d0913ca0dcd58bff32fe0da4e5951fbde5216e"

gof_code_hash <- function() {
  ns <- asNamespace("ebrahim.gof")
  fns <- sort(Filter(function(k) is.function(get(k, envir = ns)), ls(ns, all.names = TRUE)))
  txt <- vapply(fns, function(k) {
    f <- get(k, envir = ns)
    paste(c(k, deparse(formals(f)), deparse(body(f))), collapse = "\n")
  }, "")
  digest::digest(paste(txt, collapse = "\n"), algo = "sha256", serialize = FALSE)
}

.gof_checked <- new.env()

use_project_lib <- function() {
  if (dir.exists(RLIB)) .libPaths(unique(c(RLIB, .libPaths())))
  suppressPackageStartupMessages({
    library(ebrahim.gof)
    library(digest)
  })
  where <- dirname(find.package("ebrahim.gof"))
  if (normalizePath(where, winslash = "/") != normalizePath(RLIB, winslash = "/", mustWork = FALSE))
    stop("ebrahim.gof was loaded from ", where, ", not from the project library ", RLIB,
         ". Install it there from commit ", GOF_REF, " first (see R/README.md).", call. = FALSE)
  for (f in c("run.all.external", "localize.external", "deepgof1.external", "edge.gof"))
    if (!exists(f, mode = "function")) stop("ebrahim.gof has no ", f, "(): wrong build", call. = FALSE)
  if (as.character(utils::packageVersion("ebrahim.gof")) != GOF_VERSION)
    stop("ebrahim.gof ", as.character(utils::packageVersion("ebrahim.gof")), " in ", RLIB, "; the study uses ",
         GOF_VERSION, call. = FALSE)
  ## the version number alone does not tell the 2.10.0 commits apart (2c77f3c had the likelihood-ratio
  ## Stukel row): the code itself must be the pinned commit's
  if (is.null(.gof_checked$hash)) .gof_checked$hash <- gof_code_hash()
  if (!identical(.gof_checked$hash, GOF_CODE_HASH))
    stop("ebrahim.gof in ", RLIB, " is not the build of commit ", GOF_COMMIT, " (code hash ",
         substr(.gof_checked$hash, 1, 12), ", expected ", substr(GOF_CODE_HASH, 1, 12),
         "). Install that commit (see R/README.md) and check it with R/verify_gof_build.R.", call. = FALSE)
  invisible(TRUE)
}

## ---------------------------------------------------------------- seeds and fingerprints ----

## one integer seed per (what, scenario, rep, arm, ...) string: the same task always gets the same
## stream, whichever worker runs it and in whatever order
task_seed <- function(...) {
  h <- digest::digest(paste(..., sep = "|"), algo = "xxhash32", serialize = FALSE)
  strtoi(substr(h, 1, 7), 16L)
}

## every seed goes through here, so a package that switched the generator (or called set.seed itself)
## cannot leak into the next draw
reset_rng <- function(seed) {
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
}

## fingerprint of an evaluation sample: the 0/1 outcomes only. Rounded floats are not stable between
## float32 and float64 copies of the same data (a value near a rounding boundary flips), while a
## vector of a thousand or more Bernoulli outcomes is unique to its data set; two replicates with the
## same outcomes mean a reset random stream.
data_fingerprint <- function(y) digest::digest(as.integer(y), algo = "xxhash64")

## an arm's predictions exactly as read; a change means the arm was refitted and its tests are redone
pred_hash <- function(p) digest::digest(as.numeric(p), algo = "xxhash64")

## ---------------------------------------------------------------- the battery ----

## predictions of exactly 0 or 1 (float32 rounding in the TFM outputs) are moved just inside (0, 1);
## the count is reported in the Note of every row
clamp_p <- function(p, eps = 1e-9) pmin(pmax(as.numeric(p), eps), 1 - eps)

## covariates for localize / deepgof / le Cessie: numeric, named, no constant column
clean_X <- function(X) {
  if (is.null(X)) return(NULL)
  X <- as.data.frame(X)
  X <- X[, vapply(X, function(v) is.numeric(v) && length(unique(v)) > 1L, TRUE), drop = FALSE]
  if (!ncol(X)) return(NULL)
  if (is.null(names(X)) || anyDuplicated(names(X))) names(X) <- paste0("x", seq_len(ncol(X)))
  X
}

one_row <- function(test, family, stat = NA_real_, df = NA_real_, pv = NA_real_, note = "", secs = NA_real_)
  data.frame(test = test, family = family, statistic = as.numeric(stat), df = as.numeric(df),
             p_value = as.numeric(pv), note = note, secs = as.numeric(secs), stringsAsFactors = FALSE)

timed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  val <- tryCatch(expr, error = function(e) structure(conditionMessage(e), class = "gof_fail"))
  list(val = val, secs = proc.time()[["elapsed"]] - t0)
}

## all tests on one (y, p) pair. seed_base is a string; each stochastic test gets its own seed from it.
## G = 10 as in the plan; EDGE at G = auto comes with the battery.
gof_battery <- function(y, p, X = NULL, seed_base, do_lecessie = TRUE, do_localize = TRUE,
                        do_deepgof = TRUE, M_loc = 999L, B_dg = 199L, dg_reading = "combined") {
  y <- as.numeric(y)
  nclamp <- sum(p <= 1e-9 | p >= 1 - 1e-9)
  p <- clamp_p(p)
  X <- clean_X(X)
  out <- list()

  ## (1) run.all.external: HL (chi-square G), Cox, CITL, Spiegelhalter, GiViTI, EDGE G=10 and auto,
  ##     Stukel (offset), le Cessie (when X is given)
  r <- timed(suppressWarnings(run.all.external(y, p, G = 10, X = if (do_lecessie) X else NULL,
                                               include_slow = do_lecessie && !is.null(X))))
  if (inherits(r$val, "gof_fail")) {
    out$bat <- one_row("run.all.external", "Battery", note = paste("Failed:", r$val), secs = r$secs)
  } else {
    b <- as.data.frame(r$val)
    out$bat <- one_row(b$Test, b$Family, b$Statistic, b$df, b$p_value, b$Note, c(r$secs, rep(NA, nrow(b) - 1L)))
  }

  ## (2) edge.gof in external mode, both partitions (the same statistic as the battery's EDGE rows,
  ##     kept as a cross-check of the two code paths)
  r <- timed(suppressWarnings(edge.gof(y = y, predicted_probs = p, external = TRUE)))
  if (inherits(r$val, "gof_fail")) {
    out$edge <- one_row("edge.gof", "Directed", note = paste("Failed:", r$val), secs = r$secs)
  } else {
    e <- r$val
    out$edge <- one_row(paste0("edge.gof ", e$Partition), "Directed", e$Test_Statistic, e$df, e$p_value,
                        paste("role", e$Role), c(r$secs, rep(NA, nrow(e) - 1L)))
  }

  ## (3) localize.external: which parts are named (closed testing at FWER .05, Monte Carlo reference)
  if (do_localize && !is.null(X)) {
    r <- timed(localize.external(y, p, X, M = M_loc, seed = task_seed(seed_base, "localize"), plot = FALSE))
    if (inherits(r$val, "gof_fail")) {
      out$loc <- one_row("Localize (any part named)", "Localize", note = paste("Failed:", r$val), secs = r$secs)
    } else {
      L <- r$val
      g <- names(L$single)
      named <- if (length(L$named)) paste(L$named, collapse = "+") else "none"
      out$loc <- rbind(
        one_row("Localize (any part named)", "Localize", length(L$named), NA, min(L$adjusted),
                sprintf("named=%s; highest=%s; M=%d", named, ifelse(is.na(L$highest), "none", L$highest), L$draws),
                r$secs),
        one_row(paste0("Localize ", g, " (named)"), "Localize", NA, NA, L$adjusted,
                ifelse(g %in% L$named, "named", "not named"), NA),
        one_row(paste0("Localize ", g, " (alone)"), "Localize", NA, NA, L$single, "single-group p", NA))
    }
  }

  ## (4) DeepGOF-1 external: exact Monte Carlo test of calibration within covariate subgroups
  if (do_deepgof && !is.null(X)) {
    reset_rng(task_seed(seed_base, "deepgof"))
    r <- timed(deepgof1.external(y, p, X, B = B_dg, reading = dg_reading))
    if (inherits(r$val, "gof_fail")) {
      out$dg <- one_row("DeepGOF-1 external (combined)", "DeepGOF", note = paste("Failed:", r$val), secs = r$secs)
    } else {
      d <- r$val
      cmp <- if (!is.null(d$components)) d$components else if (identical(dg_reading, "axes"))
        c(axes = d$p.value, allpairs = NA) else c(axes = NA, allpairs = NA)
      ## the first row is the reading that was asked for; with reading = "axes" (scenario T2, 30
      ## covariates) the all-pairs component is not computed and its row stays empty
      lab <- if (dg_reading == "combined") "combined" else paste(dg_reading, "reading")
      out$dg <- rbind(
        one_row(sprintf("DeepGOF-1 external (%s)", lab), "DeepGOF", d$statistic, NA, d$p.value,
                sprintf("B=%d; reading %s; axes %s", d$B, dg_reading, paste(d$axes, collapse = "x")), r$secs),
        one_row("DeepGOF-1 external (axes)", "DeepGOF", NA, NA, cmp[["axes"]], "", NA),
        one_row("DeepGOF-1 external (allpairs)", "DeepGOF", NA, NA, cmp[["allpairs"]], "", NA))
    }
  }

  res <- do.call(rbind, out)
  rownames(res) <- NULL
  ## the automatic partition has n-dependent G; keep the number in the note and one test name for all n
  auto <- grepl("G=auto, [0-9]+\\)$|default \\(G = [0-9]+\\)$", res$test)
  res$note[auto] <- paste0(res$note[auto], sprintf("; G = %s", sub(".*[ ,]([0-9]+)\\)$", "\\1", res$test[auto])))
  res$test <- sub("G=auto, [0-9]+\\)$", "G=auto)", res$test)
  res$test <- sub("default \\(G = [0-9]+\\)$", "default (G = auto)", res$test)
  if (nclamp) res$note <- paste0(res$note, sprintf(" [%d p clamped]", nclamp))
  res
}

## ---------------------------------------------------------------- reading the predictions ----

PRED_DIR <- file.path(ROOT, "results", "preds")
GOF_DIR  <- file.path(ROOT, "results", "gof")
## the whole evaluation sample goes to results/gof; a subsample run (--subset e50) to results/gof_e50
gof_out_dir <- function(subset = NULL, base = GOF_DIR) if (is.null(subset)) base else paste0(base, "_", subset)

## fread for .csv and .csv.gz alike. data.table reads .gz files only through R.utils, which is not in
## rlib, so a compressed file is decompressed by base R first.
fread_any <- function(f, ...) {
  if (grepl("\\.gz$", f, ignore.case = TRUE)) {
    con <- gzfile(f, "rt"); on.exit(close(con))
    return(data.table::fread(text = readLines(con), showProgress = FALSE, ...))
  }
  data.table::fread(f, showProgress = FALSE, ...)
}

## a data frame; for the engine's parquet files the "tfmcal" metadata (column types, fingerprints) rides
## along as attribute "meta"
read_any <- function(f) {
  if (grepl("\\.parquet$", f, ignore.case = TRUE)) {
    if (requireNamespace("arrow", quietly = TRUE)) {
      t <- arrow::read_parquet(f, as_data_frame = FALSE)
      d <- as.data.frame(t)
      js <- t$metadata$tfmcal
      if (!is.null(js) && requireNamespace("jsonlite", quietly = TRUE))
        attr(d, "meta") <- tryCatch(jsonlite::fromJSON(js, simplifyVector = TRUE), error = function(e) NULL)
      return(d)
    }
    if (requireNamespace("nanoparquet", quietly = TRUE)) return(as.data.frame(nanoparquet::read_parquet(f)))
    stop("cannot read parquet: install 'arrow' or 'nanoparquet' into rebuild/rlib")
  }
  as.data.frame(fread_any(f))
}

## categorical covariates (the engine stores T3's 6-level x8 as an integer code) become numeric dummies,
## first level as reference, so that localization, DeepGOF-1 and le Cessie do not read the code as a
## continuous variable
expand_categoricals <- function(X, cat_cols) {
  cat_cols <- intersect(tolower(cat_cols), names(X))
  for (k in cat_cols) {
    f <- factor(X[[k]])
    if (nlevels(f) > 1) {
      dm <- stats::model.matrix(~ f)[, -1, drop = FALSE]
      colnames(dm) <- paste0(k, "_", levels(f)[-1])
      X <- cbind(X[, setdiff(names(X), k), drop = FALSE], as.data.frame(dm))
    } else X[[k]] <- NULL
  }
  X
}

## the files of one scenario, grouped by replicate id (the file name without extension). Arms may sit
## in separate files or sub-folders (Python TFMs and R classical arms write at different times).
list_rep_files <- function(scenario, pred_dir = PRED_DIR) {
  f <- list.files(file.path(pred_dir, scenario), recursive = TRUE, full.names = TRUE,
                  pattern = "\\.(parquet|csv|csv\\.gz)$", ignore.case = TRUE)
  f <- f[!grepl("(^|/)_", f)]                     # files or folders starting with "_" are skipped
  if (!length(f)) return(list())
  id <- sub("\\.(parquet|csv|csv\\.gz)$", "", basename(f), ignore.case = TRUE)
  split(f, id)
}

## one file -> a list(y, pstar, X, P = matrix of arms, Ystar = matrix or NULL, row)
## subset: NULL for the whole evaluation sample, or "e50" / "e200" for the engine's nested subsamples
## with about 50 or 200 expected events (column in_sub_<subset>)
parse_pred_file <- function(f, subset = NULL) {
  d <- read_any(f)
  meta <- attr(d, "meta")
  names(d) <- tolower(names(d))
  if ("set" %in% names(d)) d <- d[tolower(d$set) %in% c("eval", "evaluation", "test"), , drop = FALSE]
  if (!is.null(subset)) {
    sc <- paste0("in_sub_", subset)
    if (!sc %in% names(d)) stop("no column ", sc, " in ", f)
    keep <- as.logical(d[[sc]]); keep[is.na(keep)] <- FALSE
    if (!any(keep)) stop("subsample ", subset, " is empty in ", f)
    d <- d[keep, , drop = FALSE]
  }
  if (!nrow(d)) stop("no evaluation rows in ", f)
  ## long format (one row per record and arm) -> wide
  if (all(c("arm", "p_hat") %in% names(d))) {
    key <- if ("row" %in% names(d)) "row" else stop("long format needs a 'row' column: ", f)
    arms <- unique(d$arm)
    base <- d[d$arm == arms[1], setdiff(names(d), c("arm", "p_hat", "y_star", "ystar")), drop = FALSE]
    base <- base[order(base[[key]]), , drop = FALSE]
    for (a in arms) {
      da <- d[d$arm == a, , drop = FALSE]; da <- da[order(da[[key]]), , drop = FALSE]
      stopifnot(identical(da[[key]], base[[key]]))
      base[[paste0("p_", a)]] <- da$p_hat
      ys <- intersect(c("y_star", "ystar"), names(da))
      if (length(ys)) base[[paste0("ystar_", a)]] <- da[[ys[1]]]
    }
    d <- base
  }
  if ("row" %in% names(d)) d <- d[order(d$row), , drop = FALSE]
  if (!"y" %in% names(d)) stop("no 'y' column in ", f)
  pcol <- setdiff(grep("^p_", names(d), value = TRUE), c("p_star", "p_oracle", "p_true"))
  scol <- intersect(c("p_star", "p_oracle", "p_true"), names(d))
  xcol <- grep("^x[0-9]+$|^x_", names(d), value = TRUE)
  ycol <- grep("^ystar_", names(d), value = TRUE)
  X <- if (length(xcol)) as.data.frame(lapply(d[, xcol, drop = FALSE], as.numeric)) else NULL
  if (!is.null(X) && length(meta$col_types$cat)) X <- expand_categoricals(X, meta$col_types$cat)
  list(row = if ("row" %in% names(d)) d$row else seq_len(nrow(d)),
       p_raw = length(xcol),          # predictors as stored, before categoricals become dummies
       y = as.integer(d$y),
       pstar = if (length(scol)) as.numeric(d[[scol[1]]]) else NULL,
       X = X,
       engine_fp = if (is.null(subset) && !is.null(meta$fingerprints$eval)) meta$fingerprints$eval else NA_character_,
       P = if (length(pcol)) setNames(lapply(pcol, function(k) as.numeric(d[[k]])), sub("^p_", "", pcol)) else list(),
       Ystar = if (length(ycol)) setNames(lapply(ycol, function(k) as.integer(d[[k]])), sub("^ystar_", "", ycol)) else list())
}

## merge the files of one replicate; every file must hold the same evaluation sample
read_rep <- function(files, subset = NULL) {
  parts <- lapply(files, parse_pred_file, subset = subset)
  ref <- parts[[1]]
  for (k in seq_along(parts)[-1]) {
    q <- parts[[k]]
    if (!is.na(q$engine_fp) && !is.na(ref$engine_fp) && q$engine_fp != ref$engine_fp)
      stop("engine fingerprints of the evaluation set differ between files of one replicate: ", files[k])
    if (is.na(ref$engine_fp)) ref$engine_fp <- q$engine_fp
    if (!identical(q$row, ref$row) || !identical(q$y, ref$y))
      stop("evaluation rows or outcomes differ between ", basename(dirname(files[1])), "/", basename(files[1]),
           " and ", basename(dirname(files[k])), "/", basename(files[k]))
    if (!is.null(q$pstar) && !is.null(ref$pstar) && max(abs(q$pstar - ref$pstar)) > 1e-6)
      stop("p_star differs between files of the same replicate: ", files[k])
    if (is.null(ref$pstar)) ref$pstar <- q$pstar
    if (is.null(ref$X)) { ref$X <- q$X; ref$p_raw <- q$p_raw }
    dup <- intersect(names(ref$P), names(q$P))
    if (length(dup)) stop("arm(s) ", paste(dup, collapse = ", "), " appear in two files of one replicate")
    ref$P <- c(ref$P, q$P); ref$Ystar <- c(ref$Ystar, q$Ystar)
  }
  if (!is.null(ref$pstar)) ref$P <- c(list(oracle = ref$pstar), ref$P)   # the oracle row of the plan
  ref$fingerprint <- data_fingerprint(ref$y)
  ref
}

## ---------------------------------------------------------------- which arm gets which test ----

## plan 4.5: DeepGOF-1 on FLIC and the three TFMs; localization on FLIC, GAM, the TFMs and the
## oracle (the plan names S3 and T3; at about 0.5 s a call it runs in every scenario). "all" runs a
## test on every arm, "none" on no arm. Both are exact Monte Carlo tests, so their size under the
## exact null Y* holds by construction: DeepGOF-1 (10-13 s a call) runs on the real Y only unless
## --deepgof-null is given; its size is checked once, in size_check.R.
## Recalibration-ladder columns (p_<arm>__R{1,2,3}_m{100,500}, plan 4.6) get the battery and edge.gof on
## the real Y only: the test-guided rule needs the Cox and EDGE rows after each level, not their size.
## The legacy TabPFN and the T = 1.0 sensitivity arm are side arms: no DeepGOF-1.
TFM_PAT <- "tabpfn|mitra|tabicl"

## all column pairs cost about 230 s a call at 30 covariates; above 12 predictors only the axes are read.
## In the core design this picks out exactly T2 (p = 30); T3 has 8 predictors (13 columns after its
## factor is dummy-coded) and keeps the combined reading.
dg_reading_for <- function(scenario, p_raw)
  if (grepl("^T2_", scenario) || (!is.na(p_raw) && p_raw > 12)) "axes" else "combined"
is_ladder <- function(arm) grepl("__", arm, fixed = TRUE)
arm_wants <- function(arm, rule, test) {
  if (rule == "none") return(FALSE)
  if (is_ladder(arm)) return(FALSE)
  if (rule == "all") return(TRUE)
  if (test == "localize") return(grepl(paste0("^oracle$|flic|gam|", TFM_PAT), arm, ignore.case = TRUE))
  grepl(paste0("flic|", TFM_PAT), arm, ignore.case = TRUE) && !grepl("legacy|_t1$", arm, ignore.case = TRUE)
}

## ---------------------------------------------------------------- one replicate ----

read_gof_csv <- function(f)
  as.data.frame(fread_any(f, colClasses = list(character = c("scenario", "rep", "fingerprint", "note",
                                                              "pred_hash"))))

## the GOF file of a replicate: <rep>.csv.gz, or an older uncompressed <rep>.csv
gof_file_of <- function(out_dir, scenario, rep) {
  gz <- file.path(out_dir, scenario, paste0(rep, ".csv.gz"))
  plain <- file.path(out_dir, scenario, paste0(rep, ".csv"))
  if (!file.exists(gz) && file.exists(plain)) plain else gz
}

atomic_write <- function(d, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp", Sys.getpid())
  ## the temporary name does not end in .gz, so the compression is set explicitly
  data.table::fwrite(d, tmp, compress = if (grepl("\\.gz$", path)) "gzip" else "none")
  if (!file.rename(tmp, path)) { file.copy(tmp, path, overwrite = TRUE); unlink(tmp) }
}

run_one_rep <- function(task, opts) {
  use_project_lib()
  scenario <- task$scenario; rep <- task$rep
  sub <- opts$subset
  out_dir <- gof_out_dir(sub, opts$out)
  in_file <- gof_file_of(out_dir, scenario, rep)
  out_file <- file.path(out_dir, scenario, paste0(rep, ".csv.gz"))
  err_file <- file.path(out_dir, scenario, paste0(rep, ".error.txt"))
  res <- tryCatch({
    d <- read_rep(task$files, subset = sub)
    old <- if (file.exists(in_file)) read_gof_csv(in_file) else NULL
    if (!is.null(old) && nrow(old) && any(old$fingerprint != d$fingerprint)) {
      ## the predictions were regenerated since the last run: keep the old file aside, start again
      file.rename(in_file, paste0(in_file, ".stale_", format(Sys.time(), "%Y%m%d%H%M%S")))
      old <- NULL
    }
    ph <- vapply(d$P, pred_hash, "")
    if (!is.null(old) && nrow(old)) {
      ## an arm whose predictions changed (refitted, new checkpoint) is redone; its old rows are kept aside
      changed <- !is.na(old$pred_hash) & old$arm %in% names(ph) & old$pred_hash != ph[old$arm]
      if (any(changed)) {
        atomic_write(old[changed, , drop = FALSE],
                     paste0(out_file, ".stale_", format(Sys.time(), "%Y%m%d%H%M%S"), ".gz"))
        old <- old[!changed, , drop = FALSE]
      }
    }
    done <- if (is.null(old)) character() else unique(paste(old$arm, old$ytype))
    ## "ystar", not "null": pandas reads the string "null" as a missing value
    ytypes <- if (isTRUE(opts$null)) c("real", "ystar") else "real"
    new <- list()
    arms_here <- names(d$P)
    if (length(opts$arms)) arms_here <- intersect(arms_here, opts$arms)
    ## DeepGOF-1 uses the axes reading only when the design has more than 12 predictors: scenario T2
    ## (30 covariates, 435 pairs) and the random designs with p > 12 (pre-registration 5.3 and 3.9)
    dg_reading <- dg_reading_for(scenario, d$p_raw)
    for (arm in arms_here) for (yt in ytypes) {
      if (paste(arm, yt) %in% done) next
      if (yt == "ystar" && is_ladder(arm)) next
      p <- d$P[[arm]]
      if (length(p) != length(d$y) || anyNA(p)) {
        new[[paste(arm, yt)]] <- cbind(one_row("predictions", "Input", note = "Not run: missing or NA predictions"),
                                       arm = arm, ytype = yt, ysource = NA, events_used = NA, seed_base = NA,
                                       pred_hash = ph[[arm]])
        next
      }
      sb <- paste("gof", if (is.null(sub)) "eval" else sub, scenario, rep, arm, yt, sep = "|")
      if (yt == "real") { yy <- d$y; ysrc <- "Y" } else if (!is.null(d$Ystar[[arm]])) {
        yy <- d$Ystar[[arm]]; ysrc <- "Ystar stored"
      } else {
        reset_rng(task_seed(sb, "ystar")); yy <- stats::rbinom(length(p), 1L, clamp_p(p)); ysrc <- "Ystar drawn in R"
      }
      b <- gof_battery(yy, p, d$X, seed_base = sb,
                       do_lecessie = opts$lecessie,
                       do_localize = arm_wants(arm, opts$localize, "localize"),
                       do_deepgof  = arm_wants(arm, opts$deepgof, "deepgof") && (yt == "real" || opts$dg_null),
                       M_loc = opts$M_loc, B_dg = opts$B_dg, dg_reading = dg_reading)
      new[[paste(arm, yt)]] <- cbind(b, arm = arm, ytype = yt, ysource = ysrc, events_used = sum(yy), seed_base = sb,
                                     pred_hash = ph[[arm]])
    }
    if (length(new)) {
      nd <- do.call(rbind, new)
      nd <- cbind(scenario = scenario, rep = rep, n_eval = length(d$y), events = sum(d$y),
                  fingerprint = d$fingerprint, engine_fingerprint = d$engine_fp,
                  eval_set = if (is.null(sub)) "eval" else sub, gof_ref = GOF_REF, nd)
      if (!is.null(old)) for (k in setdiff(names(nd), names(old))) old[[k]] <- NA   # older files, fewer columns
      all <- if (is.null(old)) nd else rbind(old[, names(nd)], nd)
      atomic_write(all, out_file)
      if (in_file != out_file && file.exists(in_file)) unlink(in_file)   # the old .csv, now compressed
    }
    if (file.exists(err_file)) file.rename(err_file, paste0(err_file, ".resolved"))
    sprintf("%s/%s ok (%d new arm-y pairs)", scenario, rep, length(new))
  }, error = function(e) {
    dir.create(dirname(err_file), recursive = TRUE, showWarnings = FALSE)
    writeLines(c(format(Sys.time()), conditionMessage(e), paste(task$files, collapse = "\n")), err_file)
    sprintf("%s/%s FAILED: %s", scenario, rep, conditionMessage(e))
  })
  res
}

## ---------------------------------------------------------------- workers ----

## leave cores free for the author's other jobs: never more than 12, and fewer when the CPU is busy
pick_workers <- function(requested = 12L) {
  # GOF_FORCE_WORKERS fixes the count outright (used once the model fits are finished and the
  # machine is otherwise idle); the results do not depend on it, every task has its own seed
  forced <- suppressWarnings(as.integer(Sys.getenv("GOF_FORCE_WORKERS", "")))
  if (!is.na(forced) && forced > 0) {
    message(sprintf("workers fixed by GOF_FORCE_WORKERS: using %d workers", forced))
    return(forced)
  }
  # one reading can catch a momentary spike, so take the lowest of three
  load <- tryCatch({
    vals <- vapply(1:3, function(i) {
      if (i > 1) Sys.sleep(2)
      v <- suppressWarnings(system("wmic cpu get loadpercentage", intern = TRUE))
      mean(as.numeric(gsub("[^0-9]", "", v[grepl("[0-9]", v)])))
    }, 0)
    min(vals)
  }, error = function(e) NA)
  cores <- parallel::detectCores()
  free <- if (is.na(load)) cores - 4L else floor(cores * (1 - load / 100)) - 2L
  w <- max(1L, min(as.integer(requested), 12L, free))
  message(sprintf("CPU load %s%%, %d cores: using %d workers", format(load), cores, w))
  w
}

## sources: the R files every worker loads (size_check.R adds itself)
run_tasks <- function(tasks, fun, opts, workers, log_file, sources = file.path(ROOT, "R", "gof_layer.R")) {
  dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("%s start: %d tasks, %d workers\n", format(Sys.time()), length(tasks), workers),
      file = log_file, append = TRUE)
  if (!length(tasks)) return(invisible(character()))
  cl <- parallel::makePSOCKcluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterCall(cl, function(root, files) {
    Sys.setenv(TFM_REBUILD_ROOT = root, OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
               GOF_LAYER_NO_MAIN = "1")
    for (f in files) sys.source(f, envir = globalenv())
    use_project_lib()
    TRUE
  }, ROOT, sources)
  ## chunks of 10 x workers: the log moves and an interruption loses little, while one slow task
  ## (DeepGOF-1 takes ten times the rest) does not leave the other workers idle at each chunk's end
  chunk <- split(seq_along(tasks), ceiling(seq_along(tasks) / (10L * workers)))
  msgs <- character()
  for (ix in chunk) {
    m <- unlist(parallel::parLapplyLB(cl, tasks[ix], fun, opts = opts))
    cat(paste0(format(Sys.time()), " ", m, "\n"), sep = "", file = log_file, append = TRUE)
    msgs <- c(msgs, m)
  }
  invisible(msgs)
}

## ---------------------------------------------------------------- checks ----

## every replicate of a scenario must carry its own data: equal fingerprints across replicates mean the
## random stream was reset somewhere upstream
check_outputs <- function(scenarios, dir = GOF_DIR) {
  bad <- 0L
  for (s in scenarios) {
    f <- list.files(file.path(dir, s), pattern = "\\.csv(\\.gz)?$", full.names = TRUE)
    if (!length(f)) next
    # read as text: a hex fingerprint made only of digits would otherwise come back as a number
    fp <- vapply(f, function(x) as.character(fread_any(x, select = "fingerprint", nrows = 1,
                                                        colClasses = list(character = "fingerprint"))$fingerprint), "")
    dup <- fp[duplicated(fp) | duplicated(fp, fromLast = TRUE)]
    if (length(dup)) {
      bad <- bad + 1L
      message("DUPLICATE DATA in ", s, ": ", paste(basename(names(dup)), collapse = ", "))
    }
    err <- list.files(file.path(dir, s), pattern = "\\.error\\.txt$")
    message(sprintf("%s: %d replicate files, %d unique fingerprints, %d open errors", s, length(f),
                    length(unique(fp)), length(err)))
  }
  invisible(bad)
}

## ---------------------------------------------------------------- command line ----

parse_args <- function(a) {
  get <- function(flag, default) { i <- match(flag, a); if (is.na(i)) default else a[i + 1] }
  list(cmd = if (length(a)) a[1] else "run",
       scenario = get("--scenario", NA),
       workers = as.integer(get("--workers", 12)),
       reps = get("--reps", NA),
       null = !("--no-null" %in% a),
       lecessie = !("--no-lecessie" %in% a),
       localize = get("--localize", "plan"),
       deepgof = get("--deepgof", "plan"),
       dg_null = "--deepgof-null" %in% a,
       M_loc = as.integer(get("--M", 999)),
       B_dg = as.integer(get("--B", 199)),
       arms = { v <- get("--arms", NA); if (is.na(v)) character() else strsplit(tolower(v), ",")[[1]] },
       subset = { v <- get("--subset", NA); if (is.na(v)) NULL else v },
       preds = get("--preds", PRED_DIR),
       out = get("--out", GOF_DIR))
}

main_gof <- function(args = commandArgs(TRUE)) {
  o <- parse_args(args)
  use_project_lib()
  scen <- if (is.na(o$scenario)) list.dirs(o$preds, recursive = FALSE, full.names = FALSE) else
    strsplit(o$scenario, ",")[[1]]
  scen <- scen[!startsWith(scen, "_")]
  outd <- gof_out_dir(o$subset, o$out)
  if (o$cmd == "check") return(invisible(check_outputs(scen, outd)))
  if (o$cmd != "run") stop("unknown command: ", o$cmd)
  if (!length(scen)) stop("no scenarios under ", o$preds)
  tasks <- list()
  for (s in scen) {
    reps <- list_rep_files(s, o$preds)
    if (!is.na(o$reps)) {
      r <- eval(parse(text = o$reps))
      reps <- reps[as.integer(gsub("[^0-9]", "", names(reps))) %in% r]
    }
    for (r in names(reps)) tasks[[length(tasks) + 1L]] <- list(scenario = s, rep = r, files = reps[[r]])
  }
  message(length(tasks), " replicate tasks over ", length(scen), " scenarios")
  run_tasks(tasks, run_one_rep, o, pick_workers(o$workers), file.path(ROOT, "logs", "gof_layer.log"))
  check_outputs(scen, outd)
}

if (sys.nframe() == 0L && !nzchar(Sys.getenv("GOF_LAYER_NO_MAIN"))) main_gof()
