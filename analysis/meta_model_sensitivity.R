# Sensitivity analysis of the Q6 transfer check (7.6): does adding an n x p interaction to the
# random-design meta-model of D_ICI change the agreement with the core grid?
#
# The registered meta-model (meta_model.R) is additive in log n, log pi and p. At the core settings
# (p = 8) it could therefore borrow the small-n behaviour of the wide designs. This script refits
# the same scaled-t GAM (gam, REML, optimizer efs) with tensor-product interaction terms added,
# predicts D_ICI at the 48 core scenarios exactly as meta_model.R does, and applies the agreement
# rule of summarise.py q6 (its own function, called through Python).
#
#   Rscript analysis/meta_model_sensitivity.R --variant registered   # the registered formula, refitted
#   Rscript analysis/meta_model_sensitivity.R --variant np           # + ti(log_n, p, by = model)
#   Rscript analysis/meta_model_sensitivity.R --variant np_pi        # + ti(log_n, p) + ti(log_n, log_pi)
#   Rscript analysis/meta_model_sensitivity.R --combine              # comparison tables (after the fits)
#
# The three fits are independent and may run at the same time (one core each, about 35 min or more).
# Outputs, in tables/meta_sensitivity_te/:
#   <variant>/gam_summary.txt            summary(m) and k.check(m)
#   <variant>/fit_stats.csv              edf, deviance explained, adjusted R2, REML, log-likelihood, AIC
#   <variant>/smooth_table.csv           edf and test of every smooth term
#   <variant>/transfer_predictions.csv   same layout as tables/meta/transfer_predictions.csv
#   <variant>/q6_transfer.csv            summarise.py q6() applied to that file
#   model_comparison.csv                 fit_stats of all variants side by side, AIC difference to registered
#   agreement_counts.csv                 agreements per variant x TFM, overall and by n
#   means_by_n.csv                       mean core estimate and mean prediction per variant x TFM x n
#   registered_reproduction.csv          refitted registered predictions vs the stored ones
# Nothing outside this folder is written; meta_model.R and tables/meta/ are left as they are.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "_common.R"))
suppressPackageStartupMessages({ library(mgcv); library(arrow); library(data.table) })

out_root <- check_out(file.path(analysis_dir, "tables", "meta_sensitivity_te"))
SOURCE <- "random-design tier"
python <- file.path(dirname(rebuild_dir), ".venv", "Scripts", "python.exe")

SMOOTHS <- c(log_n = 6, log_pi = 6, p = 5, auc_target = 5, w_nl = 4, w_int = 4, w_link = 4)
form_registered <- paste(
  "D ~ model +", paste(sprintf("s(%s, by = model, k = %d)", names(SMOOTHS), SMOOTHS), collapse = " + "),
  "+ model:rho + model:frac_disc + s(design, bs = 're')")
# the interaction terms use the marginal bases of the main effects (k = 6 for log n and log pi, 5 for p)
EXTRA_TERMS <- list(
  registered = character(0),
  np = "ti(log_n, p, by = model, k = c(6, 5))",
  np_pi = c("ti(log_n, p, by = model, k = c(6, 5))", "ti(log_n, log_pi, by = model, k = c(6, 6))"))

# ---------------------------------------------------------------------------- one fit per call

fit_variant <- function(variant) {
  out_dir <- check_out(file.path(out_root, variant))
  set.seed(task_seed(paste("meta", SOURCE)), kind = "Mersenne-Twister", normal.kind = "Inversion",
           sample.kind = "Rejection")
  t0 <- Sys.time()
  elapsed <- function() as.numeric(difftime(Sys.time(), t0, units = "secs"))
  put <- function(d, name) { d <- as.data.table(d); d[, data_source := SOURCE]; fwrite(d, file.path(out_dir, name)) }

  tier <- as.data.table(read_parquet(file.path(analysis_dir, "tables", "random_tier.parquet")))
  tier[, `:=`(model = factor(model, levels = TFMS), design = factor(design),
              log_n = log(n), log_pi = log(pi))]
  form <- as.formula(paste(c(form_registered, EXTRA_TERMS[[variant]]), collapse = " + "))
  d <- copy(tier)[, D := D_ICI][is.finite(D)]
  cat(sprintf("[%5.0f s] %s: fitting the scaled-t GAM on %d rows (%d designs)\n",
              elapsed(), variant, nrow(d), uniqueN(d$design)))
  m <- gam(form, family = scat(), method = "REML", optimizer = "efs", data = d)
  i_re <- which(vapply(m$smooth, function(s) s$label == "s(design)", logical(1)))
  m$re <- m$smooth[[i_re]]$first.para:m$smooth[[i_re]]$last.para
  cat(sprintf("[%5.0f s] %s: done (edf %.1f, %s)\n", elapsed(), variant, sum(m$edf), m$family$family))

  sink(file.path(out_dir, "gam_summary.txt"))
  cat(SOURCE, "; variant", variant, "\n\n"); sm <- summary(m); print(sm); print(k.check(m)); sink()

  st <- as.data.table(sm$s.table, keep.rownames = "term")
  put(st, "smooth_table.csv")
  ti_rows <- grepl("^ti\\(", st$term)
  put(data.table(variant = variant, formula = paste(deparse(form, width.cutoff = 500L), collapse = " "),
                 n_rows = nrow(d), n_designs = uniqueN(d$design), family = m$family$family,
                 edf_total = sum(m$edf), edf_fixed_smooths = sum(st$edf[st$term != "s(design)"]),
                 edf_interactions = sum(st$edf[ti_rows]), dev_expl = sm$dev.expl, r_sq_adj = sm$r.sq,
                 reml = m$gcv.ubre, loglik = as.numeric(logLik(m)), aic = AIC(m),
                 converged = isTRUE(m$converged), seconds = round(elapsed())),
      "fit_stats.csv")

  # Q6 transfer predictions: the core-scenario covariates and SE exactly as in meta_model.R
  sc <- fread(file.path(rebuild_dir, "code", "scenarios.csv"))[family %in% c("S1", "S2", "S3", "S4")]
  sev <- list(S1 = c(0, 0, 0), S2 = c(0.5, 0, 0), S3 = c(0, 1, 0), S4 = c(0, 0, 1))
  core <- sc[, .(scenario, family, n, pi, auc_target = oracle_auc)]
  core[, `:=`(log_n = log(n), log_pi = log(pi), p = 8, rho = 0.018, frac_disc = 0.25,
              w_nl = vapply(family, function(f) sev[[f]][1], 0), w_int = vapply(family, function(f) sev[[f]][2], 0),
              w_link = vapply(family, function(f) sev[[f]][3], 0))]
  mitra_ran <- function(n, p) n^2 * p <= 8.0e7
  tp <- list()
  for (mod in TFMS) {
    nd <- copy(core)[, model := factor(mod, levels = TFMS)]
    X <- predict(m, newdata = transform(as.data.frame(nd), design = factor(levels(tier$design)[1], levels = levels(tier$design))),
                 type = "lpmatrix")
    keep <- setdiff(seq_along(coef(m)), m$re)
    V <- vcov(m, unconditional = TRUE)[keep, keep]
    Xk <- X[, keep, drop = FALSE]
    tp[[mod]] <- data.table(scenario = core$scenario, model = mod, pred = as.numeric(Xk %*% coef(m)[keep]),
                            se = sqrt(rowSums((Xk %*% V) * Xk)),
                            in_region = if (mod == "mitra") mitra_ran(core$n, 8) else TRUE)
  }
  put(rbindlist(tp), "transfer_predictions.csv")
  cat(sprintf("[%5.0f s] %s: transfer predictions written -> %s\n", elapsed(), variant, out_dir))
}

# ---------------------------------------------------------------------------- comparison tables

# summarise.py q6() on one variant's folder, so the agreement rule is the registered one
run_q6 <- function(dir) {
  py <- tempfile(fileext = ".py")
  writeLines(c(
    "import sys",
    "from pathlib import Path",
    sprintf("sys.path.insert(0, r'%s')", analysis_dir),
    "import pandas as pd",
    "import _common as C",
    "from summarise import q6",
    "paired = pd.read_parquet(C.TABLES / 'summary' / 'paired_vs_flic.parquet')",
    sprintf("out = q6(paired, Path(r'%s'))", dir),
    sprintf("out.to_csv(Path(r'%s') / 'q6_transfer.csv', index=False)", dir)), py)
  rc <- system2(python, c("-I", shQuote(py)))
  if (rc != 0) stop("q6 failed for ", dir)
  fread(file.path(dir, "q6_transfer.csv"))
}

combine <- function() {
  variants <- names(EXTRA_TERMS)[dir.exists(file.path(out_root, names(EXTRA_TERMS))) &
                                 file.exists(file.path(out_root, names(EXTRA_TERMS), "transfer_predictions.csv"))]
  if (!length(variants)) stop("no fitted variant in ", out_root)
  put <- function(d, name) { d <- as.data.table(d); d[, data_source := SOURCE]; fwrite(d, file.path(out_root, name)) }

  fs <- rbindlist(lapply(variants, function(v) fread(file.path(out_root, v, "fit_stats.csv"))), fill = TRUE)
  if ("registered" %in% fs$variant) fs[, `:=`(d_aic_vs_registered = aic - aic[variant == "registered"],
                                              d_dev_expl_vs_registered = dev_expl - dev_expl[variant == "registered"])]
  put(fs[, !"data_source"], "model_comparison.csv")

  q <- rbindlist(lapply(variants, function(v) run_q6(file.path(out_root, v))[, variant := v]))
  q[, n := as.integer(sub(".*_n", "", scenario))]
  q[, variant := factor(variant, levels = names(EXTRA_TERMS))]
  ac <- rbind(q[, .(n = "all", checked = .N, agree = sum(agrees)), by = .(variant, arm)],
              q[, .(checked = .N, agree = sum(agrees)), by = .(variant, arm, n)][, n := as.character(n)])
  setorder(ac, variant, arm, n)
  put(ac, "agreement_counts.csv")

  mb <- q[, .(checked = .N, agree = sum(agrees), mean_core = mean(estimate), mean_pred = mean(pred),
              mean_gap = mean(pred - estimate), mean_abs_gap = mean(abs(pred - estimate)),
              mean_se = mean(se), mean_mcse = mean(mcse)), by = .(variant, arm, n)]
  setorder(mb, variant, arm, n)
  put(mb, "means_by_n.csv")

  # the refitted registered predictions against the stored ones (reproducibility check)
  stored <- file.path(analysis_dir, "tables", "meta", "transfer_predictions.csv")
  if ("registered" %in% variants && file.exists(stored)) {
    a <- fread(stored); b <- fread(file.path(out_root, "registered", "transfer_predictions.csv"))
    ab <- merge(a, b, by = c("scenario", "model"), suffixes = c("_stored", "_refit"))
    put(ab[, .(rows = .N, max_abs_diff_pred = max(abs(pred_stored - pred_refit)),
               max_abs_diff_se = max(abs(se_stored - se_refit)))], "registered_reproduction.csv")
  }
  print(ac[n == "all"]); print(mb[, .(variant, arm, n, agree, mean_core, mean_pred)])
}

if (has_flag("--combine")) {
  combine()
} else {
  v <- arg_value("--variant", "np")
  if (!v %in% names(EXTRA_TERMS)) stop("--variant must be one of ", paste(names(EXTRA_TERMS), collapse = ", "))
  fit_variant(v)
}
