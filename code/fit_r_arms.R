## fit_r_arms.R -- the statistical arms of one replicate: LR-MLE, FLIC, ridge-CV and GAM.
##
## Called by models.py (file hand-off, no rpy2):
##   Rscript fit_r_arms.R <workdir>
## <workdir> holds job.json (arms, seed, column types), dev.csv (y + covariates) and pred.csv
## (covariates of the rows to predict). Writes preds.csv (one column per arm that ran) and
## info.json (status, error, seconds and fit flags per arm, package versions, a check-sum of the
## development data so that Python can assert it read back the same replicate).

local({
  rlib <- Sys.getenv("TFM_RLIB", "")
  if (!nzchar(rlib)) {
    a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    here <- dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/"))
    rlib <- file.path(dirname(here), "rlib")
  }
  if (dir.exists(rlib)) .libPaths(c(rlib, .Library))
})
suppressPackageStartupMessages(library(jsonlite))

wd <- commandArgs(TRUE)[1]
job <- fromJSON(file.path(wd, "job.json"))
dev <- read.csv(file.path(wd, "dev.csv"))
new <- read.csv(file.path(wd, "pred.csv"))

cont <- unlist(job$cont); bin <- unlist(job$bin); cat_cols <- unlist(job$cat)
for (v in cat_cols) {
  lev <- as.character(seq_len(job$cat_levels[[v]]) - 1L)
  dev[[v]] <- factor(as.character(dev[[v]]), levels = lev)
  new[[v]] <- factor(as.character(new[[v]]), levels = lev)
}
rhs_main <- paste(c(cont, bin, cat_cols), collapse = " + ")
f_main <- as.formula(paste("y ~", rhs_main))

## run one arm, keeping its warnings and timing; an error is recorded, never hidden
run_arm <- function(name, fun) {
  warn <- character(0)
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(
    withCallingHandlers(fun(), warning = function(w) {
      warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning")
    }),
    error = function(e) list(error = conditionMessage(e)))
  secs <- proc.time()[["elapsed"]] - t0
  info <- list(seconds = secs, warnings = unique(warn))
  if (!is.null(res$error)) {
    info$status <- "failed"; info$error <- res$error
    return(list(p = NULL, info = info))
  }
  p <- as.numeric(res$p)
  if (length(p) != nrow(new) || any(!is.finite(p))) {
    info$status <- "failed"; info$error <- "non-finite or missing predictions"
    return(list(p = NULL, info = info))
  }
  info$status <- "ok"
  info <- c(info, res$flags)
  list(p = p, info = info)
}

## ridge lambda path (pre-registration 5.2)
RIDGE_NLAMBDA <- 200L
RIDGE_MIN_RATIO <- 1e-6

## folds stratified on y, from the arm's own seed
strat_folds <- function(y, k, seed) {
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  id <- integer(length(y))
  for (cl in unique(y)) {
    w <- which(y == cl)
    id[w] <- sample(rep(seq_len(k), length.out = length(w)))
  }
  id
}

fitters <- list(
  lr = function() {
    fit <- glm(f_main, family = binomial, data = dev)
    b <- coef(fit)
    fp <- fitted(fit)
    list(p = predict(fit, newdata = new, type = "response"),
         flags = list(converged = fit$converged, iterations = fit$iter,
                      separation_flag = !fit$converged || any(fp < 1e-8 | fp > 1 - 1e-8) ||
                        any(abs(b) > 15, na.rm = TRUE),
                      n_aliased = sum(is.na(b)), max_abs_coef = max(abs(b), na.rm = TRUE)))
  },
  flic = function() {
    fit <- logistf::logistf(f_main, data = dev, flic = TRUE, pl = FALSE)
    list(p = predict(fit, newdata = new, type = "response"),
         flags = list(iterations = unname(fit$iter[1]),
                      converged = unname(fit$iter[1]) < fit$control$maxit,
                      conv = unname(fit$conv), max_abs_coef = max(abs(coef(fit)))))
  },
  ridge = function() {
    x <- model.matrix(f_main, dev)[, -1, drop = FALSE]
    xn <- model.matrix(as.formula(paste("~", rhs_main)), new)[, -1, drop = FALSE]
    fold <- strat_folds(dev$y, 10L, job$seeds$ridge)
    ## glmnet's default path ends at lambda.min.ratio x lambda_max and, worse, stops early once
    ## the deviance stops changing (fdev), so at n >= 1000 cross-validation kept choosing the
    ## smallest lambda on offer (pilot: 19 of 20 fits at S3, pi = 30 %). The path used here starts
    ## at glmnet's own lambda_max, runs 200 log-spaced steps down to 1e-6 of it, and the early stop
    ## is switched off for this fit only. A minimum still at either end is flagged.
    lmax <- max(glmnet::glmnet(x, dev$y, family = "binomial", alpha = 0, nlambda = 5)$lambda)
    lam <- exp(seq(log(lmax), log(lmax * RIDGE_MIN_RATIO), length.out = RIDGE_NLAMBDA))
    glmnet::glmnet.control(fdev = 0)
    on.exit(glmnet::glmnet.control(factory = TRUE), add = TRUE)
    cv <- glmnet::cv.glmnet(x, dev$y, family = "binomial", alpha = 0, foldid = fold,
                            lambda = lam, type.measure = "deviance")
    k <- match(cv$lambda.min, cv$lambda)
    list(p = predict(cv, newx = xn, s = "lambda.min", type = "response"),
         flags = list(lambda_min = cv$lambda.min, lambda_path_max = max(cv$lambda),
                      lambda_path_min = min(cv$lambda), lambda_path_length = length(cv$lambda),
                      lambda_min_index = k,
                      lambda_at_edge = k == 1L || k == length(cv$lambda),
                      lambda_at_lower_edge = k == length(cv$lambda)))
  },
  gam = function() {
    sm <- sprintf("s(%s, k = %d)", cont, as.integer(job$gam_k))
    f <- as.formula(paste("y ~", paste(c(sm, bin, cat_cols), collapse = " + ")))
    fit <- mgcv::gam(f, family = binomial, data = dev, method = "REML")
    list(p = predict(fit, newdata = new, type = "response"),
         flags = list(converged = isTRUE(fit$converged), edf = sum(fit$edf),
                      outer_converged = isTRUE(fit$outer.info$conv == "full convergence")))
  }
)

arms <- unlist(job$arms)
## load the packages first, so that an arm's seconds are its fit time only
pkg_of <- c(flic = "logistf", ridge = "glmnet", gam = "mgcv")
t_load <- system.time(for (p in unique(pkg_of[intersect(arms, names(pkg_of))]))
  suppressPackageStartupMessages(loadNamespace(p)))[["elapsed"]]
out <- list(); info <- list()
for (a in arms) {
  if (is.null(fitters[[a]])) { info[[a]] <- list(status = "failed", error = "unknown R arm"); next }
  r <- run_arm(a, fitters[[a]])
  info[[a]] <- r$info
  if (!is.null(r$p)) out[[a]] <- r$p
}

if (length(out)) {
  txt <- as.data.frame(lapply(out, function(p) sprintf("%.17g", p)))
  write.csv(txt, file.path(wd, "preds.csv"), row.names = FALSE, quote = FALSE)
}
xs <- as.matrix(dev[, setdiff(names(dev), c("y", cat_cols)), drop = FALSE])
meta <- list(
  arms = info,
  check = list(n_dev = nrow(dev), events_dev = sum(dev$y), sum_x = sum(xs), n_pred = nrow(new)),
  package_load_seconds = t_load,
  versions = list(R = R.version.string,
                  logistf = as.character(packageVersion("logistf")),
                  mgcv = as.character(packageVersion("mgcv")),
                  glmnet = as.character(packageVersion("glmnet")),
                  lib = .libPaths()[1])
)
writeLines(toJSON(meta, auto_unbox = TRUE, digits = NA, null = "null", na = "null"),
           file.path(wd, "info.json"))
