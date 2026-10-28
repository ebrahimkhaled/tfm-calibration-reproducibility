# Meta-model of the random-design tier (pre-registration 6.4) and the transfer predictions of Q6 (7.6).
#
#   Rscript analysis/meta_model.R                       # the real tier: tables/random_tier*.parquet -> tables/meta/
#   Rscript analysis/meta_model.R --synthetic           # the synthetic test data -> tables/meta_synthetic/
#   options: --draws 10000  --detection all|some|none
#
# Responses, per design d and TFM m, paired with FLIC on the same data:
#   D_ICI  = true ICI(m) - true ICI(FLIC)
#   D_logb = |log b*(m)| - |log b*(FLIC)|, |log b*| capped at log 10 (b* <= 0 or none = log 10)
# Model (one per response; mgcv, REML, scaled-t family):
#   D ~ model + s(log n, by = model, k = 6) + s(log pi, by = model, k = 6) + s(p, by = model, k = 5)
#         + s(AUC target, by = model, k = 5) + s(w_nl, by = model, k = 4) + s(w_int, by = model, k = 4)
#         + s(w_link, by = model, k = 4) + model:rho + model:discrete fraction + s(design, bs = "re")
# Outputs:
#   partial_effects_<resp>.csv   every smooth (and the two linear terms) per TFM, with 95 % simultaneous
#                                bands from 10,000 posterior draws (critical value of the maximum
#                                standardized deviation over the plotted grid)
#   regions_<resp>.csv           each design's own setting classified "TFM better" (band below -margin),
#                                "TFM worse" (above +margin) or "within" (margin 0.01, or log 1.1)
#   region_shares_<resp>.csv     the shares of those classes per TFM
#   map_<resp>_lognlogpi.csv     the (log n, log pi) map at the median of the other factors
#   map_<resp>_top2.csv          the map on the two factors with the largest partial effects
#   transfer_predictions.csv     D_ICI predicted at the 48 core scenarios' coordinates (7.6), with SE
#   detection_*.csv, size_*.csv  the binomial meta-models of rejection on Y and Y* (one per arm x test)
# The design random effect is left out of every prediction (population-level predictions).

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "_common.R"))
suppressPackageStartupMessages({ library(mgcv); library(arrow); library(data.table) })

synthetic <- has_flag("--synthetic")
NDRAW <- as.integer(arg_value("--draws", "10000"))
DET <- arg_value("--detection", if (synthetic) "some" else "all")
ENGINE <- arg_value("--engine", "gam")             # gam = the pre-registered fit; bam = fast, interim only
MAX_DESIGNS <- as.integer(arg_value("--max-designs", "0"))
if (!ENGINE %in% c("gam", "bam")) stop("--engine must be gam or bam")
in_dir <- if (synthetic) file.path(analysis_dir, "tables", "synthetic") else file.path(analysis_dir, "tables")
out_dir <- check_out(arg_value("--out", file.path(analysis_dir, "tables", if (synthetic) "meta_synthetic" else "meta")))
SOURCE <- if (synthetic) "SYNTHETIC TEST DATA (not a result)" else "random-design tier"
tier_file <- file.path(in_dir, "random_tier.parquet")
if (!file.exists(tier_file)) { cat("no random-tier table yet: ", tier_file, "\n"); quit(status = 0) }

MARGIN <- c(D_ICI = 0.01, D_logb = log(1.1))
CRIT_LEVEL <- 0.95
set.seed(task_seed(paste("meta", SOURCE)), kind = "Mersenne-Twister", normal.kind = "Inversion",
         sample.kind = "Rejection")
t0 <- Sys.time()
elapsed <- function() as.numeric(difftime(Sys.time(), t0, units = "secs"))
put <- function(d, name) { d <- as.data.table(d); d[, data_source := SOURCE]; fwrite(d, file.path(out_dir, name)) }

tier <- as.data.table(read_parquet(tier_file))
if (MAX_DESIGNS > 0) tier <- tier[design %in% unique(design)[seq_len(MAX_DESIGNS)]]
if (ENGINE != "gam") SOURCE <- paste0(SOURCE, "; engine bam (approximation, not the pre-registered fit)")
tier[, `:=`(model = factor(model, levels = TFMS), design = factor(design),
            log_n = log(n), log_pi = log(pi))]
COVS <- c("log_n", "log_pi", "p", "auc_target", "w_nl", "w_int", "w_link", "rho", "frac_disc")
SMOOTHS <- c(log_n = 6, log_pi = 6, p = 5, auc_target = 5, w_nl = 4, w_int = 4, w_link = 4)
LINEAR <- c("rho", "frac_disc")
med <- tier[, lapply(.SD, median), .SDcols = COVS]

form_tfm <- as.formula(paste(
  "D ~ model +", paste(sprintf("s(%s, by = model, k = %d)", names(SMOOTHS), SMOOTHS), collapse = " + "),
  "+ model:rho + model:frac_disc + s(design, bs = 're')"))

re_cols <- function(m) {
  i <- which(vapply(m$smooth, function(s) s$label == "s(design)", logical(1)))
  m$smooth[[i]]$first.para:m$smooth[[i]]$last.para
}

# population prediction (no design effect) and the simultaneous band over the rows of newdata
sim_band <- function(m, nd, cols = NULL, link_inv = identity) {
  nd <- as.data.frame(nd)
  nd$design <- factor(levels(tier$design)[1], levels = levels(tier$design))
  X <- predict(m, newdata = nd, type = "lpmatrix")
  beta <- coef(m)
  V <- vcov(m, unconditional = TRUE)
  keep <- if (is.null(cols)) setdiff(seq_along(beta), if (!is.null(m$re)) m$re else integer(0)) else cols
  X <- X[, keep, drop = FALSE]
  V <- V[keep, keep, drop = FALSE]
  fit <- as.numeric(X %*% beta[keep])
  se <- sqrt(pmax(rowSums((X %*% V) * X), 0))
  draws <- rmvn(NDRAW, rep(0, length(keep)), V)
  dev <- X %*% t(draws)
  ok <- se > 0
  crit <- if (any(ok)) as.numeric(quantile(apply(abs(dev[ok, , drop = FALSE] / se[ok]), 2, max), CRIT_LEVEL)) else NA
  data.table(fit = link_inv(fit), se = se, lo = link_inv(fit - crit * se), hi = link_inv(fit + crit * se), crit = crit)
}

classify <- function(lo, hi, margin) fifelse(hi < -margin, "TFM better", fifelse(lo > margin, "TFM worse", "within"))

mitra_ran <- function(n, p) n^2 * p <= 8.0e7

fit_response <- function(resp) {
  d <- copy(tier)[, D := get(resp)][is.finite(D)]
  cat(sprintf("[%5.0f s] %s: fitting the scaled-t GAM on %d rows (%d designs)\n", elapsed(), resp, nrow(d), uniqueN(d$design)))
  m <- if (ENGINE == "gam") {
    # the pre-registered fit: gam, REML. The extended Fellner-Schall optimizer maximises the same
    # REML criterion as the default Newton optimizer, and is far cheaper with 1,000 design effects
    gam(form_tfm, family = scat(), method = "REML", optimizer = "efs", data = d)
  } else {
    # fast approximation for interim and test runs only (discretised covariates, fREML)
    bam(form_tfm, family = scat(), method = "fREML", discrete = TRUE, nthreads = 1, data = d)
  }
  m$re <- re_cols(m)
  cat(sprintf("[%5.0f s] %s: done (edf %.1f, %s)\n", elapsed(), resp, sum(m$edf), m$family$family))
  sink(file.path(out_dir, sprintf("gam_summary_%s.txt", resp))); cat(SOURCE, "\n\n"); print(summary(m)); print(k.check(m)); sink()

  # partial effects with simultaneous bands
  pe <- list()
  for (mod in levels(d$model)) {
    dm <- d[model == mod]
    for (v in c(names(SMOOTHS), LINEAR)) {
      grid <- seq(min(dm[[v]]), max(dm[[v]]), length.out = 100)
      nd <- med[rep(1, 100)]; nd[[v]] <- grid; nd$model <- factor(mod, levels = levels(d$model))
      lab <- if (v %in% names(SMOOTHS)) sprintf("s(%s):model%s", v, mod) else sprintf("model%s:%s", mod, v)
      cols <- if (v %in% names(SMOOTHS)) {
        sm <- m$smooth[[which(vapply(m$smooth, function(s) s$label == lab, logical(1)))]]
        sm$first.para:sm$last.para
      } else which(names(coef(m)) == lab)
      b <- sim_band(m, nd, cols)
      if (v %in% LINEAR) {           # a linear term is drawn centred at the median of its factor
        c0 <- coef(m)[cols] * med[[v]]
        b[, `:=`(fit = fit - c0, lo = lo - c0, hi = hi - c0)]
      }
      pe[[paste(mod, v)]] <- data.table(model = mod, term = v, x = grid, b)
    }
  }
  pe <- rbindlist(pe)
  put(pe, sprintf("partial_effects_%s.csv", resp))

  # regions: each design's own setting
  reg <- list()
  for (mod in levels(d$model)) {
    dm <- unique(tier[model == mod, c("design", COVS), with = FALSE])
    nd <- copy(dm)[, model := factor(mod, levels = levels(d$model))]
    b <- sim_band(m, nd)
    reg[[mod]] <- data.table(model = mod, dm, b, class = classify(b$lo, b$hi, MARGIN[[resp]]))
  }
  reg <- rbindlist(reg)
  put(reg, sprintf("regions_%s.csv", resp))
  sh <- reg[, .N, by = .(model, class)][, share := N / sum(N), by = model]
  put(sh, sprintf("region_shares_%s.csv", resp))

  # maps: (log n, log pi), and the two factors with the largest partial effects
  rng <- pe[, .(range = max(fit) - min(fit)), by = .(model, term)]
  maps1 <- list(); maps2 <- list()
  for (mod in levels(d$model)) {
    gn <- seq(log(150), log(4000), length.out = 40); gp <- seq(log(0.02), log(0.40), length.out = 40)
    nd <- med[rep(1, 1600)]; nd$log_n <- rep(gn, 40); nd$log_pi <- rep(gp, each = 40)
    nd$model <- factor(mod, levels = levels(d$model))
    b <- sim_band(m, nd)
    maps1[[mod]] <- data.table(model = mod, log_n = nd$log_n, log_pi = nd$log_pi, b,
                               class = classify(b$lo, b$hi, MARGIN[[resp]]),
                               mitra_not_run = if (mod == "mitra") !mitra_ran(exp(nd$log_n), med$p) else FALSE)
    top <- rng[model == mod][order(-range)][1:2, term]
    g1 <- seq(min(tier[[top[1]]]), max(tier[[top[1]]]), length.out = 40)
    g2 <- seq(min(tier[[top[2]]]), max(tier[[top[2]]]), length.out = 40)
    nd <- med[rep(1, 1600)]; nd[[top[1]]] <- rep(g1, 40); nd[[top[2]]] <- rep(g2, each = 40)
    nd$model <- factor(mod, levels = levels(d$model))
    b <- sim_band(m, nd)
    maps2[[mod]] <- data.table(model = mod, x_name = top[1], y_name = top[2], x = nd[[top[1]]], y = nd[[top[2]]], b,
                               class = classify(b$lo, b$hi, MARGIN[[resp]]))
  }
  put(rbindlist(maps1), sprintf("map_%s_lognlogpi.csv", resp))
  put(rbindlist(maps2), sprintf("map_%s_top2.csv", resp))
  m
}

m_ici <- fit_response("D_ICI")
m_logb <- fit_response("D_logb")

# Q6: predictions at the 48 core scenarios' own coordinates (7.6)
sc <- fread(file.path(rebuild_dir, "code", "scenarios.csv"))[family %in% c("S1", "S2", "S3", "S4")]
sev <- list(S1 = c(0, 0, 0), S2 = c(0.5, 0, 0), S3 = c(0, 1, 0), S4 = c(0, 0, 1))
core <- sc[, .(scenario, family, n, pi, auc_target = oracle_auc)]
core[, `:=`(log_n = log(n), log_pi = log(pi), p = 8, rho = 0.018, frac_disc = 0.25,
            w_nl = vapply(family, function(f) sev[[f]][1], 0), w_int = vapply(family, function(f) sev[[f]][2], 0),
            w_link = vapply(family, function(f) sev[[f]][3], 0))]
tp <- list()
for (mod in TFMS) {
  nd <- copy(core)[, model := factor(mod, levels = TFMS)]
  X <- predict(m_ici, newdata = transform(as.data.frame(nd), design = factor(levels(tier$design)[1], levels = levels(tier$design))),
               type = "lpmatrix")
  keep <- setdiff(seq_along(coef(m_ici)), m_ici$re)
  V <- vcov(m_ici, unconditional = TRUE)[keep, keep]
  Xk <- X[, keep, drop = FALSE]
  tp[[mod]] <- data.table(scenario = core$scenario, model = mod, pred = as.numeric(Xk %*% coef(m_ici)[keep]),
                          se = sqrt(rowSums((Xk %*% V) * Xk)),
                          in_region = if (mod == "mitra") mitra_ran(core$n, 8) else TRUE)
}
put(rbindlist(tp), "transfer_predictions.csv")
cat(sprintf("[%5.0f s] transfer predictions written\n", elapsed()))

# detection and size meta-models (one binomial GAM per arm x test; no design random effect)
gof_file <- file.path(in_dir, "random_tier_gof.parquet")
if (DET != "none" && file.exists(gof_file)) {
  gd <- as.data.table(read_parquet(gof_file))[, `:=`(log_n = log(n), log_pi = log(pi))]
  form_bin <- as.formula(paste("reject ~", paste(sprintf("s(%s, k = %d)", names(SMOOTHS), SMOOTHS), collapse = " + "),
                               "+ rho + frac_disc"))
  combos <- unique(gd[, .(arm, test)])
  if (DET == "some") combos <- combos[seq_len(min(.N, 4))]
  det <- list(); siz <- list(); szreg <- list()
  for (i in seq_len(nrow(combos))) {
    a <- combos$arm[i]; tt <- combos$test[i]
    for (yt in c("real", "ystar")) {
      dd <- gd[arm == a & test == tt & (ytype == yt | (yt == "ystar" & a == "oracle" & ytype == "real"))]
      if (yt == "real" && a == "oracle") next          # the oracle's real Y is a null: it goes with the size rows
      if (nrow(dd) < 100 || length(unique(dd$reject)) < 2) next
      mb <- tryCatch(gam(form_bin, family = binomial, method = "REML", data = dd), error = function(e) NULL)
      if (is.null(mb)) next
      mb$re <- integer(0)
      dm <- unique(dd[, c("design", COVS), with = FALSE])
      b <- sim_band(mb, dm, link_inv = plogis)
      if (yt == "real") {
        gn <- seq(log(150), log(4000), length.out = 30); gp <- seq(log(0.02), log(0.40), length.out = 30)
        nd <- med[rep(1, 900)]; nd$log_n <- rep(gn, 30); nd$log_pi <- rep(gp, each = 30)
        bm <- sim_band(mb, nd, link_inv = plogis)
        det[[paste(a, tt)]] <- data.table(arm = a, test = tt, log_n = nd$log_n, log_pi = nd$log_pi, bm)
      } else {
        out <- fifelse(b$lo > 0.06, "above [0.04, 0.06]", fifelse(b$hi < 0.04, "below [0.04, 0.06]", "overlaps [0.04, 0.06]"))
        siz[[paste(a, tt)]] <- data.table(arm = a, test = tt, design = dm$design, dm[, COVS, with = FALSE], b, size_region = out)
        szreg[[paste(a, tt)]] <- data.table(arm = a, test = tt, n_designs = nrow(dm),
                                            share_above = mean(out == "above [0.04, 0.06]"),
                                            share_below = mean(out == "below [0.04, 0.06]"))
      }
    }
    cat(sprintf("[%5.0f s] detection/size: %s, %s\n", elapsed(), a, tt))
  }
  if (length(det)) put(rbindlist(det), "detection_map_lognlogpi.csv")
  if (length(siz)) put(rbindlist(siz), "size_regions_by_design.csv")
  if (length(szreg)) put(rbindlist(szreg), "size_region_shares.csv")
}
cat(sprintf("meta-model done in %.0f s -> %s (%s)\n", elapsed(), out_dir, SOURCE))
