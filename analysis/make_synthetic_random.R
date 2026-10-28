# A SYNTHETIC random-design dataset for testing meta_model.R before the random tier has run.
#
#   Rscript analysis/make_synthetic_random.R
#
# It takes only the design settings of code/random_designs.csv (n, pi, p, rho, AUC target,
# severities, discrete fraction: no result of any kind) and invents the responses from a known
# formula, so that the meta-model can be checked against a truth. Nothing here is a study result.
# Output: analysis/tables/synthetic/random_tier.parquet and random_tier_gof.parquet, with exactly
# the columns that summarise.py writes for the real tier, plus the true values used.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "_common.R"))
suppressPackageStartupMessages({ library(arrow); library(data.table) })

out_dir <- check_out(file.path(analysis_dir, "tables", "synthetic"))
rd <- fread(file.path(rebuild_dir, "code", "random_designs.csv"))
set.seed(task_seed("synthetic random tier"), kind = "Mersenne-Twister", normal.kind = "Inversion",
         sample.kind = "Rejection")

D <- nrow(rd)
design_re <- rnorm(D, 0, 0.004)                         # what the three TFMs share on one data set
mitra_fits <- rd$n^2 * rd$p <= 8.0e7                     # the device rule of 3.9

# known truth of the synthetic D_ICI (TFM minus FLIC): FLIC is worse at small n and low pi, the
# TFMs worse under strong non-linearity; one model shifted, one with a link effect
truth_ici <- function(model, x) {
  base <- -0.018 * pmax(0, log(1000) - log(x$n)) / log(1000 / 150) +
          -0.006 * (log(x$pi) - log(0.1)) / log(4) + 0.012 * x$w_nl - 0.004 * x$rho
  shift <- c(tabpfn = -0.004, tabicl = 0.002, mitra = 0.000)[model]
  link <- c(tabpfn = 0, tabicl = 0.010, mitra = 0.004)[model] * x$w_link
  base + shift + link
}
truth_logb <- function(model, x) {
  -0.20 * pmax(0, log(800) - log(x$n)) / log(800 / 150) + c(tabpfn = 0.02, tabicl = 0.06, mitra = 0)[model] +
    0.05 * x$w_int
}

rows <- list()
for (m in TFMS) {
  keep <- if (m == "mitra") mitra_fits else rep(TRUE, D)
  x <- rd[keep]
  ti <- truth_ici(m, x)
  tl <- truth_logb(m, x)
  # a scaled-t error with a few gross values, as a separated FLIC fit would give
  e_ici <- 0.010 * rt(nrow(x), df = 4)
  e_lb <- 0.12 * rt(nrow(x), df = 3)
  rows[[m]] <- data.table(design = x$scenario, model = m, D_ICI = ti + design_re[keep] + e_ici,
                          D_logb = pmin(pmax(tl + e_lb, -log(10)), log(10)),
                          n = x$n, pi = x$pi, p = x$p, auc_target = x$auc_target, w_nl = x$w_nl,
                          w_int = x$w_int, w_link = x$w_link, rho = x$rho, frac_disc = x$frac_disc,
                          true_D_ICI = ti, true_D_logb = tl)
}
tier <- rbindlist(rows)
write_parquet(tier, file.path(out_dir, "random_tier.parquet"))

# synthetic rejections for the detection and size meta-models: two tests, three arms plus the
# oracle; Y* rows at a true size of 0.05 except one liberal corner (low pi, small n) for one test
arms <- c("oracle", "flic", "tabpfn")
tests <- c("EDGE (G=10)", "Hosmer-Lemeshow (external)")
g <- list()
for (a in arms) for (tt in tests) {
  x <- rd
  lp_real <- if (a == "oracle") rep(qlogis(0.05), D) else
    -1.5 + 0.8 * (log(x$n) - log(700)) + 1.2 * x$w_nl + 0.8 * x$w_int + (tt == "EDGE (G=10)") * 0.3
  g[[paste(a, tt, "real")]] <- data.table(design = x$scenario, arm = a, ytype = "real", test = tt,
                                          reject = rbinom(D, 1, plogis(lp_real)))
  lib <- (tt == "Hosmer-Lemeshow (external)") * 1.2 * (x$pi < 0.03) * (x$n < 400)
  g[[paste(a, tt, "ystar")]] <- data.table(design = x$scenario, arm = a, ytype = "ystar", test = tt,
                                           reject = rbinom(D, 1, plogis(qlogis(0.05) + lib)))
}
gof <- rbindlist(g)
cov <- rd[, .(design = scenario, n, pi, p, auc_target, w_nl, w_int, w_link, rho, frac_disc)]
gof <- merge(gof, cov, by = "design")
write_parquet(gof, file.path(out_dir, "random_tier_gof.parquet"))
cat(sprintf("synthetic tier: %d TFM rows (%d designs; Mitra-v2 in %d), %d rejection rows -> %s\n",
            nrow(tier), D, sum(mitra_fits), nrow(gof), out_dir))
