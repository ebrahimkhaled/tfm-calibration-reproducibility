## Unit tests of the GOF layer changes made before the freeze (2026-10-07).
##   Rscript R/tests/test_gof_layer.R
## Checks: ebrahim.gof 2.10.0 from rlib with the "Osius-Rojek (external)" row, the battery recording that
## row and Stukel, gzip output read back by the layer, the --arms filter, and DeepGOF-1's axes reading.
## Added 2026-10-07 (b): the pin to commit 19480aa by code hash, the refusal of any other build, the
## score form of the Stukel row (df 2, or 1 with no prediction above 0.5) and the DeepGOF-1 reading rule.

Sys.setenv(GOF_LAYER_NO_MAIN = "1")
here <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/")))
})
Sys.setenv(TFM_REBUILD_ROOT = dirname(here))
sys.source(file.path(here, "gof_layer.R"), envir = globalenv())
use_project_lib()

passed <- 0L; failed <- 0L
check <- function(name, expr) {
  ok <- tryCatch(isTRUE(expr), error = function(e) { message("  error: ", conditionMessage(e)); FALSE })
  if (ok) passed <<- passed + 1L else failed <<- failed + 1L
  cat(if (ok) "ok   " else "FAIL ", name, "\n")
}

check("ebrahim.gof 2.10.0 is loaded from rebuild/rlib",
      as.character(packageVersion("ebrahim.gof")) == "2.10.0" &&
        normalizePath(dirname(find.package("ebrahim.gof")), winslash = "/") == normalizePath(RLIB, winslash = "/"))

set.seed(7)
n <- 1000
X <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rbinom(n, 1, 0.4))
p <- plogis(-2.2 + 0.9 * X$x1 - 0.5 * X$x3)
y <- rbinom(n, 1, p)

b <- gof_battery(y, p, X, seed_base = "unit|test", do_lecessie = FALSE, do_localize = FALSE,
                 do_deepgof = FALSE)
check("run.all.external() gives an Osius-Rojek (external) row with a p-value",
      "Osius-Rojek (external)" %in% b$test && is.finite(b$p_value[b$test == "Osius-Rojek (external)"]))
check("the battery keeps the Stukel (offset) row", "Stukel (offset)" %in% b$test)

## the pin: commit 19480aa, checked through the code of the installed build, not the version number
check("the installed build is commit 19480aa (code hash) and rows carry 2.10.0@19480aa",
      identical(gof_code_hash(), GOF_CODE_HASH) && GOF_REF == "2.10.0@19480aa")
check("use_project_lib() refuses a build with another code hash", {
  keep <- GOF_CODE_HASH; old <- .gof_checked$hash
  GOF_CODE_HASH <<- "another-commit"; .gof_checked$hash <- NULL
  refused <- tryCatch({ use_project_lib(); FALSE }, error = function(e) grepl("not the build of commit", conditionMessage(e)))
  GOF_CODE_HASH <<- keep; .gof_checked$hash <- old
  refused && isTRUE(use_project_lib())
})

## Stukel (offset) in 19480aa: the score test at the null, df = 2 (1 when no prediction reaches 0.5)
st <- b[b$test == "Stukel (offset)", ]
score_ref <- function(y, p) {
  lp <- qlogis(p); v <- p * (1 - p)
  Z <- cbind(0.5 * lp^2 * (lp >= 0), -0.5 * lp^2 * (lp < 0))
  Z <- Z[, colSums(Z != 0) > 0, drop = FALSE]
  U <- crossprod(Z, y - p); I <- crossprod(Z * v, Z)
  c(S = drop(t(U) %*% solve(I, U)), df = ncol(Z))
}
ref <- score_ref(y, p)
check("Stukel row is the score test (Note) and equals an independent score statistic",
      grepl("score test", st$note) && st$df == ref[["df"]] &&
        abs(st$statistic - ref[["S"]]) < 1e-6 * max(1, ref[["S"]]))
p_low <- plogis(-4 + 0.5 * X$x1); y_low <- rbinom(n, 1, p_low)       # no prediction reaches 0.5
bl <- gof_battery(y_low, p_low, X, seed_base = "unit|low", do_lecessie = FALSE, do_localize = FALSE,
                  do_deepgof = FALSE)
sl <- bl[bl$test == "Stukel (offset)", ]
check("Stukel has df = 1 when no prediction reaches 0.5",
      max(p_low) < 0.5 && sl$df == 1 && grepl("one tail has no records", sl$note) && is.finite(sl$p_value))

## DeepGOF-1 reading: axes when the design has more than 12 predictors (T2, random designs), else combined
check("DeepGOF-1 reading rule: T2 and p > 12 use axes; core and T3 combined",
      dg_reading_for("T2_pi10_n500", 30) == "axes" && dg_reading_for("R0007", 13) == "axes" &&
        dg_reading_for("R0008", 12) == "combined" && dg_reading_for("T3_pi10_n500", 8) == "combined" &&
        dg_reading_for("S1_pi03_n200", 8) == "combined")

bd <- gof_battery(y, p, X, seed_base = "unit|test", do_lecessie = FALSE, do_localize = FALSE,
                  do_deepgof = TRUE, B_dg = 19L, dg_reading = "axes")
dg <- bd[bd$family == "DeepGOF", ]
check("DeepGOF-1 axes reading: labelled row with a p-value, empty all-pairs row",
      dg$test[1] == "DeepGOF-1 external (axes reading)" && is.finite(dg$p_value[1]) &&
        grepl("reading axes", dg$note[1]) && is.na(dg$p_value[dg$test == "DeepGOF-1 external (allpairs)"]))

tmpd <- tempfile("gofgz"); dir.create(tmpd)
f <- file.path(tmpd, "S1", "3.csv.gz")
d <- cbind(scenario = "S1", rep = "3", fingerprint = "abc", pred_hash = "h", b)
atomic_write(d, f)
check("atomic_write() writes gzip when the name ends in .gz",
      identical(readBin(f, "raw", 2), as.raw(c(0x1f, 0x8b))))
back <- read_gof_csv(f)
check("read_gof_csv() reads the .csv.gz back intact",
      nrow(back) == nrow(d) && identical(back$test, d$test) &&
        isTRUE(all.equal(back$p_value, d$p_value, tolerance = 1e-12)))
plain <- file.path(tmpd, "S1", "4.csv")
atomic_write(d, plain)
check("an older uncompressed <rep>.csv is still found and read",
      gof_file_of(tmpd, "S1", "4") == plain && nrow(read_gof_csv(plain)) == nrow(d) &&
        gof_file_of(tmpd, "S1", "3") == f)
check("check_outputs() counts .csv.gz files",
      { msg <- capture.output(check_outputs("S1", tmpd), type = "message"); any(grepl("2 replicate files", msg)) })

o <- parse_args(c("run", "--arms", "oracle,FLIC", "--B", "99"))
check("--arms is parsed (lower case) and --B is kept", identical(o$arms, c("oracle", "flic")) && o$B_dg == 99L)
check("no --arms means every arm", length(parse_args("run")$arms) == 0L)
check("temperature arms get localization but not DeepGOF-1",
      arm_wants("tabpfn_t1", "plan", "localize") && !arm_wants("tabpfn_t1", "plan", "deepgof") &&
        !arm_wants("tabicl_t1", "plan", "deepgof") && arm_wants("tabpfn", "plan", "deepgof"))

unlink(tmpd, recursive = TRUE)
cat(sprintf("%d/%d passed\n", passed, passed + failed))
quit(status = if (failed) 1L else 0L)
