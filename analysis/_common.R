# Shared settings of the R scripts in analysis/: the project library, one BLAS/OpenMP thread and
# below-normal priority (the main run must not be slowed), and small helpers.

for (v in c("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) do.call(Sys.setenv, setNames(list("1"), v))

analysis_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) normalizePath(dirname(f[1]), winslash = "/") else normalizePath("analysis", winslash = "/")
})
rebuild_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")
.libPaths(c(file.path(rebuild_dir, "rlib"), .Library))

lower_priority <- function() {
  if (.Platform$OS.type == "windows") {
    cmd <- sprintf("(Get-Process -Id %d).PriorityClass = 'BelowNormal'", Sys.getpid())
    try(system2("powershell", c("-NoProfile", "-Command", shQuote(cmd)), stdout = FALSE, stderr = FALSE),
        silent = TRUE)
  } else {
    try(tools::psnice(Sys.getpid(), 10), silent = TRUE)
  }
}
lower_priority()

arg_value <- function(name, default = NULL) {
  a <- commandArgs(trailingOnly = TRUE)
  i <- match(name, a)
  if (is.na(i) || i == length(a)) return(default)
  a[i + 1]
}
has_flag <- function(name) name %in% commandArgs(trailingOnly = TRUE)

# writing is allowed only inside analysis/
check_out <- function(path) {
  p <- normalizePath(path, winslash = "/", mustWork = FALSE)
  if (!startsWith(p, analysis_dir)) stop("refusing to write outside analysis/: ", p)
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
  p
}

BATTERY <- c("EDGE (G=10)", "EDGE (G=auto)", "Cox recalibration", "Calibration-in-the-large",
             "Spiegelhalter-z", "Osius-Rojek (external)", "GiViTI", "Hosmer-Lemeshow (external)",
             "Stukel (offset)", "le Cessie (external)")
EXTRA <- c("DeepGOF-1 external (combined)", "DeepGOF-1 external (axes reading)", "Localize (any part named)")
TFMS <- c("tabpfn", "tabicl", "mitra")

# a seed per task string, so results do not depend on which worker runs what
task_seed <- function(s) {
  h <- 0
  for (ch in utf8ToInt(s)) h <- (h * 31 + ch) %% 2147483647
  as.integer(h)
}
