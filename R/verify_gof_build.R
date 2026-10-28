## verify_gof_build.R -- is the ebrahim.gof in rebuild/rlib the build of the pinned commit?
##
##   Rscript R/verify_gof_build.R [commit]          (default: the commit pinned in gof_layer.R)
##
## Reads the package's R sources at that commit from the release-2.10.0 worktree (git archive into a
## temporary folder; the worktree itself is never changed), sources them into an empty environment
## and compares every function of the installed namespace with them: formals and body, deparsed.
## Then it prints the code hash of the installed build, the value that gof_layer.R keeps in
## GOF_CODE_HASH and checks on every start.

Sys.setenv(GOF_LAYER_NO_MAIN = "1")
here <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", a[1]), winslash = "/"))
})
source(file.path(here, "gof_layer.R"))

args <- commandArgs(TRUE)
commit <- if (length(args)) args[1] else GOF_COMMIT
## EBGOF_WORKTREE: a git clone of the ebrahim.gof repository that contains the pinned commit
wt <- Sys.getenv("EBGOF_WORKTREE", "")
if (!nzchar(wt)) stop("set EBGOF_WORKTREE to a git clone of ebrahim.gof that contains commit ", commit)

## sources at the commit, through git archive (read only)
tmp <- tempfile("ebgof_src_"); dir.create(tmp)
tarf <- file.path(tmp, "src.tar")
st <- system2("git", c("-C", shQuote(wt), "archive", "--format=tar", "-o", shQuote(tarf), commit, "R"))
if (st != 0) stop("git archive of ", commit, " failed")
utils::untar(tarf, exdir = tmp)
full <- system2("git", c("-C", shQuote(wt), "rev-parse", commit), stdout = TRUE)

src_env <- new.env()
for (f in sort(list.files(file.path(tmp, "R"), pattern = "\\.[Rr]$", full.names = TRUE)))
  sys.source(f, envir = src_env, keep.source = FALSE)

.libPaths(unique(c(RLIB, .libPaths())))
ns <- asNamespace("ebrahim.gof")
cat("installed from:", dirname(find.package("ebrahim.gof")), "\n")
cat("version:", as.character(utils::packageVersion("ebrahim.gof")), "\n")
cat("commit:", full, "\n")

fns <- sort(Filter(function(k) is.function(get(k, envir = ns)), ls(ns, all.names = TRUE)))
same <- function(a, b) identical(deparse(formals(a)), deparse(formals(b))) &&
  identical(deparse(body(a)), deparse(body(b)))
missing_in_src <- setdiff(fns, ls(src_env, all.names = TRUE))
differ <- Filter(function(k) exists(k, envir = src_env, inherits = FALSE) &&
                   !same(get(k, envir = ns), get(k, envir = src_env)), fns)
cat(sprintf("%d functions in the installed namespace; %d not in the sources; %d differ\n",
            length(fns), length(missing_in_src), length(differ)))
if (length(missing_in_src)) cat("  not in the sources:", paste(missing_in_src, collapse = ", "), "\n")
if (length(differ)) cat("  differ:", paste(differ, collapse = ", "), "\n")

cat("installed code hash:", gof_code_hash(), "\n")
cat("pinned code hash:   ", GOF_CODE_HASH, "\n")
ok <- !length(differ) && !length(missing_in_src) && identical(gof_code_hash(), GOF_CODE_HASH)
cat(if (ok) "VERIFIED: the installed build is commit" else "MISMATCH: the installed build is not commit",
    substr(full, 1, 7), "\n")
unlink(tmp, recursive = TRUE)
quit(status = if (ok) 0 else 1)
