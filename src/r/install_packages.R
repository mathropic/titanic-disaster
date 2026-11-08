install.packages("remotes")
remotes::install_version("readr", version = "2.2.0", upgrade = "never")
remotes::install_version("dplyr", version = "1.2.1", upgrade = "never")
remotes::install_version("lightgbm", version = "4.7.0", upgrade = "never")

# Fail the Docker build if anything is missing, and show what was installed
for (pkg in c("readr", "dplyr", "lightgbm")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Failed to install ", pkg)
  cat(pkg, as.character(packageVersion(pkg)), "installed\n")
}