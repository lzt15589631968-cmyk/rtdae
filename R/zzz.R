# R/zzz.R
.pkgenv <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Bind dedicated virtual environment if present
  env_name <- "r-rtdae-env"

  if (reticulate::virtualenv_exists(env_name)) {
    reticulate::use_virtualenv(env_name, required = FALSE)
  }

  # Defer Python module imports to optimize package startup latency
  .pkgenv$torch  <- reticulate::import("torch", delay_load = TRUE)
  .pkgenv$gudhi  <- reticulate::import("gudhi", delay_load = TRUE)
  .pkgenv$scipy  <- reticulate::import("scipy", delay_load = TRUE)
  .pkgenv$numpy  <- reticulate::import("numpy", delay_load = TRUE)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("rtdae: Topological Autoencoder for R initialized successfully.")
}
