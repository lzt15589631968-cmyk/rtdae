# 1. Initialize a global variable at the package level to store the Python module object (initially set to NULL)
scipy <- NULL
# 2. Encapsulate the environment-binding logic entirely within the unified .onLoad hook
.onLoad <- function(libname, pkgname) {
  # Define the designated name for the isolated virtual environment internally
  env_name <- "r-rtdae-env"
  # Check for the existence of the specific virtualenv and enforce its prioritization if available
  if (reticulate::virtualenv_exists(env_name)) {

    reticulate::use_virtualenv(env_name, required = TRUE)
  }
  # Leverage the delay-load mechanism to safely import the scipy module at runtime
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
}
