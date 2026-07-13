# Create a global variable at the top level to reserve a space for the Python module (set to NULL initially)
scipy <- NULL
# Reserve a global variable to hold the reference to the Python 'gudhi' module
gd <- NULL
# Automatically set up the Python environment when the R package is loaded
.onLoad <- function(libname, pkgname) {
  # Define our package's specific virtual environment name
  env_name <- "r-rtdae-env"
  # If our specific virtual environment already exists, force R to use it first
  if (reticulate::virtualenv_exists(env_name)) {

    reticulate::use_virtualenv(env_name, required = TRUE)
  }
  # Safely load scipy only when a user actually calls a function (Delay-Loading)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  # Lazily import 'gudhi' for Topological Data Analysis (TDA), loaded only when accessed
  gd <<- reticulate::import("gudhi", delay_load = TRUE)
}
