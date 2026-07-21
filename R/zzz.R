# Global references to Python modules (will be initialized in .onLoad)
scipy <- NULL
gd <- NULL
torch <- NULL
.onLoad <- function(libname, pkgname) {
  env_name <- "r-rtdae-env"
  # Configure isolated virtual environment if it exists
  if (reticulate::virtualenv_exists(env_name)) {
    reticulate::use_virtualenv(env_name, required = TRUE)
  }
  # Superassignment to bind python modules with delay-loading enabled
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  gd <<- reticulate::import("gudhi", delay_load = TRUE)
  torch <<- reticulate::import("torch", delay_load = TRUE)
}
