#' Initialize Vanilla Autoencoder Model
#'
#' @param input_dim An integer specifying the dimension of the input data.
#' @param latent_dim An integer specifying the dimension of the latent space.
#' @return A PyTorch Autoencoder model object.
#' @export
run_tae <- function(input_dim = 10L, latent_dim = 2L) {

  # Ensure Python 'torch' dependency is met
  if (!reticulate::py_module_available("torch")) {
    stop("Python dependency 'torch' not found. Please install PyTorch in your active environment.")
  }

  if (!reticulate::py_module_available("aleph")) {
    stop("Python 'aleph' module is not installed. Please install it first.")
  }

  vanilla <- reticulate::import_from_path("vanilla", path = system.file("python", package = "rtdae"))
  topology <- reticulate::import_from_path("topology", path = system.file("python", package = "rtdae"))

  # Resolve the absolute path to python source files within the R package inst/ directory
  python_path <- system.file("python", package = "rtdae")
  if (python_path == "") {
    stop("Failed to locate Python sources inside the 'rtdae' package. Please reinstall the package.")
  }

  # Import the local 'vanilla' module
  vanilla_mod <- reticulate::import_from_path("vanilla", path = python_path)

  # Instantiate the Autoencoder class from python module
  model <- vanilla_mod$Autoencoder(input_dim = as.integer(input_dim),
                                   latent_dim = as.integer(latent_dim))

  return(model)
}
