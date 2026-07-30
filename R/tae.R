#' Fit Topological Autoencoder (TAE)
#'
#' Trains a topological autoencoder on an R matrix or data.frame to perform
#' topology-preserving dimensionality reduction.
#'
#' @param x A numeric matrix or data.frame representing input data (samples x features).
#' @param latent_dim Integer. The dimension of the low-dimensional latent space (default: 2).
#' @param epochs Integer. Number of training epochs (default: 100).
#' @param lr Numeric. Learning rate for the Adam optimizer (default: 0.001).
#' @param lam Numeric. Weight parameter for the topological loss term (default: 1.0).
#' @param return_type Character. Output format for the low-dimensional representation:
#'   \code{"matrix"} (default) or \code{"data.frame"}.
#'
#' @return A numeric \code{matrix} or \code{data.frame} of dimensions \code{c(nrow(x), latent_dim)}
#'   containing the low-dimensional latent representations, with model training history attached.
#' @export
#'
#' @examples
#' \dontrun{
#' library(rtdae)
#' mat <- matrix(rnorm(100 * 10), nrow = 100, ncol = 10)
#' res <- fit_tae(mat, latent_dim = 2, epochs = 20)
#' head(res)
#' }
fit_tae <- function(x,
                    latent_dim = 2L,
                    epochs = 100L,
                    lr = 0.001,
                    lam = 1.0,
                    return_type = c("matrix", "data.frame")) {

  return_type <- match.arg(return_type)

  # Validate backend initialization
  if (is.null(.pkgenv$torch) || is.null(.pkgenv$gudhi)) {
    stop("Python backend (torch/gudhi) is not initialized. Please re-load the library or check environment setup.")
  }

  # Input validation and matrix coercion
  if (is.data.frame(x)) {
    x_mat <- as.matrix(x)
  } else if (is.matrix(x)) {
    x_mat <- x
  } else {
    stop("Input 'x' must be a matrix or a data.frame.")
  }

  if (!is.numeric(x_mat)) {
    stop("Input 'x' must contain numeric values.")
  }

  # Tensor allocation from native R matrix
  torch <- .pkgenv$torch
  x_tensor <- torch$tensor(x_mat, dtype = torch$float32)

  # Cast dimensions and import python model architecture
  input_dim  <- as.integer(ncol(x_mat))
  latent_dim <- as.integer(latent_dim)
  epochs     <- as.integer(epochs)

  py_path <- system.file("python", package = "rtdae")
  if (py_path == "") {
    py_path <- file.path(getwd(), "inst", "python")
  }

  vanilla_mod <- reticulate::import_from_path("vanilla", path = py_path)
  # Construct network architecture vector: input_dim -> 32 -> 32 -> latent_dim
  arch_vec <- as.integer(c(input_dim, 32, 32, latent_dim))

  # Pack model kwargs for VanillaAutoencoderModel instantiation
  ae_kwargs <- list(
    arch = arch_vec
  )

  # Instantiate autoencoder model backend
  model <- vanilla_mod$VanillaAutoencoderModel(
    autoencoder_model = "MLPAutoencoder",
    ae_kwargs         = ae_kwargs
  )

  # Execution loop for optimization
  optimizer <- torch$optim$Adam(model$parameters(), lr = lr)
  loss_history <- numeric(epochs)

  for (epoch in seq_len(epochs)) {
    optimizer$zero_grad()

    # Forward pass: joint reconstruction and topological persistent homology loss
    res <- model$forward(x_tensor)
    loss <- res[[1]]  # Extract total objective loss tensor

    # Backward pass & parameter update
    loss$backward()
    optimizer$step()

    loss_history[epoch] <- loss$item()
  }

  # Extract low-dimensional latent representation
  model$eval()
  with(torch$no_grad(), {
    latent_tensor <- model$encode(x_tensor)
    latent_r <- reticulate::py_to_r(latent_tensor$numpy())
  })

  # Format output object
  colnames(latent_r) <- paste0("Latent_", seq_len(latent_dim))

  if (return_type == "data.frame") {
    out <- as.data.frame(latent_r)
  } else {
    out <- latent_r
  }

  # Attach loss history and trained model object as attributes
  attr(out, "loss_history") <- loss_history
  attr(out, "model") <- model

  return(out)
}
