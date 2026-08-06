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
fit_tae <- function(x,
                    latent_dim = 2L,
                    epochs = 100L,
                    lr = 0.001,
                    lam = 1.0,
                    return_type = c("matrix", "data.frame")) {

  return_type <- match.arg(return_type)

  # 1. Validate backend initialization
  if (is.null(.pkgenv$torch) || is.null(.pkgenv$gudhi)) {
    stop("Python backend (torch/gudhi) is not initialized.")
  }

  # 2. Validate input matrix format
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

  # 3. Convert to Torch Tensor
  torch <- .pkgenv$torch
  x_tensor <- torch$tensor(x_mat, dtype = torch$float32)

  # 4. Extract dimension metadata
  input_dim  <- as.integer(ncol(x_mat))
  latent_dim <- as.integer(latent_dim)
  epochs     <- as.integer(epochs)

  # 5. Safely inject Python path without reticulate type conversion issues
  py_path <- system.file("python", package = "rtdae")
  if (py_path == "" || !file.exists(py_path)) {
    py_path <- file.path(getwd(), "inst", "python")
  }
  py_path <- normalizePath(py_path, winslash = "/", mustWork = FALSE)

  reticulate::py_run_string(sprintf(
    "import sys; p = '%s'\nif p not in sys.path: sys.path.insert(0, p)",
    py_path
  ))

  # Safely import / reload Python modules
  importlib <- reticulate::import("importlib")
  sys <- reticulate::import("sys", convert = FALSE)

  sys_modules <- reticulate::py_to_r(sys$modules)

  if ("submodules" %in% names(sys_modules)) {
    submodules_mod <- importlib$reload(sys_modules[["submodules"]])
  } else {
    submodules_mod <- reticulate::import("submodules")
  }

  if ("vanilla" %in% names(sys_modules)) {
    vanilla_mod <- importlib$reload(sys_modules[["vanilla"]])
  } else {
    vanilla_mod <- reticulate::import("vanilla")
  }

  # 6. Construct dynamic network architecture as a clean Python list
  # [input_dim, 32, 32, latent_dim]
  arch_list <- list(input_dim, 32L, 32L, latent_dim)
  ae_kwargs <- list(arch = arch_list)

  # 7. Instantiate the model
  model <- vanilla_mod$VanillaAutoencoderModel(
    autoencoder_model = "MLPAutoencoder",
    ae_kwargs         = ae_kwargs,
    lam               = lam
  )

  # 8. Optimization and training loop
  optimizer <- torch$optim$Adam(model$parameters(), lr = lr)
  loss_history <- numeric(epochs)

  for (epoch in seq_len(epochs)) {
    optimizer$zero_grad()

    res <- model$forward(x_tensor)
    if (is.list(res) || is.tuple(res)) {
      loss <- res[[1]]
    } else {
      loss <- res
    }

    loss$backward()
    optimizer$step()

    loss_history[epoch] <- loss$item()
  }

  # 9. Extract low-dimensional latent representation
  model$eval()
  with(torch$no_grad(), {
    latent_tensor <- model$encode(x_tensor)
    if (inherits(latent_tensor, "torch.Tensor")) {
      latent_np <- latent_tensor$detach()$cpu()$numpy()
    } else {
      latent_np <- latent_tensor
    }
    latent_r <- reticulate::py_to_r(latent_np)
  })

  # 10. Format output
  colnames(latent_r) <- paste0("Latent_", seq_len(latent_dim))

  if (return_type == "data.frame") {
    out <- as.data.frame(latent_r)
  } else {
    out <- latent_r
  }

  attr(out, "loss_history") <- loss_history
  attr(out, "model") <- model

  return(out)
}
