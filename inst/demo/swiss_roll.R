# Load required libraries
library(rtdae)
library(ggplot2)

# Set seed for reproducibility
set.seed(42)

# Generate 3D Swiss Roll dataset
n_points <- 1000
t <- 1.5 * pi * (1 + 2 * runif(n_points))
height <- 30 * runif(n_points)

x <- t * cos(t)
y <- height
z <- t * sin(t)

swiss_roll_mat <- cbind(x, y, z)

# Train Topological Autoencoder (3D -> 2D)
cat("Training TAE on Swiss Roll dataset...\n")
res_swiss <- fit_tae(
  x = swiss_roll_mat,
  latent_dim = 2L,
  epochs = 200L,
  lr = 0.005,
  lam = 1.0,
  return_type = "data.frame"
)


# Attach manifold parameter for visualization
res_swiss$manifold_t <- t

# Plot 2D Latent Representation
ggplot(res_swiss, aes(x = Latent_1, y = Latent_2, color = manifold_t)) +
  geom_point(size = 1.8, alpha = 0.8) +
  scale_color_viridis_c(option = "magma") +
  theme_minimal() +
  labs(
    title = "Swiss Roll Latent Representation (rtdae)",
    x = "Latent Dimension 1",
    y = "Latent Dimension 2",
    color = "Manifold Angle (t)"
  )
