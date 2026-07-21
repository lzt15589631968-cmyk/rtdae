"""Submodules used by models."""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from base import AutoencoderModel



class ConvolutionalAutoencoder(AutoencoderModel):
    """Convolutional Autoencoder for MNIST/Fashion MNIST."""

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh(),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Print(nn.Module):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def forward(self, x):
        print(self.name, x.size())
        return x


class ConvolutionalAutoencoder_2D(AutoencoderModel):
    def __init__(self, input_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, 9, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 4, 9, stride=1, padding=0),
            nn.ReLU(),
            View((-1, 576)),
            nn.Linear(576, 250),
            nn.ReLU(),
            nn.Linear(250, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 250),
            nn.ReLU(),
            View((-1, 250, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(250, 4, 12, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 17, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, stride=1, padding=0),
            nn.Tanh(),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class DeepAE(AutoencoderModel):
    def __init__(self, input_dims=(1, 28, 28)):
        super().__init__()
        self.input_dims = input_dims
        n_input_dims = np.prod(input_dims)
        self.encoder = nn.Sequential(
            View((-1, n_input_dims)),
            nn.Linear(n_input_dims, 1000),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 250),
            nn.ReLU(True),
            nn.BatchNorm1d(250),
            nn.Linear(250, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 250),
            nn.ReLU(True),
            nn.BatchNorm1d(250),
            nn.Linear(250, 500),
            nn.ReLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 1000),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, n_input_dims),
            View((-1,) + tuple(input_dims)),
            nn.Tanh(),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class LinearAE(AutoencoderModel):
    def __init__(self, input_dims=(1, 28, 28)):
        super().__init__()
        self.input_dims = input_dims
        n_input_dims = np.prod(input_dims)
        self.encoder = nn.Sequential(
            View((-1, n_input_dims)), nn.Linear(n_input_dims, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, n_input_dims),
            View((-1,) + tuple(input_dims)),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class ConvAE_CIFAR(AutoencoderModel):
    def __init__(self, latent_linear_dim=2, topo_latent_dim=None):
        self.latent_linear_dim = latent_linear_dim
        super().__init__()
        if latent_linear_dim is not None:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 12, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, 48, 4, stride=2, padding=1),
                nn.ReLU(),
                View((-1, 768)),
                nn.Linear(768, 250),
                nn.ReLU(True),
                nn.BatchNorm1d(250),
                nn.Linear(250, latent_linear_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_linear_dim, 250),
                nn.ReLU(True),
                nn.BatchNorm1d(250),
                nn.Linear(250, 768),
                nn.ReLU(True),
                nn.BatchNorm1d(768),
                View((-1, 48, 4, 4)),
                nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 12, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, 48, 4, stride=2, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        if isinstance(latent, tuple):
            latent = latent[0]
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class MLPAutoencoder(AutoencoderModel):
    def __init__(self, arch=[3, 32, 32, 2]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 3),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class MLPAutoencoder_Spheres(AutoencoderModel):
    def __init__(self, arch=[3, 32, 32, 2]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(101, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 101),
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class LinearAE_Spheres(AutoencoderModel):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(101, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 101))
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class MLPVAE(AutoencoderModel):
    def __init__(self, input_dim=3, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, input_dim * 2),
        )
        self.reconst_error = nn.MSELoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _split_to_parameters(self, x):
        return x.split(x.size(-1) // 2, dim=-1)

    def encode(self, x):
        mu, logvar = self._split_to_parameters(self.encoder(x))
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        mu, logvar = self._split_to_parameters(self.decoder(z))
        return self.reparameterize(mu, logvar)

    def log_likelihood(self, x, reconst_mean, reconst_std):
        predicted_x = Normal(loc=reconst_mean, scale=reconst_std)
        return predicted_x.log_prob(x).sum(dim=-1)

    def forward(self, x):
        latent_mu, latent_logvar = self._split_to_parameters(self.encoder(x))
        latent = self.reparameterize(latent_mu, latent_logvar)

        data_mu, data_logvar = self._split_to_parameters(self.decoder(latent))
        data_std = torch.exp(0.5 * data_logvar)

        likelihood = -self.log_likelihood(x, data_mu, data_std).mean(dim=0)

        kl_div = -0.5 * torch.sum(
            1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp(),
            dim=-1,
        ).mean(dim=0)
        loss = likelihood + kl_div
        reconst_error = self.reconst_error(x, data_mu)
        return loss, {
            'loss.likelihood': likelihood,
            'loss.kl_divergence': kl_div,
            'reconstruction_error': reconst_error,
        }
