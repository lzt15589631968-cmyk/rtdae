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


class MLPAutoencoder(AutoencoderModel):
    def __init__(self, arch=[3, 32, 32, 2], **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(arch[0], arch[1]),
            nn.ReLU(True),
            nn.BatchNorm1d(arch[1]),
            nn.Linear(arch[1], arch[2]),
            nn.ReLU(True),
            nn.BatchNorm1d(arch[2]),
            nn.Linear(arch[2], arch[3]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(arch[3], arch[2]),
            nn.ReLU(True),
            nn.BatchNorm1d(arch[2]),
            nn.Linear(arch[2], arch[1]),
            nn.ReLU(True),
            nn.BatchNorm1d(arch[1]),
            nn.Linear(arch[1], arch[0]),
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
