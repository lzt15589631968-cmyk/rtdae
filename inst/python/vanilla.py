"""Vanilla models."""
import os
import sys
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import submodules
from base import AutoencoderModel


class VanillaAutoencoderModel(AutoencoderModel):
    def __init__(self, autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None, lam=1.0, **kwargs):
        super().__init__()
        self.lam = lam
        ae_kwargs = ae_kwargs if ae_kwargs is not None else {}
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)

    def forward(self, x):
        return self.autoencoder(x)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)
