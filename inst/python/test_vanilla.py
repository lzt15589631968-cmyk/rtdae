"""Standalone test script for ConvolutionalAutoencoder."""
import os
import sys

# Safe path injection
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath("inst/python")

if current_dir not in sys.path:
    sys.path.append(current_dir)

import torch
import torch.optim as optim
from submodules import ConvolutionalAutoencoder
from topology import PersistentHomologyCalculation


def main():
    # 1. Generate synthetic input tensor X (batch_size=16, channel=1, H=28, W=28)
    X = torch.randn(16, 1, 28, 28)

    # 2. Instantiate model and optimizer
    model = ConvolutionalAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Fit model on data X
    model.train()
    for epoch in range(1, 6):
        optimizer.zero_grad()
        # forward returns (loss, info_dict)
        loss, _ = model(X)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/5] Loss: {loss.item():.6f}")

    # 4. Latent space extraction & persistent homology calculation
    model.eval()
    with torch.no_grad():
        latent = model.encode(X)
        flattened_z = latent.view(X.size(0), -1)
        dist_matrix = torch.cdist(flattened_z, flattened_z).numpy()

        ph_calc = PersistentHomologyCalculation()
        pairs_0, _ = ph_calc(dist_matrix)
        print(f"Persistence pairs calculated: {len(pairs_0)}")


if __name__ == "__main__":
    main()
