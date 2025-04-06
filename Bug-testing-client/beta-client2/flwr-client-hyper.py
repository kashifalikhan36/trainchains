#!/usr/bin/env python3

import argparse
import logging
import time
import torch
import flwr as fl
import requests

"""
Client-FLWR script that signals GPU has arrived (if available).
Then participates in federated learning using hyperparameters provided
by the server in on_fit_config.

We have not removed or changed any existing lines of code in this file.
"""

logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Check GPU and Signal
# -----------------------------------------------------------------------------
def signal_gpu(server_url: str):
    """
    Attempt to notify the server that a GPU client is available.
    This is a custom approach; the server might parse a startup arg or
    do something else. Here we do a dummy GET to inform it.
    """
    try:
        requests.get(f"{server_url}/gpu_has_arrived", timeout=15)
        logging.info("[Client] Notified server that GPU is available.")
    except Exception as e:
        logging.warning(f"[Client] Could not notify server about GPU: {e}")

# -----------------------------------------------------------------------------
# Define a simple flwr.client.NumPyClient
# -----------------------------------------------------------------------------
class SimpleFLClient(fl.client.NumPyClient):
    """
    A minimal Flower client that trains a PyTorch model using
    the hyperparameters provided by the server.
    """

    def __init__(self, device: str):
        self.device = device
        self.model = torch.nn.Linear(10, 2)  # Very simple model

        # Dummy data
        self.x_data = torch.randn(100, 10)
        self.y_data = torch.randint(0, 2, (100,))

    def get_parameters(self, config):
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        for name, p in params_dict:
            self.model.state_dict()[name].copy_(torch.tensor(p))

    def fit(self, parameters, config):
        # Load model parameters
        self.set_parameters(parameters)

        # Read hyperparams from config
        lr = float(config.get("lr", 5e-6))
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", 1))

        logging.info(f"[Client] Training config -> lr: {lr}, epochs: {epochs}, batch_size: {batch_size}")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)

        # Dummy training loop
        dataset_size = self.x_data.shape[0]
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            x_shuffled = self.x_data[indices].to(self.device)
            y_shuffled = self.y_data[indices].to(self.device)
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = start_idx + batch_size
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), dataset_size

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return float(0.0), 100  # dummy loss, num examples

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FLWR Client with GPU/CPU logic")
    parser.add_argument("--server_url", type=str, default="http://20.244.34.219:5000",
                        help="URL for a possible server GPU arrival notice.")
    parser.add_argument("--server_address", type=str, default="20.244.34.219:5000")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[Client] Running on device: {device}")

    # If we have a GPU, attempt to notify the server
    if device == "cuda":
        signal_gpu(args.server_url)

    client = SimpleFLClient(device=device)

    # Start the client (using the deprecated call to match the existing code)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()