#!/usr/bin/env python3

import argparse
import logging
import threading
import time
import multiprocessing
import os
import json
import optuna

import torch
import flwr as fl

"""
Server-FLWR script that waits up to 30 minutes for a GPU-based client to arrive
to perform hyperparameter search. If no GPU arrives within that time, a CPU-
based hyperparameter search is performed. Once hyperparameters are determined,
they are broadcast to all clients in the on_fit_config callback.
"""

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
GPU_AVAILABLE = False          # Will be set to True if a GPU client arrives
HYPERPARAMS = None            # Dictionary of best hyperparams
HYPERPARAM_FILE = "best_hyperparams.json"
STOP_CPU_EVENT = threading.Event()

# -----------------------------------------------------------------------------
# Hyperparameter Search
# -----------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    # If GPU has arrived (STOP_CPU_EVENT is set), prune the CPU-based trial
    if STOP_CPU_EVENT.is_set():
        raise optuna.TrialPruned("Stopping CPU search; GPU client arrived.")
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 1, 10)
    score = 1.0 / (lr * epochs)  # a dummy objective
    return score

def gpu_hparam_search():
    """Perform a GPU-based hyperparameter search."""
    global HYPERPARAMS
    logging.info("[Server] Starting GPU-based hyperparameter search.")
    cpu_count = multiprocessing.cpu_count()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=600, n_jobs=cpu_count)  # 10 min timeout
    HYPERPARAMS = study.best_trial.params
    logging.info(f"[Server] GPU-based search results: {HYPERPARAMS}")
    with open(HYPERPARAM_FILE, "w") as f:
        json.dump(HYPERPARAMS, f, indent=2)

def cpu_hparam_search():
    """Perform a CPU-based hyperparameter search (up to 30 min)."""
    global HYPERPARAMS
    logging.info("[Server] Starting CPU-based hyperparameter search.")
    cpu_count = multiprocessing.cpu_count()
    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(objective, timeout=1800, n_jobs=cpu_count)  # 30 min
        HYPERPARAMS = study.best_trial.params
        logging.info(f"[Server] CPU-based search results: {HYPERPARAMS}")
        with open(HYPERPARAM_FILE, "w") as f:
            json.dump(HYPERPARAMS, f, indent=2)
    except optuna.TrialPruned:
        # If pruned, we do nothing here
        pass

def background_search_thread():
    """
    Wait up to 30 minutes for GPU arrival. If no GPU arrives in that time,
    do CPU-based hyperparam search. If GPU arrives earlier, do GPU-based.
    """
    global HYPERPARAMS
    logging.info("[Server] Waiting up to 30 minutes for GPU client to arrive...")
    # Wait 30 minutes for GPU client
    start_time = time.time()
    timeout_sec = 1800
    while time.time() - start_time < timeout_sec:
        if GPU_AVAILABLE:
            # GPU arrived: do GPU-based search
            gpu_hparam_search()
            return
        time.sleep(5)
    # If we get here, GPU never arrived in 30 minutes
    cpu_hparam_search()

# -----------------------------------------------------------------------------
# FLWR Strategy
# -----------------------------------------------------------------------------
class HyperparamAveragingStrategy(fl.server.strategy.FedAvg):
    """
    A custom FedAvg strategy that:
    - On initialization, loads or spawns hyperparameter search if needed.
    - On each fit round, passes the hyperparams to the clients in `on_fit_config`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_config_fn(self, rnd: int):
        """
        Pass the discovered hyperparameters to clients. If not available yet,
        pass defaults. 
        """
        if HYPERPARAMS is not None:
            config = {
                "lr": HYPERPARAMS.get("lr", 5e-6),
                "epochs": HYPERPARAMS.get("epochs", 1),
                "batch_size": 1,
                "max_length": 128
            }
        else:
            config = {"lr": 5e-6, "epochs": 1, "batch_size": 1, "max_length": 128}
        return config

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FLWR Server with GPU or CPU-based Hyperparameter Search")
    parser.add_argument("--gpu_has_arrived", action="store_true",
                        help="Set this flag to notify that a GPU client is already available.")
    args = parser.parse_args()

    # If a GPU is already known at startup
    global GPU_AVAILABLE
    GPU_AVAILABLE = args.gpu_has_arrived

    # If we have HYPERPARAM_FILE, load it
    global HYPERPARAMS
    if os.path.exists(HYPERPARAM_FILE):
        with open(HYPERPARAM_FILE, "r") as f:
            HYPERPARAMS = json.load(f)
        logging.info("[Server] Loaded hyperparams from file, skipping search.")
    else:
        # Start background thread to find hyperparams
        thr = threading.Thread(target=background_search_thread, daemon=True)
        thr.start()

    # Start a simple FL server with a custom strategy
    strategy = HyperparamAveragingStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:5000",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    main()