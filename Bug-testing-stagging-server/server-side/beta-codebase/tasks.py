import time
import optuna
import torch
import logging
import json
import multiprocessing

from globals import (
    stop_cpu_search_event,
    hyperparam_lock,
    cpu_pruned_once,
    gpu_has_arrived_event,
    best_hyperparams,
    hyperparams_file
)

def objective(trial: optuna.Trial) -> float:
    global cpu_pruned_once
    if stop_cpu_search_event.is_set():
        if not cpu_pruned_once:
            logging.info("[Server] CPU-based search pruned because GPU arrived.")
            cpu_pruned_once = True
        raise optuna.TrialPruned("CPU-based search stopped; GPU arrived.")
    lr = trial.suggest_float("lr", 1, 2, log=True)
    epochs = trial.suggest_int("epochs", 1, 2)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    max_length = trial.suggest_int("max_length", 64, 65)
    device = "cuda" if (torch.cuda.is_available() and gpu_has_arrived_event.is_set()) else "cpu"
    dummy_tensor = torch.rand(100, 100, device=device)
    for _ in range(epochs):
        if stop_cpu_search_event.is_set():
            if not cpu_pruned_once:
                logging.info("[Server] CPU-based search pruned mid-run because GPU arrived.")
                cpu_pruned_once = True
            raise optuna.TrialPruned("CPU-based search pruned mid-run; GPU arrived.")
        dummy_tensor *= lr
    score = 1.0 / (lr * epochs * batch_size * (max_length / 128.0))
    return score

def run_optuna_search(use_gpu_first: bool):
    global best_hyperparams

    if best_hyperparams is not None:
        logging.info("[Server] Hyperparameters already set. Skipping Optuna search.")
        return

    def do_study(mode: str):
        logging.info(f"[Server] Starting Optuna study on {mode.upper()}.")
        study = optuna.create_study(direction="minimize")
        cpu_count = multiprocessing.cpu_count()
        study.optimize(objective, timeout=5, n_jobs=cpu_count) # Replace objective with your actual objective function
        return study.best_trial

    def save_best_hyperparams_to_json(hyperparams):
        hyperparams_file = "best_hyperparam.json" # Define the file path for saving
        try:
            with open(hyperparams_file, "w") as f:
                json.dump(hyperparams, f, indent=4)
            logging.info(f"[Server] Best hyperparameters saved to {hyperparams_file}")
        except Exception as e:
            logging.error(f"[Server] Error saving hyperparameters to JSON: {e}")

    if use_gpu_first and torch.cuda.is_available():
        with hyperparam_lock:
            trial = do_study("gpu")
            best_hyperparams = trial.params
            save_best_hyperparams_to_json(best_hyperparams) # Save best_hyperparams to JSON here
    else:
        stop_cpu_search_event.clear()
        try:
            with hyperparam_lock:
                trial = do_study("cpu")
                best_hyperparams = trial.params
            save_best_hyperparams_to_json(best_hyperparams) # Save best_hyperparams to JSON here
        except optuna.TrialPruned:
            logging.info("[Server] CPU-based search pruned. Waiting 30 minutes for GPU-based hyperparams.")
            time.sleep(30 * 60)
            if best_hyperparams is None:
                logging.info("[Server] No GPU-based hyperparams arrived in 30 minutes. Retrying CPU-based search.")
                stop_cpu_search_event.clear()
                with hyperparam_lock:
                    trial = do_study("cpu")
                    best_hyperparams = trial.params
                    save_best_hyperparams_to_json(best_hyperparams) # Save best_hyperparams to JSON here


def background_wait_and_optimize():
    global best_hyperparams
    import threading

    if best_hyperparams is not None:
        logging.info("[Background] Hyperparameters already exist, skipping search.")
        return
    logging.info("[Background] Waiting up to 30 minutes for GPU before starting hyperparam search...")
    got_gpu = gpu_has_arrived_event.wait(timeout=1800)
    if got_gpu:
        run_optuna_search(use_gpu_first=True)
    else:
        run_optuna_search(use_gpu_first=False)