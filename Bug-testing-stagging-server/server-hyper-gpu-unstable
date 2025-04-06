#!/usr/bin/env python3

import os
import io
import time
import json
import argparse
import logging
import threading
import multiprocessing

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from datasets import load_dataset
import optuna
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Global variables and aggregator state logic
server_dataset = None
best_hyperparams = None
hyperparams_file = "best_hyperparams.json"
aggregator_state_file = "aggregator_state.json"

gpu_has_arrived_event = threading.Event()
stop_cpu_search_event = threading.Event()

hyperparam_lock = threading.Lock()

# Prevent repeated logging of prune message
cpu_pruned_once = False

def create_or_update_state_file(num_clients: int = 2):
    state = {
        "no_of_slot_available": num_clients,
        "active_slot": 0
    }
    with open(aggregator_state_file, "w") as f:
        json.dump(state, f, indent=2)

def update_state_on_gpu_start():
    if not os.path.exists(aggregator_state_file):
        return
    with open(aggregator_state_file, "r") as f:
        state = json.load(f)
    if state["no_of_slot_available"] > 0:
        state["active_slot"] = 1
        state["no_of_slot_available"] -= 1
    with open(aggregator_state_file, "w") as f:
        json.dump(state, f, indent=2)

def update_state_on_gpu_finish():
    if not os.path.exists(aggregator_state_file):
        return
    with open(aggregator_state_file, "r") as f:
        state = json.load(f)
    state["active_slot"] = 0
    state["no_of_slot_available"] += 1
    with open(aggregator_state_file, "w") as f:
        json.dump(state, f, indent=2)

# Optuna logic
def objective(trial: optuna.Trial) -> float:
    global cpu_pruned_once

    if stop_cpu_search_event.is_set():
        if not cpu_pruned_once:
            logging.info("[Server] CPU-based search pruned because GPU arrived. Waiting up to 30 minutes for GPU-based hyperparams...")
            cpu_pruned_once = True
        raise optuna.TrialPruned("CPU-based search stopped; GPU arrived.")

    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 1, 10)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
    max_length = trial.suggest_int("max_length", 64, 1024)

    device = "cuda" if (torch.cuda.is_available() and gpu_has_arrived_event.is_set()) else "cpu"

    if device == "cuda":
        update_state_on_gpu_start()

    dummy_tensor = torch.rand(100, 100, device=device)
    for _ in range(epochs):
        if stop_cpu_search_event.is_set():
            if not cpu_pruned_once:
                logging.info("[Server] CPU-based search pruned mid-run because GPU arrived.")
                cpu_pruned_once = True
            raise optuna.TrialPruned("CPU-based search pruned mid-run; GPU arrived.")
        dummy_tensor *= lr

    score = 1.0 / (lr * epochs * batch_size * (max_length / 128))

    if device == "cuda":
        update_state_on_gpu_finish()

    return score

def run_optuna_search(use_gpu_first: bool):
    global best_hyperparams

    def do_study(mode: str):
        logging.info(f"[Server] Starting Optuna study on {mode.upper()}.")
        study = optuna.create_study(direction="minimize")
        cpu_count = multiprocessing.cpu_count()
        study.optimize(objective, timeout=1800, n_jobs=cpu_count) # 30-minute total
        return study.best_trial

    if use_gpu_first and torch.cuda.is_available():
        with hyperparam_lock:
            trial = do_study("gpu")
            best_hyperparams = trial.params

    else:
        stop_cpu_search_event.clear()
        try:
            with hyperparam_lock:
                trial = do_study("cpu")
                best_hyperparams = trial.params
        except optuna.TrialPruned:
            logging.info("[Server] CPU-based search pruned. Waiting 30 minutes for GPU-based hyperparameters.")
            time.sleep(30 * 60) # Wait 30 minutes
            if best_hyperparams is None:
                logging.info("[Server] No GPU-based hyperparams found in 30 minutes. Retrying CPU-based search.")
                stop_cpu_search_event.clear()
                with hyperparam_lock:
                    trial = do_study("cpu")
                    best_hyperparams = trial.params

# Background thread
def background_wait_and_optimize():
    logging.info("[Background] Waiting up to 30 minutes for GPU to arrive before starting hyperparam search...")
    got_gpu = gpu_has_arrived_event.wait(timeout=1800)  # 30 minutes
    if got_gpu:
        run_optuna_search(use_gpu_first=True)
    else:
        run_optuna_search(use_gpu_first=False)

# FastAPI Application
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global server_dataset
    create_or_update_state_file(2)
    logging.info("[Server] Loading dataset at startup...")
    server_dataset = load_dataset("open-r1/OpenR1-Math-220k")
    logging.info("[Server] Dataset loaded successfully.")
    thread = threading.Thread(target=background_wait_and_optimize, daemon=True)
    thread.start()

@app.get("/gpu_has_arrived")
def gpu_has_arrived():
    gpu_has_arrived_event.set()
    stop_cpu_search_event.set()
    return {"message": "GPU acknowledged. CPU search will be stopped if running."}

@app.post("/submit_hyperparameters")
async def submit_hyperparameters(hyperparams: dict):
    global best_hyperparams
    best_hyperparams = hyperparams
    with open(hyperparams_file, "w") as f:
        json.dump(hyperparams, f, indent=2)
    logging.info(f"[Server] Received and saved hyperparameters: {hyperparams}")
    return {"message": "Hyperparameters received"}

@app.get("/get_hyperparameters")
def get_hyperparameters():
    global best_hyperparams
    if best_hyperparams is None:
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, "r") as f:
                best_hyperparams = json.load(f)
        else:
            raise HTTPException(status_code=503, detail="Hyperparameters not yet available.")
    return best_hyperparams

@app.get("/get_dataset_shard")
def get_dataset_shard(client_id: int, total_clients: int):
    if not server_dataset or "train" not in server_dataset:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
    if client_id < 0 or client_id >= total_clients:
        raise HTTPException(status_code=400, detail="Invalid client_id or total_clients.")

    train_data = server_dataset["train"]
    data_len = len(train_data)
    shard_size = data_len // total_clients
    start_idx = client_id * shard_size
    end_idx = start_idx + shard_size
    if client_id == (total_clients - 1):
        end_idx = data_len

    subset = train_data.select(range(start_idx, end_idx))
    return {"problems": subset["problem"], "answers": subset["answer"]}

@app.post("/upload_model")
async def upload_model(model: UploadFile = File(...)):
    contents = await model.read()
    try:
        _ = torch.load(io.BytesIO(contents), map_location="cpu")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
    filename = f"integrated_model_{int(time.time())}.pth"
    with open(filename, "wb") as f:
        f.write(contents)
    logging.info(f"[Server] Integrated model received and saved as {filename}")
    return {"message": "Model received", "path": filename}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server with optional GPU-based training, single prune message")
    parser.add_argument("--port", type=int, default=5000, help="Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
