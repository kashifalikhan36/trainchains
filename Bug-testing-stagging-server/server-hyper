import os
import io
import time
import argparse
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from datasets import load_dataset
import optuna
from optuna.trial import TrialState
import multiprocessing

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Global dataset for partitioning among clients
server_dataset = None
best_hyperparams = None

def objective(trial: optuna.Trial) -> float:
    """
    Define the objective function for Optuna hyperparameter optimization.
    This function is called repeatedly with different hyperparameters.
    """
    # Suggest hyperparameters within a broad range for optimization
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    epochs = trial.suggest_int("epochs", 1, 10)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
    max_length = trial.suggest_int("max_length", 64, 1024)

    # Simulate an evaluation metric (e.g., validation loss) for hyperparameter selection
    # Replace with actual model training and evaluation code
    score = 1.0 / (lr * epochs * batch_size * (max_length / 128))

    return score

def run_hyperparameter_optimization():
    """
    Run Optuna study to find the best hyperparameters within a 30-minute constraint.
    Maximizes CPU usage with multi-threading.
    """
    logging.info("[Optuna] Starting aggressive hyperparameter optimization...")

    # Set up the study to minimize the objective function
    study = optuna.create_study(direction="minimize")

    # Use all available CPU cores for parallel optimization
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"[Optuna] Utilizing {cpu_count} CPU cores for optimization.")

    # Run optimization with a 30-minute timeout and full CPU load
    study.optimize(objective, timeout=1800, n_jobs=cpu_count)  # 30 minutes = 1800 seconds

    # Output the best trial results
    best_trial = study.best_trial
    logging.info(f"[Optuna] Best hyperparameters found: {best_trial.params}")
    
    return best_trial.params

def run_server(port: int, expected_clients: int):
    """
    CPU-based server:
      1) Loads the dataset once globally (to partition among clients).
      2) Finds optimal hyperparameters using aggressive Optuna search.
      3) Provides dataset shards to each client upon request.
      4) Receives the final aggregated model.
    """
    global server_dataset, best_hyperparams
    logging.info("[Server] Loading dataset for partitioning...")

    # Example dataset loading: "open-r1/OpenR1-Math-220k"
    server_dataset = load_dataset("open-r1/OpenR1-Math-220k")

    # Run hyperparameter optimization
    best_hyperparams = run_hyperparameter_optimization()

    app = FastAPI()

    @app.get("/get_hyperparameters")
    async def get_hyperparameters():
        if not best_hyperparams:
            raise HTTPException(status_code=500, detail="Hyperparameters not yet optimized.")
        return best_hyperparams

    @app.get("/get_dataset_shard")
    async def get_dataset_shard(client_id: int, total_clients: int):
        """
        Returns a partition of the training dataset specific to the client_id.
        """
        if not server_dataset or "train" not in server_dataset:
            raise HTTPException(status_code=500, detail="Server dataset not initialized properly.")

        if client_id < 0 or client_id >= total_clients:
            raise HTTPException(status_code=400, detail="Invalid client_id or total_clients.")

        train_data = server_dataset["train"]
        dataset_len = len(train_data)
        shard_size = dataset_len // total_clients

        start_idx = client_id * shard_size
        end_idx = start_idx + shard_size
        if client_id == (total_clients - 1):
            end_idx = dataset_len

        subset = train_data.select(range(start_idx, end_idx))
        problems = subset["problem"]
        answers = subset["answer"]
        return {"problems": problems, "answers": answers}

    @app.post("/upload_model")
    async def upload_model(model: UploadFile = File(...)):
        contents = await model.read()
        try:
            import torch
            _ = torch.load(io.BytesIO(contents), map_location="cpu")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
        filename = f"integrated_model_{int(time.time())}.pth"
        with open(filename, "wb") as f:
            f.write(contents)
        logging.info(f"[Server] Integrated model received and saved as {filename}")
        return {"message": "Model received", "path": filename}

    logging.info(f"[Server] Starting FastAPI server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-based server for Federated Training")
    parser.add_argument("--port", type=int, default=5000, help="Port for the server")
    parser.add_argument("--expected_clients", type=int, default=2, help="Expected client count")
    args = parser.parse_args()

    run_server(args.port, args.expected_clients)
