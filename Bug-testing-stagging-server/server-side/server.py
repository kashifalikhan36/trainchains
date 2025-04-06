import os
import io
import time
import argparse
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from datasets import load_dataset

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Global dataset for partitioning among clients
server_dataset = None

def run_server(port: int, expected_clients: int):
    """
    CPU-based server:
      1) Loads the dataset once globally (to partition among clients).
      2) Provides hyperparameters.
      3) Provides dataset shards to each client upon request.
      4) Receives final aggregated model.
    """
    global server_dataset
    logging.info("[Server] Loading dataset for partitioning...")

    # Example dataset loading: "open-r1/OpenR1-Math-220k"
    # Replace as needed for real training tasks
    server_dataset = load_dataset("open-r1/OpenR1-Math-220k")

    app = FastAPI()

    # Define hyperparameters
    hyperparams = {
        "lr": 2e-5,
        "epochs": 2,
        "batch_size": 2,
        "max_length": 128
    }

    @app.get("/get_hyperparameters")
    async def get_hyperparameters():
        return hyperparams

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
        # Last shard grabs remainder if needed
        if client_id == (total_clients - 1):
            end_idx = dataset_len

        subset = train_data.select(range(start_idx, end_idx))
        problems = subset["problem"]
        answers = subset["answer"]
        return {"problems": problems, "answers": answers}

    @app.post("/upload_model")
    async def upload_model(model: UploadFile = File(...)):
        # Accept the final integrated model
        contents = await model.read()
        try:
            import torch  # Validate model
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
