import os
import io
import time
import threading
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import uvicorn
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Allow dynamic GPU memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global dataset for the server to partition among clients
server_dataset = None


##############################
#      LOCAL TRAINING CODE   #
##############################

def train(hyperparams, problems, answers):
    """
    Performs local training on a single GPU (or CPU if CUDA not available).
    Receives:
      hyperparams (dict)
      problems (list of text entries)
      answers (list of text entries)
    Returns model state as bytes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Client] Running single GPU training on {device}...")

    # Hyperparameters
    lr = hyperparams.get("lr", 2e-5)
    epochs = hyperparams.get("epochs", 2)
    batch_size = hyperparams.get("batch_size", 2)
    max_length = hyperparams.get("max_length", 128)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    # Define local Dataset class
    class TextDataset(Dataset):
        def __init__(self, problems_list, answers_list, tokenizer, max_length):
            self.texts = [p + " " + a for p, a in zip(problems_list, answers_list)]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    # Create DataLoader
    dataset_obj = TextDataset(problems, answers, tokenizer, max_length)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4
    )

    # Model setup
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    # Variables to track training time
    total_steps = len(dataloader) * epochs
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            step_start = time.time()

            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()

            with autocast():
                outputs = model(inputs, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            # Print progress every 10 steps
            if step % 10 == 0:
                logging.info(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}")

            # Estimate remaining time
            done_steps = epoch * len(dataloader) + (step + 1)
            elapsed = time.time() - start_time
            avg_time_per_step = elapsed / done_steps
            remaining_steps = total_steps - done_steps
            remaining_time = avg_time_per_step * remaining_steps
            # Print out the approximate remaining time every 10 steps
            if step % 10 == 0:
                logging.info(f"[Client] Approx. remaining time: {remaining_time:.2f} seconds")

        logging.info(f"[Epoch {epoch}] Avg Loss: {epoch_loss/len(dataloader):.4f}")

    # Save the trained model state to an in-memory buffer and return bytes
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    logging.info("[Client] Training completed successfully")
    return buffer.getvalue()


##############################
#       AGGREGATION CODE     #
##############################

def aggregate_models(state_dicts):
    """
    Simple federated averaging:
      For each key in the state dict, average the corresponding tensors.
    """
    aggregated_state = {}
    # Initialize with zeros using the first client's state
    for key in state_dicts[0].keys():
        aggregated_state[key] = torch.zeros_like(state_dicts[0][key])

    # Sum up all client states
    for state in state_dicts:
        for key in state.keys():
            aggregated_state[key] += state[key]

    # Average
    for key in aggregated_state:
        aggregated_state[key] /= len(state_dicts)
    return aggregated_state


##############################
#        FASTAPI SERVER      #
##############################

def run_server(port, expected_clients):
    """
    CPU-based server:
      1) Loads the dataset once globally (to partition among clients).
      2) Provides hyperparameters.
      3) Provides dataset shards to each client upon request.
      4) Receives final aggregated model.
    """
    global server_dataset
    logging.info("[Server] Loading dataset for partitioning...")
    server_dataset = load_dataset("open-r1/OpenR1-Math-220k")

    app = FastAPI()

    # Define hyperparameters; these may be updated dynamically if needed
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

        # Include remainder in the last shard if dataset doesn't split evenly
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


##############################
#      FASTAPI AGGREGATOR    #
##############################

# Globals for aggregator role
client_models = []
expected_clients_global = None
server_addr_global = None

def run_aggregator(port, server_addr, expected_clients):
    """
    The aggregator (GPU-based) collects client model updates, aggregates them,
    and sends the integrated model to the server.
    """
    global expected_clients_global, server_addr_global
    expected_clients_global = expected_clients
    server_addr_global = server_addr

    app = FastAPI()

    @app.post("/upload_client_model")
    async def upload_client_model(model: UploadFile = File(...)):
        global client_models
        contents = await model.read()
        try:
            state_dict = torch.load(io.BytesIO(contents), map_location="cpu")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load client model: {e}")
        client_models.append(state_dict)
        logging.info(f"[Aggregator] Received client model ({len(client_models)}/{expected_clients_global})")
        if len(client_models) >= expected_clients_global:
            threading.Thread(target=aggregate_and_send).start()
        return {"message": "Client model received"}

    def aggregate_and_send():
        global client_models, server_addr_global
        logging.info("[Aggregator] Aggregating client models...")
        aggregated_state = aggregate_models(client_models)
        buffer = io.BytesIO()
        torch.save(aggregated_state, buffer)
        buffer.seek(0)
        url = f"{server_addr_global}/upload_model"
        files = {"model": ("aggregated_model.pth", buffer.getvalue())}
        try:
            response = requests.post(url, files=files)
            logging.info(f"[Aggregator] Aggregated model sent to server: {response.json()}")
        except Exception as e:
            logging.error(f"[Aggregator] Failed to send aggregated model: {e}")
        client_models = []

    logging.info(
        f"[Aggregator] Starting aggregator on port {port} expecting {expected_clients} clients.\n"
        f"Aggregated model will be sent to server at {server_addr_global}"
    )
    uvicorn.run(app, host="0.0.0.0", port=port)


##############################
#         FASTAPI CLIENT     #
##############################

def run_client(aggregator_addr, server_addr, client_id, total_clients):
    """
    Client (GPU-based):
      1) Fetches hyperparameters from the server.
      2) Requests its dataset shard from the server.
      3) Performs local training on that shard.
      4) Uploads the trained model to the aggregator.
    """
    try:
        response = requests.get(f"{server_addr}/get_hyperparameters")
        hyperparams = response.json()
        logging.info(f"[Client] Received hyperparameters: {hyperparams}")
    except Exception as e:
        logging.error(f"[Client] Failed to retrieve hyperparameters: {e}")
        return

    # Request dataset shard
    try:
        shard_resp = requests.get(
            f"{server_addr}/get_dataset_shard",
            params={"client_id": client_id, "total_clients": total_clients}
        )
        shard_data = shard_resp.json()
        problems = shard_data["problems"]
        answers = shard_data["answers"]
        logging.info(f"[Client] Received dataset shard: {len(problems)} samples")
    except Exception as e:
        logging.error(f"[Client] Failed to retrieve dataset shard: {e}")
        return

    # Perform local training
    model_bytes = train(hyperparams, problems, answers)

    # Upload the model to the aggregator
    buffer = io.BytesIO(model_bytes)
    url = f"{aggregator_addr}/upload_client_model"
    files = {"model": ("client_model.pth", buffer.getvalue())}
    try:
        resp = requests.post(url, files=files)
        logging.info(f"[Client] Uploaded model to aggregator: {resp.json()}")
    except Exception as e:
        logging.error(f"[Client] Failed to upload model to aggregator: {e}")


##############################
#            MAIN            #
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Training with FastAPI Demo using single GPU training code"
    )
    parser.add_argument("--role", type=str, required=True,
                        choices=["server", "aggregator", "client"],
                        help="Role to run: 'server', 'aggregator', or 'client'")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for the server or aggregator")
    parser.add_argument("--aggregator_addr", type=str,
                        help="For client role: aggregator's address (e.g., http://<aggregator_ip>:<port>)")
    parser.add_argument("--server_addr", type=str,
                        help="For client/aggregator role: server's address (e.g., http://<server_ip>:<port>)")
    parser.add_argument("--expected_clients", type=int, default=2,
                        help="(For aggregator/server) Expected number of client updates before aggregation")
    parser.add_argument("--client_id", type=int, default=0,
                        help="(For client) Client index ID (0-based)")
    parser.add_argument("--total_clients", type=int, default=2,
                        help="(For client) Total number of clients participating")
    args = parser.parse_args()

    if args.role == "server":
        run_server(args.port, args.expected_clients)
    elif args.role == "aggregator":
        if not args.server_addr:
            logging.error("Aggregator role requires --server_addr")
            exit(1)
        run_aggregator(args.port, args.server_addr, args.expected_clients)
    elif args.role == "client":
        if not args.aggregator_addr or not args.server_addr:
            logging.error("Client role requires --aggregator_addr and --server_addr")
            exit(1)
        run_client(args.aggregator_addr, args.server_addr, args.client_id, args.total_clients)
