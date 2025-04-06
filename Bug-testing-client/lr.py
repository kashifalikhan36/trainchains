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
from datasets import load_dataset
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Allow dynamic GPU memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


##############################
#      LOCAL TRAINING CODE   #
##############################

def train(hyperparams):
    """
    Performs local training using your provided single-GPU training code.
    Hyperparameters (with defaults) are passed in as a dict.
    Returns the model state as a bytes object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Client] Running single GPU training on {device}...")

    # Load dataset
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    problems = dataset["problem"]
    answers = dataset["answer"]

    # Hyperparameters
    lr = hyperparams.get("lr", 2e-5)
    epochs = hyperparams.get("epochs", 2)
    batch_size = hyperparams.get("batch_size", 2)
    max_length = hyperparams.get("max_length", 128)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    # Define Dataset class
    class TextDataset(Dataset):
        def __init__(self, problems, answers, tokenizer, max_length):
            self.texts = [p + " " + a for p, a in zip(problems, answers)]
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

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
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
            if step % 10 == 0:
                logging.info(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}")

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
      For each key in the state dict, averages the corresponding tensors.
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

def run_server(port):
    """
    The server (CPU-based) provides hyperparameters and accepts the aggregated model.
    """
    app = FastAPI()
    # Define hyperparameters; these may be updated dynamically
    hyperparams = {
        "lr": 2e-5,
        "epochs": 2,
        "batch_size": 2,
        "max_length": 128
    }

    @app.get("/get_hyperparameters")
    async def get_hyperparameters():
        return hyperparams

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

    logging.info(f"[Aggregator] Starting aggregator on port {port} expecting {expected_clients} clients.\n"
                 f"Aggregated model will be sent to server at {server_addr_global}")
    uvicorn.run(app, host="0.0.0.0", port=port)


##############################
#         FASTAPI CLIENT     #
##############################

def run_client(aggregator_addr, server_addr):
    """
    The client (GPU-based) fetches hyperparameters from the server, performs local training,
    and uploads its trained model to the aggregator.
    """
    try:
        response = requests.get(f"{server_addr}/get_hyperparameters")
        hyperparams = response.json()
        logging.info(f"[Client] Received hyperparameters: {hyperparams}")
    except Exception as e:
        logging.error(f"[Client] Failed to retrieve hyperparameters: {e}")
        return

    # Perform local training using the provided training function
    model_bytes = train(hyperparams)

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
                        help="(For aggregator) Expected number of client updates before aggregation")
    args = parser.parse_args()

    if args.role == "server":
        run_server(args.port)
    elif args.role == "aggregator":
        if not args.server_addr:
            logging.error("Aggregator role requires --server_addr")
            exit(1)
        run_aggregator(args.port, args.server_addr, args.expected_clients)
    elif args.role == "client":
        if not args.aggregator_addr or not args.server_addr:
            logging.error("Client role requires --aggregator_addr and --server_addr")
            exit(1)
        run_client(args.aggregator_addr, args.server_addr)
