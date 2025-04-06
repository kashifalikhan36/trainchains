import os
import io
import time
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Check for GPU availability
if not torch.cuda.is_available():
    print("Not compatible - No CUDA Based GPU Found")
    exit(1)

##############################
#      LOCAL TRAINING CODE   #
##############################

def train(hyperparams, problems, answers):
    """
    Performs local training on a GPU.
    If no CUDA GPU is detected, the script exits above.
    """
    device = torch.device("cuda")
    logging.info(f"[Client] Running single GPU training on {device}...")

    lr = hyperparams.get("lr", 2e-5)
    epochs = hyperparams.get("epochs", 2)
    batch_size = hyperparams.get("batch_size", 2)
    max_length = hyperparams.get("max_length", 128)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

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

    dataset_obj = TextDataset(problems, answers, tokenizer, max_length)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    total_steps = len(dataloader) * epochs
    start_time = time.time()

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

            # Estimate remaining training time
            steps_done = (epoch * len(dataloader)) + (step + 1)
            elapsed_time = time.time() - start_time
            avg_time_per_step = elapsed_time / steps_done if steps_done else 0
            steps_left = total_steps - steps_done
            remaining_time = steps_left * avg_time_per_step
            if step % 10 == 0:
                logging.info(f"[Client] Approx. remaining time: {remaining_time:.2f} seconds")

        logging.info(f"[Epoch {epoch}] Avg Loss: {epoch_loss / len(dataloader):.4f}")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    logging.info("[Client] Training completed successfully")
    return buffer.getvalue()


##############################
#         FASTAPI CLIENT     #
##############################

def run_client(aggregator_addr: str, server_addr: str, client_id: int, total_clients: int):
    """
    Steps:
      1) Fetch hyperparameters from the server.
      2) Request dataset shard from the server.
      3) Train locally on GPU.
      4) Upload model to the aggregator.
    """
    # Fetch hyperparameters
    try:
        response = requests.get(f"{server_addr}/get_hyperparameters")
        hyperparams = response.json()
        logging.info(f"[Client] Hyperparameters: {hyperparams}")
    except Exception as e:
        logging.error(f"[Client] Could not retrieve hyperparameters: {e}")
        return

    # Get dataset shard
    try:
        shard_resp = requests.get(
            f"{server_addr}/get_dataset_shard",
            params={"client_id": client_id, "total_clients": total_clients}
        )
        shard_data = shard_resp.json()
        problems = shard_data["problems"]
        answers = shard_data["answers"]
        logging.info(f"[Client] Received dataset shard with {len(problems)} samples")
    except Exception as e:
        logging.error(f"[Client] Failed to retrieve dataset shard: {e}")
        return

    # Train locally
    model_bytes = train(hyperparams, problems, answers)

    # Upload trained model to aggregator
    buffer = io.BytesIO(model_bytes)
    url = f"{aggregator_addr}/upload_client_model"
    files = {"model": ("client_model.pth", buffer.getvalue())}

    try:
        resp = requests.post(url, files=files)
        logging.info(f"[Client] Model upload response: {resp.json()}")
    except Exception as e:
        logging.error(f"[Client] Failed to upload model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-based client for Federated Training")
    parser.add_argument("--aggregator_addr", type=str, required=True, help="Aggregator address (http://<ip>:<port>)")
    parser.add_argument("--server_addr", type=str, required=True, help="Server address (http://<ip>:<port>)")
    parser.add_argument("--client_id", type=int, default=0, help="Client index ID (0-based)")
    parser.add_argument("--total_clients", type=int, default=2, help="Total number of clients")
    args = parser.parse_args()

    run_client(
        aggregator_addr=args.aggregator_addr,
        server_addr=args.server_addr,
        client_id=args.client_id,
        total_clients=args.total_clients
    )
